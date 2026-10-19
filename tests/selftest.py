#!/usr/bin/env python
"""Self-test for the reorganised AURORA code system.

Runs with the Python standard library only, so it can be executed before any
dependency is installed.  Every check maps to one of the six reorganisation
requirements.

    python tests/selftest.py
    python tests/selftest.py -v            # show every check
    python tests/selftest.py --keep        # keep the temporary project folder
"""

from __future__ import annotations

import argparse
import ast
import csv
import importlib
import re
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Callable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"

_results: List[Tuple[str, str, str]] = []
_registry: List[Tuple[str, Callable, bool]] = []
_verbose = False


# --------------------------------------------------------------------------- #
# Tiny test harness
# --------------------------------------------------------------------------- #


def check(name: str, requirement: str = "") -> Callable:
    """Register a test.  ``use_tmp=True`` tests receive the scratch directory."""

    def decorator(function: Callable) -> Callable:
        import inspect

        wants_tmp = bool(inspect.signature(function).parameters)
        label = f"{name}  [{requirement}]" if requirement else name
        _registry.append((label, function, wants_tmp))
        return function

    return decorator


def _run_registered(temp_dir: Path) -> None:
    for label, function, wants_tmp in _registry:
        try:
            detail = function(temp_dir) if wants_tmp else function()
        except AssertionError as exc:
            _results.append((label, FAIL, str(exc) or "assertion failed"))
            if _verbose:
                traceback.print_exc()
        except Exception as exc:  # noqa: BLE001
            _results.append((label, FAIL, f"{type(exc).__name__}: {exc}"))
            if _verbose:
                traceback.print_exc()
        else:
            if detail == SKIP:
                _results.append((label, SKIP, ""))
            else:
                _results.append((label, PASS, detail or ""))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run_cli(arguments: List[str], timeout: int = 300) -> subprocess.CompletedProcess:
    """Run ``python -m aurora …`` and capture the result."""
    return subprocess.run(
        [sys.executable, "-B", "-m", "aurora", *arguments],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


# --------------------------------------------------------------------------- #
# Test fixtures
# --------------------------------------------------------------------------- #


def build_project(root: Path) -> Path:
    """Create a tiny project directory the pipeline can plan against."""
    project = root / "project"
    (project / "wsis").mkdir(parents=True, exist_ok=True)
    (project / "wsis" / "TCGA-TEST-01.svs").write_text("dummy\n", encoding="ascii")
    (project / "wsis" / "TCGA-TEST-02.svs").write_text("dummy\n", encoding="ascii")
    with open(project / "Inference_samples.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Image_name", "Sample_name", "Bulk_sample_name"])
        writer.writerow(["TCGA-TEST-01.svs", "TCGA-TEST-01", "SAMPLE-A"])
        writer.writerow(["TCGA-TEST-02.svs", "TCGA-TEST-02", "SAMPLE-B"])
    with open(project / "Training_samples.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Sample", "Dir_name", "Image_name"])
        writer.writerow(["BRCA1", "BRCA1", "image1.tif"])
    return project


def find_combined_model() -> Optional[Path]:
    from aurora import bundle

    directory = REPO_ROOT / "trained_model"
    if not directory.is_dir():
        return None
    for path in sorted(directory.glob("*.pth")):
        if bundle.is_combined_checkpoint(path):
            return path
    return None


def find_legacy_model() -> Optional[Path]:
    from aurora import bundle

    for path in sorted((REPO_ROOT / "trained_model").glob("*.pth")):
        if not bundle.is_combined_checkpoint(path):
            return path
    return None


# --------------------------------------------------------------------------- #
# Requirement 1 -- one combined checkpoint, and only that
# --------------------------------------------------------------------------- #


@check("import aurora pulls in no third-party package", "R3")
def test_import_is_dependency_free() -> str:
    code = (
        "import sys; import aurora; "
        "heavy=[m for m in ('torch','numpy','pandas','tifffile','cv2','timm',"
        "'scipy','sklearn','h5py','PIL','yaml','rasterio','openslide','matplotlib','tqdm') "
        "if m in sys.modules]; print(','.join(heavy) or 'NONE')"
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", code], cwd=str(REPO_ROOT), capture_output=True, text=True
    )
    require(result.returncode == 0, result.stderr.strip())
    pulled = result.stdout.strip()
    require(pulled == "NONE", f"importing aurora pulled in: {pulled}")
    return "stdlib only"


@check("a combined checkpoint exists and validates", "R1")
def test_combined_checkpoint() -> str:
    model = find_combined_model()
    if model is None:
        return SKIP
    from aurora import bundle

    result = bundle.validate_bundle(model)
    require(result.ok, result.describe())
    checkpoint = bundle.CombinedCheckpoint.open(model)
    stages = checkpoint.stages()
    require("lowres" in stages, f"'lowres' missing from {stages}")
    require("highres" in stages, f"'highres' missing from {stages}")
    require(
        checkpoint.requires("highres") == ["lowres"],
        f"the checkpoint must declare highres -> lowres, got {checkpoint.requires('highres')}",
    )
    for asset in ("gene_list", "cell_types", "gene_norm_factor"):
        require(asset in checkpoint.manifest.assets, f"asset '{asset}' is not bundled")
    return f"{model.name}: stages={stages}"


@check("a legacy single-stage .pth is rejected with migration help", "R1")
def test_legacy_checkpoint_rejected() -> str:
    from aurora import bundle
    from aurora.errors import CheckpointFormatError

    legacy = find_legacy_model()
    if legacy is None:
        return SKIP
    require(
        not bundle.is_combined_checkpoint(legacy),
        f"{legacy.name} should not classify as combined",
    )
    try:
        bundle.CombinedCheckpoint.open(legacy)
    except CheckpointFormatError as exc:
        require(
            "merge_aurora_checkpoints" in (exc.hint or ""),
            "the error must tell the user how to build a combined checkpoint",
        )
        return "rejected with instructions"
    raise AssertionError("opening a legacy checkpoint should have raised")


@check("payloads are embedded byte-identically (hash verified)", "R1")
def test_payload_integrity() -> str:
    model = find_combined_model()
    if model is None:
        return SKIP
    from aurora.bundle import CombinedCheckpoint

    checkpoint = CombinedCheckpoint.open(model)
    for stage in checkpoint.stages():
        payload = checkpoint.payload_bytes(stage)
        entry = checkpoint.stage(stage)
        require(len(payload) == entry.size_bytes, f"{stage}: size mismatch")
        require(entry.sha256, f"{stage}: no payload hash recorded")
    return f"{len(checkpoint.stages())} stage payloads verified"


@check("embedded payloads keep the archive root torch.load requires", "R1")
def test_payload_layout() -> str:
    """Guard the 'file in archive is not in a subdirectory' failure.

    ``torch.save`` nests every record under a root directory named after the
    output file, and ``torch.load`` rejects an archive without one.  The payload
    must therefore be embedded verbatim -- unlike, for example, a flat re-zip.
    """
    import io
    import zipfile

    model = find_combined_model()
    if model is None:
        return SKIP
    from aurora.bundle import CombinedCheckpoint, torch_archive_layout

    checkpoint = CombinedCheckpoint.open(model)
    roots = []
    for stage in checkpoint.stages():
        with zipfile.ZipFile(io.BytesIO(checkpoint.payload_bytes(stage))) as archive:
            names = archive.namelist()
        ok, root, problem = torch_archive_layout(names)
        require(ok, f"{stage}: {problem}")
        require(root, f"{stage}: no archive root detected")
        require(
            f"{root}data.pkl" in names,
            f"{stage}: expected {root}data.pkl, found {sorted(names)[:4]}",
        )
        require(
            "data.pkl" not in names,
            f"{stage}: the payload was flattened; torch.load needs {root}data.pkl",
        )
        entry = checkpoint.stage(stage)
        require(
            entry.archive_root == root,
            f"{stage}: manifest records {entry.archive_root!r} but the payload root is {root!r}",
        )
        roots.append(root)
    validation = __import__("aurora.bundle", fromlist=["x"]).validate_bundle(model)
    require(validation.ok, validation.describe())
    return f"roots kept: {', '.join(roots)}"


@check("a flattened payload is rejected by the layout check", "R1")
def test_flat_payload_rejected() -> str:
    """The check itself must fail on the layout that broke at run time."""
    from aurora.bundle import torch_archive_layout

    ok, root, problem = torch_archive_layout(
        ["data.pkl", "data/0", "version", "byteorder", ".data/serialization_id"]
    )
    require(not ok, "a flat archive must be rejected")
    require("not in a subdirectory" in problem or "flat" in problem, problem)
    ok, root, problem = torch_archive_layout(
        ["AURORA_LUAD/data.pkl", "AURORA_LUAD/data/0", "AURORA_LUAD/version"]
    )
    require(ok, f"a rooted archive must be accepted: {problem}")
    require(root == "AURORA_LUAD/", root)
    ok, root, problem = torch_archive_layout(
        ["a/data.pkl", "b/data/0"]
    )
    require(not ok, "two top-level directories must be rejected")
    return "flat rejected, rooted accepted, split rejected"


@check("a checkpoint with flat payloads is refused before any stage runs", "R1")
def test_flat_bundle_refused() -> str:
    script = REPO_ROOT / "tests" / "test_payload_layout.py"
    require(script.is_file(), f"{script} is missing")
    result = subprocess.run(
        [sys.executable, "-B", str(script)], cwd=str(REPO_ROOT), capture_output=True, text=True
    )
    require(result.returncode == 0, (result.stdout + result.stderr)[-2500:])
    return "validate, loader and aurora run all refuse it with the rebuild command"


# --------------------------------------------------------------------------- #
# Requirement 2 -- the high-resolution model cannot run alone
# --------------------------------------------------------------------------- #


@check("predict_highres declares predict_lowres as a prerequisite", "R2")
def test_prerequisite_declared() -> str:
    from aurora.spec import prerequisites_of

    direct = prerequisites_of("predict_highres")
    require("predict_lowres" in direct, f"direct prerequisites are {direct}")
    transitive = prerequisites_of("predict_highres", recursive=True)
    require("features_lowres" in transitive, f"transitive prerequisites are {transitive}")
    return f"direct={direct}"


@check("the gate refuses predict_highres on a fresh run", "R2")
def test_gate_refuses_highres(tmp: Path) -> str:
    model = find_combined_model()
    if model is None:
        return SKIP
    project = build_project(tmp)
    result = run_cli(
        [
            "run",
            "--project",
            str(project),
            "--run",
            "gate_test",
            "--samples",
            str(project / "Inference_samples.csv"),
            "--model",
            str(model),
            "--stage",
            "predict_highres",
            "--quiet",
        ]
    )
    require(result.returncode != 0, "the run should have failed")
    combined = result.stdout + result.stderr
    require(
        "refusing to run stage 'predict_highres'" in combined,
        "the error must name the refused stage",
    )
    require(
        "low-resolution" in combined,
        "the error must explain why the low-resolution stage is required",
    )
    require(
        "--auto-prerequisites" in combined,
        "the error must offer the escape hatch that runs the prerequisites",
    )
    return "refused with an actionable message"


@check("--auto-prerequisites plans the low-resolution stage first", "R2")
def test_auto_prerequisites(tmp: Path) -> str:
    model = find_combined_model()
    if model is None:
        return SKIP
    project = build_project(tmp)
    result = run_cli(
        [
            "run",
            "--project",
            str(project),
            "--run",
            "auto_test",
            "--samples",
            str(project / "Inference_samples.csv"),
            "--model",
            str(model),
            "--stage",
            "predict_highres",
            "--auto-prerequisites",
            "--dry-run",
            "--quiet",
            "--uni-local-path",
            "./uni2h",
        ]
    )
    combined = result.stdout + result.stderr
    require(result.returncode == 0, combined[-2000:])
    require(
        "Adding missing prerequisite stage(s) automatically" in combined,
        "the CLI should announce the added stages",
    )
    order = [
        line.split("=== stage ")[1].split(" ")[0]
        for line in combined.splitlines()
        if line.strip().startswith("=== stage ")
    ]
    require("predict_lowres" in order, f"predict_lowres was not planned: {order}")
    require(
        order.index("predict_lowres") < order.index("predict_highres"),
        f"low-resolution must be planned before high-resolution: {order}",
    )
    return " -> ".join(order)


@check("a stage marked complete but with missing files is treated as stale", "R2")
def test_stale_detection(tmp: Path) -> str:
    from aurora.paths import RunLayout
    from aurora.state import RunState

    model = find_combined_model()
    if model is None:
        return SKIP
    project = build_project(tmp)
    run_name = "stale_test"
    # Create the run directory and run.json.
    run_cli(
        [
            "run",
            "--project",
            str(project),
            "--run",
            run_name,
            "--samples",
            str(project / "Inference_samples.csv"),
            "--model",
            str(model),
            "--stage",
            "rescale",
            "--dry-run",
            "--quiet",
        ]
    )
    layout = RunLayout.create(project, run_name)
    state = RunState.load(layout)
    # Claim 'segment' finished without producing anything.
    state.begin_stage("segment", command="fabricated for the test")
    state.complete_stage("segment", outputs=[str(layout.masks_dir / "TCGA-TEST-01.geojson")])
    require(state.is_recorded_complete("segment"), "the fabricated record was not stored")

    result = run_cli(
        [
            "run",
            "--project",
            str(project),
            "--run",
            run_name,
            "--samples",
            str(project / "Inference_samples.csv"),
            "--model",
            str(model),
            "--stage",
            "rescale",
            "--quiet",
        ]
    )
    combined = result.stdout + result.stderr
    require(result.returncode != 0, "the run should have refused to start")
    require(
        "output files are missing" in combined,
        f"the gate must verify artefacts, not just the flag: {combined[-900:]}",
    )

    status = run_cli(["status", "--project", str(project), "--run", run_name])
    require(status.returncode == 0, status.stderr[-500:])
    require("stale" in status.stdout, "aurora status should report the stage as stale")
    return "artefacts re-checked on disk, status reports 'stale'"


@check("iStar is gone from the pipeline", "R2")
def test_istar_removed() -> str:
    from aurora.deps import GROUP_PARENT
    from aurora.spec import STAGE_ORDER, STAGE_PACKAGES

    require("istar" not in STAGE_ORDER, "the istar stage is still registered")
    require("istar" not in STAGE_PACKAGES, "the istar package list is still registered")
    require("istar" not in GROUP_PARENT, "the istar dependency group still exists")
    requirements = REPO_ROOT / "requirements" / "optional-istar.txt"
    require(not requirements.exists(), f"{requirements} should no longer be generated")
    evidence = (REPO_ROOT / "docs" / "legacy" / "requirements_optional-istar.txt").is_file()
    require(evidence, "the old optional-istar requirements file was not preserved")

    # The entry script the pipeline used to drive is kept; the iStar library
    # itself was removed from the repository by the user (it is in the Recycle
    # Bin), so it is not asserted here.
    kept = REPO_ROOT / "code" / "run_istar.py"
    require(kept.is_file(), "code/run_istar.py was deleted")
    legacy_library = REPO_ROOT / "code" / "istar"
    note = "" if legacy_library.is_dir() else " (code/istar/ now lives in the Recycle Bin)"
    return "no stage, no dependency group, no requirements file" + note


@check("PatchDataset returns a collate-safe keep_flag", "keep-old")
def test_dataloader_collate() -> str:
    """Guard the fix for the 'numpy.bool' collate TypeError."""
    script = REPO_ROOT / "tests" / "test_dataloader_collate.py"
    require(script.is_file(), f"{script} is missing")
    result = subprocess.run(
        [sys.executable, "-B", str(script)], cwd=str(REPO_ROOT), capture_output=True, text=True
    )
    combined = result.stdout + result.stderr
    require(result.returncode == 0, combined[-2000:])

    # The comparison must be wrapped in bool(); a bare NumPy scalar leaks into
    # torch.utils.data.default_collate and raises TypeError.
    import ast

    source = (REPO_ROOT / "code" / "datasets.py").read_text(encoding="utf-8")
    tree = ast.parse(source, filename="code/datasets.py")
    guarded = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "keep_flag" for t in node.targets):
            continue
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
            if node.value.func.id == "bool":
                guarded = True
    require(
        guarded,
        "code/datasets.py must wrap the low-resolution keep_flag comparison in bool(); "
        "np.sum(...) > ... returns np.bool_, which default_collate cannot tensorise",
    )
    torch_note = "torch present" if "SKIP:" not in combined else "torch absent (mechanism checked)"
    return f"bool() guard present; {torch_note}"


@check("artefacts are keyed by Sample_name, not Image_name", "R2")
def test_naming_convention() -> str:
    script = REPO_ROOT / "tests" / "test_naming.py"
    require(script.is_file(), f"{script} is missing")
    result = subprocess.run(
        [sys.executable, "-B", str(script)], cwd=str(REPO_ROOT), capture_output=True, text=True
    )
    require(result.returncode == 0, (result.stdout + result.stderr)[-2000:])
    from aurora.naming import naming_for
    from aurora.samples import Sample

    naming = naming_for(
        Sample(
            sample_name="TCGA-44-5645-01B-04-BS4",
            image_name="TCGA-44-5645-01B-04-BS4.76ED5128-FB3D-44C7-A679-848860CAA3C5.svs",
        )
    )
    require(naming.names_differ, "a UUID-suffixed Image_name must count as different")
    require(naming.sample_name == "TCGA-44-5645-01B-04-BS4", naming.sample_name)
    return "7 naming checks pass"


@check("the run state records the sample fingerprint", "R2")
def test_state_records(tmp: Path) -> str:
    from aurora.paths import RunLayout
    from aurora.state import RunState

    model = find_combined_model()
    if model is None:
        return SKIP
    project = build_project(tmp)
    layout = RunLayout.create(project, "state_test")
    run_cli(
        [
            "run",
            "--project",
            str(project),
            "--run",
            "state_test",
            "--samples",
            str(project / "Inference_samples.csv"),
            "--model",
            str(model),
            "--stage",
            "predict_highres",
            "--dry-run",
            "--quiet",
        ]
    )
    state = RunState.load(layout)
    require(state.samples_fingerprint.startswith("sha256:"), "no sample fingerprint recorded")
    require(len(state.samples_info.get("names", [])) == 2, "sample names not recorded")
    require(state.model_info.get("model_name"), "model name not recorded")
    return state.samples_fingerprint[:24] + "…"


@check("the predictor module refuses to run without the low-res stage", "R2")
def test_predictor_module_gate(tmp: Path) -> str:
    """Calling the predictor directly must not bypass the gate."""
    model = find_combined_model()
    if model is None:
        return SKIP
    project = build_project(tmp)
    # Create the run (this writes run.json) without running anything.
    run_cli(
        [
            "run",
            "--project",
            str(project),
            "--run",
            "direct_test",
            "--samples",
            str(project / "Inference_samples.csv"),
            "--model",
            str(model),
            "--stage",
            "predict_highres",
            "--dry-run",
            "--quiet",
        ]
    )
    run_root = project / "aurora_runs" / "direct_test"
    if not (run_root / "run.json").is_file():
        return SKIP
    code = (
        "import sys;"
        f"sys.argv=['x','--run-root',r'{run_root}','--model',r'{model}',"
        "'--samples','config/samples.csv','--pred-dir','03_predictions/highres'];"
        "from aurora.inference import predict_highres as p;"
        "import aurora.errors as e;"
        "\ntry:\n p.main()\nexcept e.StagePrerequisiteError as exc:\n"
        " print('GATE_OK', exc.stage, ','.join(exc.missing)); sys.exit(0)\n"
        "except e.MissingDependencyError as exc:\n print('NO_TORCH'); sys.exit(0)\n"
        "else:\n print('GATE_MISSING'); sys.exit(3)\n"
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", code], cwd=str(REPO_ROOT), capture_output=True, text=True
    )
    combined = result.stdout + result.stderr
    if "NO_TORCH" in combined:
        return SKIP
    require("GATE_OK" in combined, f"the predictor did not gate: {combined[-1500:]}")
    require("predict_lowres" in combined, f"the gate must name predict_lowres: {combined[-800:]}")
    return "gated inside the predictor too"


# --------------------------------------------------------------------------- #
# Requirement 3 -- inference-only dependencies
# --------------------------------------------------------------------------- #


@check("static audit: no inference entry point reaches a training-only package", "R3")
def test_dependency_audit() -> str:
    result = subprocess.run(
        [sys.executable, "-B", str(REPO_ROOT / "tools" / "audit_inference_deps.py"), "--json"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    require(result.returncode == 0, f"audit failed:\n{result.stdout[-2000:]}{result.stderr[-800:]}")
    report = json.loads(result.stdout)
    require(report["verdict"] == "PASS", report.get("training_only_offenders"))
    eager = set(report["eager_third_party"])
    for forbidden in ("sklearn", "scanpy", "geopandas", "tensorboard", "pybiomart", "hest", "pyvips"):
        require(forbidden not in eager, f"'{forbidden}' is imported eagerly on the inference path")
    return f"{len(eager)} eager third-party modules, all inference-safe"


@check("requirements/*.txt match the dependency policy", "R3")
def test_requirements_in_sync() -> str:
    result = subprocess.run(
        [sys.executable, "-B", str(REPO_ROOT / "tools" / "sync_requirements.py"), "--check"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    require(result.returncode == 0, result.stdout + result.stderr)
    return "in sync"


@check("the inference requirements file excludes training packages", "R3")
def test_inference_requirements() -> str:
    path = REPO_ROOT / "requirements" / "inference.txt"
    require(path.is_file(), f"{path} is missing")
    text = path.read_text(encoding="utf-8")
    # Genuinely training-only.  `scikit-learn` used to be on this list: it moved
    # to the inference group when HistoSweep became a mandatory stage, because
    # HistoSweep's textureAnalysis does `from sklearn.mixture import ...` at
    # module level.  The rule this check encodes is unchanged -- an inference
    # install must not carry training-only packages -- but scikit-learn is no
    # longer one of them, and tests/test_histosweep_deps.py proves it is needed.
    for forbidden in (
        "scanpy",
        "geopandas",
        "tensorboard",
        "pybiomart",
        "torch_geometric",
        "hest",
        "pyvips",
    ):
        require(forbidden not in text, f"'{forbidden}' must not be in the inference requirements")
    # And the packages a full default inference run genuinely does need.
    for needed in ("scikit-learn", "scikit-image", "numba", "hdf5plugin"):
        require(
            needed in text,
            f"'{needed}' is needed by a mandatory inference stage but is not in inference.txt",
        )
    core = (REPO_ROOT / "requirements" / "core.txt").read_text(encoding="utf-8")
    training = (REPO_ROOT / "requirements" / "training.txt").read_text(encoding="utf-8")
    require("-r core.txt" in text, "inference.txt must include core.txt")
    require("-r inference.txt" in training, "training.txt must include inference.txt")
    require("torch" in core, "core.txt must list torch")
    return "inference.txt -> core.txt; training.txt -> inference.txt"


@check("aurora doctor reports the split as healthy", "R3")
def test_doctor() -> str:
    result = run_cli(["doctor"])
    combined = result.stdout + result.stderr
    require(result.returncode == 0, combined[-1500:])
    require(
        "no inference stage needs a training-only package" in combined,
        "doctor did not confirm the split",
    )
    return "split healthy"


# --------------------------------------------------------------------------- #
# Requirement 4 -- tidy output layout
# --------------------------------------------------------------------------- #


@check("a dry run prepares the whole tidy run layout", "R4")
def test_layout(tmp: Path) -> str:
    model = find_combined_model()
    if model is None:
        return SKIP
    project = build_project(tmp)
    result = run_cli(
        [
            "run",
            "--project",
            str(project),
            "--run",
            "layout_test",
            "--samples",
            str(project / "Inference_samples.csv"),
            "--model",
            str(model),
            "--all",
            "--dry-run",
            "--quiet",
            "--uni-local-path",
            "./uni2h",
        ]
    )
    require(result.returncode == 0, (result.stdout + result.stderr)[-2000:])
    root = project / "aurora_runs" / "layout_test"
    expected = [
        "run.json",
        "config/samples.csv",
        "logs",
        "01_segmentation/masks",
        "02_features/UNI_multiscale_patches/wsis_rescaled",
        "03_predictions/lowres/pred",
        "03_predictions/highres/pred",
        "04_histosweep",
        "05_plots",
        "tmp",
    ]
    missing = [name for name in expected if not (root / name).exists()]
    require(not missing, f"missing from the layout: {missing}")
    # iStar was dropped from the pipeline; no stage directory may reappear for it.
    require(
        not (root / "04_istar").exists(),
        "the layout still contains an iStar directory",
    )
    # Nothing may be written outside the run directory.
    strays = [
        p.name
        for p in project.iterdir()
        if p.name not in {"wsis", "Inference_samples.csv", "Training_samples.csv", "aurora_runs"}
    ]
    require(not strays, f"the run wrote outside its own directory: {strays}")
    return f"{len(expected)} layout entries present, no strays"


@check("every stage declares concrete output files", "R4")
def test_stage_outputs(tmp: Path) -> str:
    from aurora.config import RunConfig
    from aurora.paths import RunLayout
    from aurora.samples import read_sample_sheet
    from aurora.spec import StageContext, STAGE_SPECS

    project = build_project(tmp)
    layout = RunLayout.create(project, "outputs_test")
    sheet = read_sample_sheet(project / "Inference_samples.csv")
    ctx = StageContext(
        layout=layout,
        samples=sheet,
        lowres_params={"patch_size": [3584, 896, 224], "img_emb_dim": 1536},
        highres_params={"patch_size": [56, 14], "img_emb_dim": 3072},
        config=RunConfig(project_path=project, run_name="outputs_test"),
    )
    requirement_stages = {
        "segment",
        "rescale",
        "features_lowres",
        "features_highres",
        "predict_lowres",
        "predict_highres",
    }
    # Every stage, not just the inference ones: a literal list where a resolver
    # belongs (``outputs=[]``) only failed when that stage ran deep inside the
    # pipeline, and the training stages had no coverage at all.
    for spec in STAGE_SPECS:
        require(
            callable(spec.outputs),
            f"stage '{spec.name}': outputs must be a callable, got {type(spec.outputs).__name__}",
        )
        require(
            callable(spec.output_dirs),
            f"stage '{spec.name}': output_dirs must be a callable, "
            f"got {type(spec.output_dirs).__name__}",
        )
        outputs = spec.outputs(ctx)
        directories = spec.output_dirs(ctx)
        require(
            isinstance(outputs, list),
            f"stage '{spec.name}': outputs() must return a list, got {type(outputs).__name__}",
        )
        require(
            isinstance(directories, list),
            f"stage '{spec.name}': output_dirs() must return a list, "
            f"got {type(directories).__name__}",
        )
        if spec.output_groups is not None:
            require(
                callable(spec.output_groups),
                f"stage '{spec.name}': output_groups must be a callable",
            )
            groups = spec.output_groups(ctx)
            require(
                isinstance(groups, list),
                f"stage '{spec.name}': output_groups() must return a list",
            )
        if spec.name not in requirement_stages:
            # Training stages legitimately declare no files (preprocess only
            # produces a directory), but they must still resolve.
            continue
        require(outputs, f"stage '{spec.name}' declares no outputs")
        for path in outputs:
            require(
                str(path).startswith(str(layout.root)),
                f"stage '{spec.name}' writes outside the run directory: {path}",
            )
    return f"{len(STAGE_SPECS)} stage specs resolve their outputs"


@check("legacy output directories can be detected and migrated", "R4")
def test_legacy_migration(tmp: Path) -> str:
    from aurora.paths import detect_legacy_layout, plan_legacy_migration

    project = build_project(tmp)
    (project / "AURORA_interim" / "UNI_multiscale_patches" / "emb_224").mkdir(parents=True)
    (project / "AURORA_pred" / "pred").mkdir(parents=True)
    (project / "wsis_segmentation" / "masks").mkdir(parents=True)
    report = detect_legacy_layout(project)
    require(report.any_found, "no legacy directories detected")
    plan = plan_legacy_migration(project, "migrated")
    require(plan, "no migration plan produced")
    for source, destination in plan:
        require(source.exists(), f"plan references a missing source {source}")
        require("aurora_runs" in destination.as_posix(), f"bad destination {destination}")
    result = run_cli(
        ["migrate-layout", "--project", str(project), "--run", "migrated", "--dry-run"]
    )
    require(result.returncode == 0, (result.stdout + result.stderr)[-1200:])
    return f"{len(plan)} directories planned"


# --------------------------------------------------------------------------- #
# Requirement 5 -- the example command file
# --------------------------------------------------------------------------- #


@check("command.sh is still present and unmodified", "R5")
def test_command_sh_present() -> str:
    path = REPO_ROOT / "command.sh"
    require(path.is_file(), "command.sh was removed")
    text = path.read_text(encoding="utf-8")
    for expected in (
        "extract_features_highres.py",
        "train_highres.py",
        "predict_highres.py",
        "run_chunked_tcga.py",
    ):
        require(expected in text, f"command.sh no longer mentions {expected}")
    return f"{len(text.splitlines())} lines preserved"


@check("every documented aurora command parses", "R5")
def test_documented_cli() -> str:
    """Keep the README, run.sh and the parser from drifting apart.

    The ``--gene IL7R KRT8`` failure was a documentation/parser mismatch, so the
    docs are now executed (parsed) rather than trusted.
    """
    script = REPO_ROOT / "tests" / "test_documented_cli.py"
    require(script.is_file(), f"{script} is missing")
    result = subprocess.run(
        [sys.executable, "-B", str(script)], cwd=str(REPO_ROOT), capture_output=True, text=True
    )
    require(result.returncode == 0, (result.stdout + result.stderr)[-2500:])
    from aurora.cli import build_parser

    namespace = build_parser().parse_args(
        ["run", "--run", "R", "--stage", "plot", "--gene", "IL7R", "KRT8",
         "--celltype", "Malignant", "T_cell"]
    )
    require(namespace.gene == ["IL7R", "KRT8"], f"--gene parsed as {namespace.gene}")
    require(namespace.celltype == ["Malignant", "T_cell"], f"--celltype parsed as {namespace.celltype}")
    return "README + run.sh commands all parse; both flag spellings accepted"


@check("predictions can be plotted without re-running them", "R4")
def test_prediction_layout() -> str:
    """code/plot.py reads the legacy layout; the plot stage must bridge it."""
    script = REPO_ROOT / "tests" / "test_prediction_layout.py"
    require(script.is_file(), f"{script} is missing")
    result = subprocess.run(
        [sys.executable, "-B", str(script)], cwd=str(REPO_ROOT), capture_output=True, text=True
    )
    require(result.returncode == 0, (result.stdout + result.stderr)[-2500:])
    return "modern -> legacy projection is plot-able and correctly oriented"


@check("the plot stage points code/plot.py at a legacy view", "R4")
def test_plot_command_uses_view(tmp: Path) -> str:
    from aurora.commands import build_commands
    from aurora.config import RunConfig
    from aurora.paths import RunLayout
    from aurora.samples import Sample, SampleSheet
    from aurora.spec import StageContext

    project = build_project(tmp)
    run_root = project / "aurora_runs" / "plotcmd"
    layout = RunLayout.from_root(run_root, project)
    layout.ensure()
    sample = "TCGA-TEST-01"
    import pickle

    import numpy as np  # noqa: PLC0415

    spots = 3
    payload = {
        "exp_pred": {224: np.zeros((spots, 2), dtype=np.float32)},
        "log_prop_pred": {224: np.zeros((spots, 2), dtype=np.float32)},
        "pos": {224: np.array([[0, 0], [1, 0], [0, 1]])},
        "patches_loc_before_pad": {224: np.array([[0, 0], [1, 0], [0, 1]])},
        "genes": ["A", "B"],
        "cell_types": ["X", "Y"],
        "resolutions": [224],
    }
    pred_dir = layout.prediction_output_dir("lowres")
    pred_dir.mkdir(parents=True, exist_ok=True)
    with open(pred_dir / f"{sample}-pred.pickle", "wb") as handle:
        pickle.dump(payload, handle)

    config = RunConfig(
        project_path=project, run_name="plotcmd", gene_names=["A"], cell_types=[]
    )
    ctx = StageContext(
        layout=layout,
        samples=SampleSheet(
            path=None,
            samples=[Sample(sample_name=sample, image_name=f"{sample}.svs")],
            columns=["Image_name", "Sample_name"],
        ),
        options=config.stage_options(),
        config=config,
    )
    argv = [str(a) for a in build_commands("plot", ctx)[0].argv]
    target = argv[argv.index("-t") + 1]
    require("legacy_pred_view" in target, f"-t should point at the view, got {target!r}")
    require(
        (layout.tmp_dir / "legacy_pred_view" / "pred" / f"{sample}-pred.pickle").is_file(),
        "the view file was not written",
    )
    require("--genes" in argv and "A" in argv, f"--genes missing from {argv}")
    return f"-t {target}"


@check("the plot stage can plot the high-resolution predictions", "R4")
def test_plot_highres(tmp: Path) -> str:
    """`--plot-predictions highres` must rename into the file code/plot.py opens.

    predict_highres writes `<sample>-highres-pred.pickle`, but code/plot.py always
    opens `<sample>-pred.pickle`.  The view bridges the two, and the two kinds
    must not share a directory.
    """
    import pickle

    import numpy as np  # noqa: PLC0415

    project = build_project(tmp)
    from aurora.paths import RunLayout

    layout = RunLayout.from_root(project / "aurora_runs" / "hrplot", project)
    layout.ensure()
    sample = "TCGA-TEST-01"
    spots = 3
    payload = {
        "exp_pred": {
            56: np.zeros((spots, 2), dtype=np.float32),
            14: np.zeros((spots, 2), dtype=np.float32),
        },
        "log_prop_pred": {
            56: np.zeros((spots, 2), dtype=np.float32),
            14: np.zeros((spots, 2), dtype=np.float32),
        },
        "pos": {r: np.array([[0, 0], [1, 0], [0, 1]]) for r in (56, 14)},
        "patches_loc_before_pad": {r: np.array([[0, 0], [1, 0], [0, 1]]) for r in (56, 14)},
        "genes": ["A", "B"],
        "cell_types": np.asarray(["X", "Y"]),
        "resolutions": [56, 14],
    }
    pred_dir = layout.prediction_output_dir("highres")
    pred_dir.mkdir(parents=True, exist_ok=True)
    with open(pred_dir / f"{sample}-highres-pred.pickle", "wb") as handle:
        pickle.dump(payload, handle)

    from aurora.commands import build_commands
    from aurora.config import RunConfig
    from aurora.samples import Sample, SampleSheet
    from aurora.spec import StageContext

    def context(**options) -> StageContext:
        config = RunConfig(
            project_path=project, run_name="hrplot", gene_names=["A"], **options
        )
        return StageContext(
            layout=layout,
            samples=SampleSheet(
                path=None,
                samples=[Sample(sample_name=sample, image_name=f"{sample}.svs")],
                columns=["Image_name", "Sample_name"],
            ),
            options=config.stage_options(),
            config=config,
        )

    ctx = context(plot_predictions="highres", plot_resolution=56)
    argv = [str(a) for a in build_commands("plot", ctx)[0].argv]
    target = argv[argv.index("-t") + 1]
    require(
        "legacy_pred_view_highres" in target,
        f"high-res needs its own view, got {target!r}",
    )
    require(
        (layout.tmp_dir / "legacy_pred_view_highres" / "pred" / f"{sample}-pred.pickle").is_file(),
        "the high-res view file was not written under the name code/plot.py opens",
    )
    require(argv[argv.index("--resolution") + 1] == "56", f"--resolution wrong in {argv}")
    require(
        argv[argv.index("--plot_dir") + 1].endswith("05_plots/highres"),
        f"high-res figures need their own folder, got {argv[argv.index('--plot_dir') + 1]!r}",
    )
    # The low-resolution default is asserted in the check above; here the point is
    # that high-res does not silently fall back to it.

    # The default resolution (224) does not exist at high resolution; the stage
    # must say which levels do, not fail obscurely.
    from aurora.errors import ConfigError

    try:
        build_commands("plot", context(plot_predictions="highres"))
    except ConfigError as exc:
        require("56" in str(exc) and "14" in str(exc), f"the levels are listed: {exc}")
    else:
        require(False, "plotting high-res at 224 should have been rejected")
    return "high-res view at tmp/legacy_pred_view_highres, figures in 05_plots/highres"


@check("the HistoSweep mask filters the right patches", "R2")
def test_histosweep_mask_units() -> str:
    script = REPO_ROOT / "tests" / "test_histosweep_mask.py"
    require(script.is_file(), f"{script} is missing")
    result = subprocess.run(
        [sys.executable, "-B", str(script)], cwd=str(REPO_ROOT), capture_output=True, text=True
    )
    require(result.returncode == 0, (result.stdout + result.stderr)[-2500:])
    return "axis order and row alignment verified against an asymmetric mask"


@check("HistoSweep is a mandatory step", "R2")
def test_histosweep_mandatory(tmp: Path) -> str:
    """HistoSweep must be in the default chain and required by predict_highres.

    Its mask decides which high-resolution patches are kept, so it cannot be an
    optional extra -- and the declared output must be the mask it leaves behind,
    not the native files run_HistoSweep.py deletes.
    """
    from aurora.spec import INFERENCE_PIPELINE, get_stage

    spec = get_stage("histosweep")
    require(not spec.optional, "histosweep must not be optional")
    require(
        "histosweep" in INFERENCE_PIPELINE,
        f"histosweep missing from the default chain {INFERENCE_PIPELINE}",
    )
    require(
        INFERENCE_PIPELINE.index("histosweep") < INFERENCE_PIPELINE.index("predict_highres"),
        "histosweep must run before predict_highres",
    )
    require(
        "histosweep" in get_stage("predict_highres").prerequisites,
        f"predict_highres must require histosweep: {get_stage('predict_highres').prerequisites}",
    )
    require(
        spec.prerequisites == ("rescale",),
        f"histosweep only needs the rescaled slide, got {spec.prerequisites}",
    )
    require(
        spec.requires_group == "histosweep",
        f"histosweep needs its dependency group, got {spec.requires_group}",
    )

    # The declared output is the mask predict_highres reads, in wsis_rescaled/.
    from aurora.config import RunConfig
    from aurora.naming import histosweep_mask_path
    from aurora.paths import RunLayout
    from aurora.samples import read_sample_sheet
    from aurora.spec import StageContext

    project = build_project(tmp)
    layout = RunLayout.create(project, "histomask")
    sheet = read_sample_sheet(project / "Inference_samples.csv")
    ctx = StageContext(
        layout=layout,
        samples=sheet,
        lowres_params={"patch_size": [3584, 896, 224], "img_emb_dim": 1536},
        highres_params={"patch_size": [56, 14], "img_emb_dim": 3072},
        config=RunConfig(project_path=project, run_name="histomask"),
    )
    outputs = spec.outputs(ctx)
    require(outputs, "histosweep declares no outputs")
    for path in outputs:
        require(
            path == histosweep_mask_path(layout.rescaled_dir, path.name.split("_")[0]),
            f"unexpected histosweep output {path}",
        )
        require(
            path.name.endswith("_HistoSweep_mask_rescaled.tif"),
            f"histosweep must declare the tissue mask, got {path.name}",
        )
        require(
            "HistoSweep" not in path.parts,
            f"histosweep must not declare its deleted native outputs: {path}",
        )

    # And the high-resolution stage must refuse to run without it.
    from aurora.state import RunState

    state = RunState.load(layout, create=True)
    for stage in ("predict_lowres", "features_highres"):
        state.begin_stage(stage, command="test")
        state.complete_stage(stage, outputs=[], message="test")
    state.save()
    from aurora.state import evaluate_gate

    gate = evaluate_gate(state, ctx, "predict_highres")
    require(not gate.ok, "predict_highres must be blocked while histosweep is missing")
    require("histosweep" in gate.missing, f"the gate names histosweep: {gate.missing}")
    return "mandatory, ordered before predict_highres, and enforced by the gate"


@check("plot figures are at least 1/7 of the rescaled slide", "R4")
def test_plot_downscale(tmp: Path) -> str:
    """The figure is 1/downscale of the slide, and 1/7 is the floor."""
    import pickle

    import numpy as np  # noqa: PLC0415

    from aurora.cli import build_parser
    from aurora.commands import MIN_PLOT_FRACTION_DENOMINATOR, build_commands
    from aurora.config import RunConfig
    from aurora.errors import ConfigError
    from aurora.paths import RunLayout
    from aurora.samples import Sample, SampleSheet
    from aurora.spec import StageContext

    require(
        MIN_PLOT_FRACTION_DENOMINATOR == 7,
        f"the minimum fraction is 1/7, got 1/{MIN_PLOT_FRACTION_DENOMINATOR}",
    )
    namespace = build_parser().parse_args(["run", "--run", "R", "--stage", "plot", "--gene", "IL7R"])
    require(namespace.plot_downscale == 7, f"the default is 7, got {namespace.plot_downscale}")

    project = build_project(tmp)
    layout = RunLayout.from_root(project / "aurora_runs" / "scale", project)
    layout.ensure()
    sample = "TCGA-TEST-01"
    spots = 3
    payload = {
        "exp_pred": {224: np.zeros((spots, 2), dtype=np.float32)},
        "log_prop_pred": {224: np.zeros((spots, 2), dtype=np.float32)},
        "pos": {224: np.array([[0, 0], [1, 0], [0, 1]])},
        "patches_loc_before_pad": {224: np.array([[0, 0], [1, 0], [0, 1]])},
        "genes": ["A", "B"],
        "cell_types": np.asarray(["X", "Y"]),
        "resolutions": [224],
    }
    pred_dir = layout.prediction_output_dir("lowres")
    pred_dir.mkdir(parents=True, exist_ok=True)
    with open(pred_dir / f"{sample}-pred.pickle", "wb") as handle:
        pickle.dump(payload, handle)

    def context(**options) -> StageContext:
        config = RunConfig(project_path=project, run_name="scale", gene_names=["A"], **options)
        return StageContext(
            layout=layout,
            samples=SampleSheet(
                path=None,
                samples=[Sample(sample_name=sample, image_name=f"{sample}.svs")],
                columns=["Image_name", "Sample_name"],
            ),
            options=config.stage_options(),
            config=config,
        )

    argv = [str(a) for a in build_commands("plot", context())[0].argv]
    require(
        argv[argv.index("--downscale") + 1] == "7",
        f"the default downscale must reach code/plot.py: {argv}",
    )
    smaller = [str(a) for a in build_commands("plot", context(plot_downscale=4))[0].argv]
    require(smaller[smaller.index("--downscale") + 1] == "4", "an explicit value is passed through")

    for bad in (8, 10, 0):
        try:
            build_commands("plot", context(plot_downscale=bad))
        except ConfigError as exc:
            message = str(exc) + (getattr(exc, "hint", "") or "")
            require(
                "1/7" in message or ">= 1" in message,
                f"--plot-downscale {bad} explains the limit: {message[:120]}",
            )
        else:
            require(False, f"--plot-downscale {bad} should have been rejected")

    # code/plot.py must keep its own default of 10 so existing callers are
    # unaffected, and must actually use the flag.
    plot_source = (REPO_ROOT / "code" / "plot.py").read_text(encoding="utf-8")
    require(
        "'--downscale', type=int, default=10" in plot_source,
        "code/plot.py must default to 10 to stay compatible",
    )
    require("//10" not in plot_source, "no hardcoded //10 may remain in code/plot.py")
    return "default 1/7, enforced, and code/plot.py keeps its 10 for direct callers"


@check("the HDF5 prediction format round-trips", "R1")
def test_h5_format() -> str:
    """The compact high-resolution format is the default, so it must decode.

    `--uint8-output` writes HDF5 with the spatial_gene_delta encoding from
    command.sh; `aurora/h5out.py` and `aurora/h5in.py` port the writer and reader
    from code/predict_highres_TCGA_optimized.py and code/highres_h5_loader.py.
    """
    script = REPO_ROOT / "tests" / "test_h5_format.py"
    require(script.is_file(), f"{script} is missing")
    result = subprocess.run(
        [sys.executable, "-B", str(script)], cwd=str(REPO_ROOT), capture_output=True, text=True
    )
    require(result.returncode == 0, (result.stdout + result.stderr)[-2500:])
    return "write -> read round-trip verified, including the spatial row order"


@check("prediction output defaults match command.sh", "R1")
def test_output_format_defaults(tmp: Path) -> str:
    """The compact formats from command.sh are the defaults.

    Low resolution: ``--uint8_output --only_224``.
    High resolution: ``--uint8_output --h5_layout spatial_gene_delta
    --h5_compression zstd --h5_compression_level 9``.

    The stage commands are the only place these can be got wrong, because the
    internal parsers are not what `aurora run` invokes.
    """
    from aurora.cli import build_parser
    from aurora.commands import build_commands
    from aurora.config import RunConfig
    from aurora.paths import RunLayout
    from aurora.samples import Sample, SampleSheet
    from aurora.spec import StageContext

    namespace = build_parser().parse_args(["run", "--run", "R", "--stage", "predict_lowres"])
    require(namespace.uint8_output is True, "uint8 output must default on")
    require(namespace.only_224 is True, "only-224 must default on")
    require(namespace.h5_layout == "spatial_gene_delta", f"h5 layout: {namespace.h5_layout}")
    require(namespace.h5_compression == "zstd", f"h5 compression: {namespace.h5_compression}")
    require(namespace.h5_compression_level == 9, f"h5 level: {namespace.h5_compression_level}")

    negated = build_parser().parse_args(
        ["run", "--run", "R", "--no-uint8-output", "--no-only-224"]
    )
    require(negated.uint8_output is False, "--no-uint8-output must turn it off")
    require(negated.only_224 is False, "--no-only-224 must turn it off")

    project = build_project(tmp)
    layout = RunLayout.from_root(project / "aurora_runs" / "fmt", project)
    layout.ensure()
    sample = "TCGA-TEST-01"

    def context(**options) -> StageContext:
        config = RunConfig(
            project_path=project,
            run_name="fmt",
            model_path=find_combined_model(),
            **options,
        )
        return StageContext(
            layout=layout,
            samples=SampleSheet(
                path=None,
                samples=[Sample(sample_name=sample, image_name=f"{sample}.svs")],
                columns=["Image_name", "Sample_name"],
            ),
            lowres_params={"patch_size": [3584, 896, 224], "img_emb_dim": 1536},
            highres_params={"patch_size": [56, 14], "img_emb_dim": 3072},
            options=config.stage_options(),
            config=config,
        )

    lowres = [str(a) for a in build_commands("predict_lowres", context())[0].argv]
    require("--uint8-output" in lowres, f"predict_lowres needs --uint8-output: {lowres}")
    require("--only-224" in lowres, f"predict_lowres needs --only-224: {lowres}")

    highres = [str(a) for a in build_commands("predict_highres", context())[0].argv]
    require("--uint8-output" in highres, f"predict_highres needs --uint8-output: {highres}")
    for flag, value in (
        ("--h5-layout", "spatial_gene_delta"),
        ("--h5-compression", "zstd"),
        ("--h5-compression-level", "9"),
    ):
        require(flag in highres, f"predict_highres needs {flag}: {highres}")
        require(
            highres[highres.index(flag) + 1] == value,
            f"{flag} should be {value}, got {highres[highres.index(flag) + 1]!r}",
        )

    # An explicit opt-out must reach the stage, not be silently dropped.
    off = [str(a) for a in build_commands("predict_highres", context(uint8_output=False))[0].argv]
    require("--no-uint8-output" in off, f"the opt-out must be passed: {off}")
    require("--uint8-output" not in off, "--uint8-output must not also be passed")
    off_low = [
        str(a)
        for a in build_commands("predict_lowres", context(only_224=False, uint8_output=False))[0].argv
    ]
    require("--no-only-224" in off_low and "--no-uint8-output" in off_low, f"{off_low}")
    return "uint8 + only-224 (low res), uint8 + spatial_gene_delta/zstd-9 (high res)"


@check("the default high-res format's dependency is declared", "R3")
def test_default_format_is_installable() -> str:
    """zstd is the default HDF5 filter, so its plugin must be a declared dep.

    `h5py` cannot write or read the Zstd filter without `hdf5plugin`.  Making
    zstd the default while leaving the plugin out of the dependency policy would
    mean a fresh install fails on the default `predict_highres` run.
    """
    from aurora.config import RunConfig
    from aurora.deps import check_group, dependencies_for_group

    config = RunConfig()
    require(config.h5_compression == "zstd", f"the default codec is {config.h5_compression!r}")
    if config.h5_compression == "zstd":
        declared = {dep.distribution for dep in dependencies_for_group("inference")}
        require(
            "hdf5plugin" in declared,
            f"the inference group must declare hdf5plugin, it has: {sorted(declared)}",
        )
    # Every codec the CLI offers must be either built into h5py or declared.
    from aurora.cli import build_parser

    parser = build_parser()
    run_parser = None
    for action in parser._actions:
        choices = getattr(action, "choices", None)
        if isinstance(choices, dict) and "run" in choices:
            run_parser = choices["run"]
    require(run_parser is not None, "could not find the 'run' subcommand")
    codecs: List[str] = []
    for action in run_parser._actions:
        if getattr(action, "dest", "") == "h5_compression":
            codecs = list(action.choices or [])
    require(codecs, "the CLI does not declare --h5-compression choices")
    for codec in codecs:
        if codec == "zstd":
            continue
        require(codec in {"gzip", "none"}, f"unexpected codec {codec!r} needs a dependency decision")
    # h5py itself is core, so .h5 output works even in a bare install.
    core = {dep.distribution for dep in dependencies_for_group("core")}
    require("h5py" in core, f"h5py must be a core dependency: {sorted(core)}")

    report = check_group("inference")
    return (
        f"zstd default, hdf5plugin declared; choices={codecs}; "
        f"inference group has {len(report.missing)} package(s) not installed here"
    )


@check("every HistoSweep import is installable", "R3")
def test_histosweep_deps() -> str:
    """A mandatory stage's imports must all be declared.

    `numba` was missing at run time and `scikit-learn` was waiting behind it --
    Python reports only the first failed import, so the imports are read out of
    the vendored source instead.
    """
    script = REPO_ROOT / "tests" / "test_histosweep_deps.py"
    require(script.is_file(), f"{script} is missing")
    result = subprocess.run(
        [sys.executable, "-B", str(script)], cwd=str(REPO_ROOT), capture_output=True, text=True
    )
    require(result.returncode == 0, (result.stdout + result.stderr)[-2500:])
    return "numba, scikit-learn and scikit-image declared; 3 exemptions justified"


@check("the demo downloader also fetches the published model", "R6")
def test_demo_downloader_model_step() -> str:
    """Inference needs weights, and the published ones are a Hugging Face download.

    The file there is the low-resolution model on its own, so the downloader has to
    merge it with the shipped high-resolution weights; handing a raw state_dict to
    ``--model`` fails with "not a combined AURORA checkpoint".
    """
    import importlib.util

    script = REPO_ROOT / "examples" / "download_demo_data.py"
    require(script.is_file(), "examples/download_demo_data.py is missing")
    spec = importlib.util.spec_from_file_location("demo_downloader", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    require(
        "AURORAData/AURORA_prediction" in module.MODEL_URL,
        f"the model URL should point at the published weights: {module.MODEL_URL}",
    )
    require(
        module.MODEL_NAME.endswith(".pth"),
        f"unexpected model file name: {module.MODEL_NAME}",
    )
    require(
        module.COMBINED_NAME.endswith(".pth"),
        f"unexpected combined file name: {module.COMBINED_NAME}",
    )
    require(
        module.COMBINED_NAME != module.MODEL_NAME,
        "the combined checkpoint must be a separate file from the download",
    )
    require(
        callable(getattr(module, "download_model", None)),
        "the downloader has no download_model step",
    )
    # The merge inputs must be the ones that ship with the repository.
    require(
        (REPO_ROOT / "trained_model" / module.HIGHRES_WEIGHTS).is_file(),
        f"the high-resolution weights {module.HIGHRES_WEIGHTS} are not in trained_model/",
    )

    # The download is verified with `aurora inspect`, so it must accept a combined
    # checkpoint and reject a legacy state_dict. This is what keeps the step from
    # declaring an unreadable file "ready".
    require(
        callable(getattr(module, "_is_combined_checkpoint", None)),
        "the downloader does not verify what it downloaded",
    )
    combined = REPO_ROOT / "trained_model" / "AURORA_LUAD_combined.pth"
    legacy = REPO_ROOT / "trained_model" / "AURORA_LUAD_V2.pth"
    if combined.is_file():
        ok, _detail = module._is_combined_checkpoint(combined)
        require(ok, "a combined checkpoint must be recognised as one")
    if legacy.is_file():
        ok, detail = module._is_combined_checkpoint(legacy)
        require(not ok, "a legacy state_dict must not be reported as ready")
        require(bool(detail), "the rejection should explain itself")

    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    require(
        "AURORA_prediction" in readme,
        "the README does not link the published model weights",
    )
    # The README tells people how to fetch the data; the model-only switch is part of
    # the script's own interface, so check it where it is defined and still reject the
    # option disappearing from the downloader.
    require(
        "download_demo_data.py" in readme,
        "the README does not mention the demo downloader",
    )
    help_text = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
    )
    require(
        help_text.returncode == 0,
        f"the downloader's --help failed: {help_text.stderr[-300:]}",
    )
    require(
        "--what" in help_text.stdout and "model" in help_text.stdout,
        "the downloader should still offer a model-only download",
    )
    return "model URL, merge inputs and a model-only switch all present"


@check("a plot resolution from the other prediction is pointed out", "R4")
def test_plot_resolution_hint(tmp: Path) -> str:
    """`--plot-resolution 14` against the low-res prediction must say where 14 lives.

    14 px is the high-resolution patch size; the low-res prediction only has 224. The
    bare error listed "(available: 224)" and left the user to work out that the fix is
    one flag away.
    """
    from aurora import plotting as plotting_module
    from aurora.errors import ConfigError
    from aurora.paths import RunLayout

    layout = RunLayout.from_root(tmp / "aurora_runs" / "hint", tmp)
    prediction_dir = layout.prediction_dir("lowres")

    original_locate = plotting_module.locate_source
    original_available = plotting_module._available_resolutions
    try:
        plotting_module.locate_source = (
            lambda directory, kind, sample: Path(directory) / f"{sample}-{kind}.pickle"
        )
        plotting_module._available_resolutions = (
            lambda path: [14, 56] if "highres" in str(path) else [224]
        )

        error = ConfigError(
            f"resolution 14 is not in {prediction_dir}/pred/S1-pred.pickle (available: 224)"
        )
        improved = plotting_module._resolution_hint(
            layout, prediction_dir, "lowres", "S1", 14, error
        )
        hint = improved.hint or ""
        require(
            "--plot-predictions highres" in hint,
            f"the hint must name the flag that fixes it: {hint}",
        )
        require("--plot-resolution 14" in hint, f"and the value to pass: {hint}")
        require("224" in hint, f"and what the low-res prediction has: {hint}")
        require(
            improved.message == error.message,
            "the original message must be preserved",
        )

        # A level that exists nowhere explains both sides instead of guessing.
        nowhere = plotting_module._resolution_hint(
            layout, prediction_dir, "lowres", "S1", 7, ConfigError("resolution 7 missing")
        )
        text = nowhere.hint or ""
        require("lowres resolutions" in text, text)
        require("highres resolutions" in text, text)
        require("--plot-predictions highres" not in text, "no false suggestion")
    finally:
        plotting_module.locate_source = original_locate
        plotting_module._available_resolutions = original_available
    return "14 px points at --plot-predictions highres; an unknown level lists both"


@check("a stage that depends on its arguments is not reused blindly", "R6")
def test_plot_args_fingerprint(tmp: Path) -> str:
    """Asking `plot` for different genes must not return the previous figures.

    Completion used to be recorded without any note of *what* was drawn, so a second
    run with different genes was reported as "already complete" and nothing new
    appeared.  The fingerprint makes the request part of the completion record.
    """
    from aurora.config import RunConfig
    from aurora.paths import RunLayout
    from aurora.pipeline import recorded_args_mismatch
    from aurora.samples import Sample, SampleSheet
    from aurora.spec import StageContext, get_stage
    from aurora.state import RunState

    project = build_project(tmp)
    layout = RunLayout.from_root(project / "aurora_runs" / "fp", project)
    layout.ensure("inference")
    sheet = SampleSheet(
        path=None,
        samples=[Sample(sample_name="S1", image_name="S1.svs")],
        columns=["Image_name", "Sample_name"],
    )

    def context(genes=(), celltypes=(), kind="lowres") -> StageContext:
        config = RunConfig(
            project_path=project,
            run_name="fp",
            mode="inference",
            gene_names=list(genes),
            cell_types=list(celltypes),
            plot_predictions=kind,
        )
        return StageContext(
            layout=layout, samples=sheet, options=config.stage_options(), config=config
        )

    spec = get_stage("plot")
    require(spec.args_fingerprint is not None, "plot must declare an args fingerprint")

    first = spec.args_fingerprint(context(["EPCAM", "IL7R"]))
    other = spec.args_fingerprint(context(["SFTPB", "CD74"]))
    require(first != other, "a different gene list must change the fingerprint")
    require(
        first == spec.args_fingerprint(context(["IL7R", "EPCAM"])),
        "gene order must not change the fingerprint",
    )
    require(
        first != spec.args_fingerprint(context(["EPCAM", "IL7R"], kind="highres")),
        "the prediction kind must change the fingerprint",
    )
    require(
        first != spec.args_fingerprint(context(["EPCAM", "IL7R"], ["Malignant"])),
        "a cell-type selection must change the fingerprint",
    )

    state = RunState.load(layout, create=True)
    # Nothing recorded yet: the stage has to run.
    require(
        recorded_args_mismatch(spec, context(["EPCAM"]), state),
        "an unrecorded stage must not count as matching",
    )
    state.begin_stage("plot", samples=["S1"])
    state.complete_stage(
        "plot", outputs=[layout.plots_dir], extra={"args_fingerprint": first}
    )
    state.save()
    require(
        recorded_args_mismatch(spec, context(["EPCAM", "IL7R"]), state) == "",
        "the same request should still be reusable",
    )
    mismatch = recorded_args_mismatch(spec, context(["SFTPB", "CD74"]), state)
    require(mismatch, "a changed gene list must block reuse")
    require("arguments" in mismatch, f"the reason should say so: {mismatch}")

    # A stage without a fingerprint keeps the old behaviour.
    plain = get_stage("segment")
    require(plain.args_fingerprint is None, "segment should not need a fingerprint")
    require(
        recorded_args_mismatch(plain, context(), state) == "",
        "stages without a fingerprint must be reusable as before",
    )
    return "plot records what it drew; a changed gene list re-runs it"

def test_stage_packages_covered_by_group() -> str:
    """A stage must never need a package that its group does not install.

    ``STAGE_PACKAGES`` is what the pre-flight check enforces at run time, so every
    name in it has to come from the stage's own group or an ancestor.  The default
    chain is checked separately; this covers *every* stage, including optional ones
    like ``plot``, which is where a missing ``matplotlib`` used to surface as
    "stage 'plot' needs 3 package(s) that are not installed".
    """
    from aurora.deps import DEPENDENCIES, GROUP_PARENT
    from aurora.spec import STAGE_PACKAGES, STAGE_SPECS

    by_import = {dep.import_name: dep for dep in DEPENDENCIES if dep.import_name}
    problems: List[str] = []
    checked = 0
    for spec in STAGE_SPECS:
        chain = set()
        current = spec.requires_group
        while current is not None:
            chain.add(current)
            current = GROUP_PARENT.get(current)
        for name in STAGE_PACKAGES.get(spec.name, ()):
            checked += 1
            dep = by_import.get(name)
            if dep is None:
                problems.append(
                    f"stage '{spec.name}' needs '{name}', which aurora/deps.py does not declare"
                )
                continue
            if dep.group not in chain:
                problems.append(
                    f"stage '{spec.name}' (group '{spec.requires_group}') needs '{name}' "
                    f"from group '{dep.group}', which installing .[{spec.requires_group}] "
                    "does not provide"
                )
    require(not problems, "; ".join(sorted(set(problems))))
    require(checked >= 20, f"only {checked} package requirements were checked")
    return f"{len(STAGE_SPECS)} stages, {checked} package requirements all covered"


@check("one extra installs the whole default inference chain", "R3")
def test_inference_extra_is_complete() -> str:
    """Every stage in the default chain must be runnable from `.[inference]`.

    `histosweep` is a mandatory stage, so its dependency belongs to the
    `inference` group. Leaving it in a separate extra produces an install where
    the default chain fails halfway through with a missing-package error -- a
    user would have to discover an extra the README never told them about.
    """
    from aurora.deps import GROUP_PARENT, dependencies_for_group, group_of
    from aurora.spec import INFERENCE_PIPELINE, get_stage

    inference_dists = {dep.distribution for dep in dependencies_for_group("inference")}
    problems: List[str] = []
    for stage in INFERENCE_PIPELINE:
        group = get_stage(stage).requires_group
        # The stage's group must be the inference group or descend from it.
        chain = []
        current = group
        while current is not None:
            chain.append(current)
            current = GROUP_PARENT.get(current)
        if "inference" not in chain:
            problems.append(
                f"stage '{stage}' requires the '{group}' group, which does not inherit inference"
            )
            continue
        for dep in dependencies_for_group(group):
            if dep.distribution not in inference_dists:
                problems.append(
                    f"stage '{stage}' (group '{group}') needs {dep.distribution!r}, "
                    "which .[inference] does not install"
                )
    require(not problems, "; ".join(sorted(set(problems))))

    # And nothing mandatory may sit in a group that is not reachable at all.
    for stage in INFERENCE_PIPELINE:
        spec = get_stage(stage)
        require(not spec.optional, f"stage '{stage}' is in the default chain but marked optional")
        require(
            group_of(spec.requires_group) == spec.requires_group
            or spec.requires_group in GROUP_PARENT,
            f"stage '{stage}' names an unknown dependency group {spec.requires_group!r}",
        )
    return (
        f"{len(INFERENCE_PIPELINE)} default stages runnable from .[inference] "
        f"({len(inference_dists)} distributions)"
    )


@check("the plot stage needs a gene or a cell type, not both", "R4")
def test_plot_argument_handling(tmp: Path) -> str:
    """`plot.py` must accept `--genes` without `--celltypes`, and vice versa.

    The original tested ``'.csv' in args.celltypes`` *before* the
    ``is not None`` guard, and `and` evaluates left to right, so the `in` ran
    against None and raised
    ``TypeError: argument of type 'NoneType' is not iterable``.  The script
    therefore only worked when BOTH were passed, even though it has its own
    "at least one of" error.  The operands are now the right way round.
    """
    import pickle

    import numpy as np  # noqa: PLC0415

    source = (REPO_ROOT / "code" / "plot.py").read_text(encoding="utf-8")
    broken = []
    for line in source.splitlines():
        if line.lstrip().startswith("#"):
            continue  # the original (broken) expression is kept as a comment
        if "'.csv' in" not in line or "is not None" not in line:
            continue
        if line.index("'.csv' in") < line.index("is not None"):
            broken.append(line.strip())
    require(
        not broken,
        "code/plot.py must test `args.X is not None` before `'.csv' in args.X`, "
        f"otherwise a None argument raises TypeError: {broken}",
    )

    from aurora.commands import build_commands
    from aurora.config import RunConfig
    from aurora.errors import ConfigError
    from aurora.paths import RunLayout
    from aurora.samples import Sample, SampleSheet
    from aurora.spec import StageContext

    project = build_project(tmp)
    layout = RunLayout.from_root(project / "aurora_runs" / "plotargs", project)
    layout.ensure()
    sample = "TCGA-TEST-01"
    payload = {
        "exp_pred": {224: np.zeros((2, 2), dtype=np.float32)},
        "log_prop_pred": {224: np.zeros((2, 2), dtype=np.float32)},
        "pos": {224: np.array([[0, 0], [1, 0]])},
        "patches_loc_before_pad": {224: np.array([[0, 0], [1, 0]])},
        "genes": ["A", "B"],
        "cell_types": np.asarray(["X", "Y"]),
        "resolutions": [224],
    }
    output = layout.prediction_output_dir("lowres")
    output.mkdir(parents=True, exist_ok=True)
    with open(output / f"{sample}-pred.pickle", "wb") as handle:
        pickle.dump(payload, handle)

    def context(**options) -> StageContext:
        config = RunConfig(project_path=project, run_name="plotargs", **options)
        return StageContext(
            layout=layout,
            samples=SampleSheet(
                path=None,
                samples=[Sample(sample_name=sample, image_name=f"{sample}.svs")],
                columns=["Image_name", "Sample_name"],
            ),
            options=config.stage_options(),
            config=config,
        )

    gene_only = [str(a) for a in build_commands("plot", context(gene_names=["IL7R"]))[0].argv]
    require("--genes" in gene_only and "--celltypes" not in gene_only, f"gene only: {gene_only}")
    cell_only = [
        str(a) for a in build_commands("plot", context(cell_types=["Malignant"]))[0].argv
    ]
    require(
        "--celltypes" in cell_only and "--genes" not in cell_only,
        f"cell type only: {cell_only}",
    )
    both = [
        str(a)
        for a in build_commands(
            "plot", context(gene_names=["IL7R"], cell_types=["Malignant"])
        )[0].argv
    ]
    require("--genes" in both and "--celltypes" in both, f"both: {both}")
    try:
        build_commands("plot", context())
    except ConfigError as exc:
        require("--gene" in str(exc) and "--celltype" in str(exc), f"neither: {exc}")
    else:
        require(False, "asking for neither should be refused")
    return "gene-only, cell-type-only and both all build; neither is refused"


@check("TRIDENT's checkpoint registry is repaired in place", "R3")
def test_tissue_segmentation_registry(tmp: Path) -> str:
    """HEST segments via TRIDENT, which refuses to run without registered weights.

    TRIDENT's error names ``trident/segmentation_models/local_ckpts.json`` --
    relative to the installed package, not to the working directory, and nothing to
    do with HEST's own tree -- and then prints the shipped *placeholder value* where
    the checkpoint path belongs. That is why it reads as "place it at:
    local_ckpts.json". The command repairs that entry in place.
    """
    import json

    from aurora import hest

    site = tmp / "site"
    trident = site / "trident"
    (trident / "segmentation_models").mkdir(parents=True)
    (trident / "__init__.py").write_text("", encoding="utf-8")
    (trident / "segmentation_models" / "__init__.py").write_text("", encoding="utf-8")
    (site / "hest").mkdir()
    (site / "hest" / "__init__.py").write_text("", encoding="utf-8")
    ckpt = tmp / "deeplabv3_seg_v4.ckpt"
    ckpt.write_bytes(b"weights")

    registry = trident / "segmentation_models" / "local_ckpts.json"
    # The registry TRIDENT actually ships: keyed by *model name*, with empty values.
    # `segmentation_model_factory("hest")` looks up "hest", so that is the entry that
    # has to be filled -- a filename key is never read.
    shipped = {"grandqc": "", "grandqc_artifact": "", "hest": ""}
    registry.write_text(json.dumps(shipped), encoding="utf-8")

    sys.path.insert(0, str(site))
    try:
        require(hest.find_trident_package() == trident, "the trident package must be found")
        require(
            hest.find_registry() == registry.resolve(),
            f"the registry must be TRIDENT's, got {hest.find_registry()}",
        )

        # Every entry is empty, and the key TRIDENT reads is the model name.
        data = hest.read_registry(registry)
        require(
            sorted(hest.placeholder_keys(data)) == ["grandqc", "grandqc_artifact", "hest"],
            f"empty values are unconfigured: {data}",
        )
        require(
            hest.target_key(data) == "hest",
            f"the entry TRIDENT reads is 'hest', got {hest.target_key(data)!r}",
        )
        ok, message = hest.check_setup()
        require(not ok, f"an empty entry must read as unconfigured: {message}")
        require("'hest'" in message, f"the message must name the entry: {message}")

        # Repairing fills 'hest' and leaves the others alone.
        data, key = hest.record_checkpoint(registry, ckpt)
        require(key == "hest", f"the wrong entry was repaired: {key!r}")
        require(
            data["hest"] == str(ckpt.resolve()),
            f"the entry must hold an absolute path: {data}",
        )
        require(
            data["grandqc"] == "" and data["grandqc_artifact"] == "",
            f"other models must be untouched: {data}",
        )
        require(
            set(data) == set(shipped),
            f"no key may be invented -- only 'hest' gets filled: {data}",
        )
        ok, message = hest.check_setup()
        require(ok, f"TRIDENT should be ready now: {message}")

        # A path that points nowhere is still reported.
        registry.write_text(
            json.dumps({"deeplabv3_seg_v4.ckpt": str(tmp / "gone.ckpt")}), encoding="utf-8"
        )
        ok, message = hest.check_setup()
        require(not ok and "does not exist" in message, message)

        # Malformed JSON is a clear error, not a traceback.
        registry.write_text("not json", encoding="utf-8")
        try:
            hest.read_registry(registry)
        except Exception as exc:  # noqa: BLE001
            require("not valid JSON" in str(exc), str(exc))
        else:
            require(False, "malformed JSON should raise")

        # The diagnosis names both packages and labels the placeholder.
        registry.write_text(
            json.dumps({"deeplabv3_seg_v4.ckpt": "local_ckpts.json"}), encoding="utf-8"
        )
        lines = "\n".join(hest.diagnose())
        require("trident package" in lines, lines)
        require("PLACEHOLDER" in lines, f"the placeholder must be labelled: {lines}")
    finally:
        if str(site) in sys.path:
            sys.path.remove(str(site))
        for name in ("trident", "hest"):
            sys.modules.pop(name, None)
    return "placeholder detected and repaired in TRIDENT's registry; others preserved"


@check("every README example box holds a real command", "R6")
def test_readme_example_boxes() -> str:
    """Each coloured ``Example`` callout must contain a fenced block.

    The README promises that these boxes run as they stand, so an empty box is a
    documentation defect. They are ``> [!TIP]`` callouts, which render as a coloured
    panel, and the ``**Example**`` line inside identifies them.
    """
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8").splitlines()
    alerts = [
        i
        for i, line in enumerate(readme)
        if line.strip().lower().startswith("> [!") and "tip" in line.lower()
    ]
    require(alerts, "no Example callouts found in the README")

    bodies = [
        i
        for i, line in enumerate(readme)
        if line.lstrip().startswith(">") and "**Example**" in line
    ]
    require(
        len(bodies) == len(alerts),
        f"{len(alerts)} callouts but {len(bodies)} Example headings",
    )

    for index in alerts:
        window = readme[index + 1 : index + 8]
        require(
            any(line.lstrip().startswith("> ```") for line in window),
            f"the Example box at README line {index + 1} has no code block inside it",
        )
    text = "\n".join(readme)
    require(
        "examples/download_demo_data.py" in text,
        "the README does not say how to fetch the demo data its examples use",
    )
    require(
        "examples/demo/" in text,
        "the examples should point at examples/demo/",
    )
    return f"{len(alerts)} Example boxes, each with a command"


@check("the README's training example needs no checkpoint", "R6")
def test_readme_training_example() -> str:
    """The documented way to train must be a plain, self-contained command.

    It must not pass ``--model`` (a training run starts from scratch and takes only
    configuration from a checkpoint), and it must show the ten-epoch default being
    adjustable through ``--param``.
    """
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    # The training part has been titled "## Train" and "## 2. Train"; match the
    # heading by its word so renumbering the tutorial does not break this check.
    heading = re.search(r"(?m)^##+ .*Train.*$", readme)
    require(heading is not None, "the README has no Train section")
    section = readme[heading.end():]
    following = re.search(r"(?m)^## ", section)
    if following is not None:
        section = section[: following.start()]

    # The first fenced bash block after the heading is the primary example.
    blocks = []
    parts = section.split("```")
    for index in range(1, len(parts), 2):
        body = parts[index]
        if body.lstrip().startswith("bash"):
            blocks.append(body.split("\n", 1)[1] if "\n" in body else "")
    require(blocks, "no bash example found in the Train section")
    primary = blocks[0]
    require(
        "aurora run" in primary and "--mode training" in primary,
        f"the first example should be a training run: {primary}",
    )
    require(
        "--model" not in primary,
        "the primary training example must not pass --model; training starts from "
        f"scratch. Found:\n{primary}",
    )
    require(
        "--param" in section and "num_epochs" in section,
        "the Train section should show how to change the epoch budget",
    )
    require(
        "10" in section,
        "the Train section should state the default number of epochs",
    )
    return "primary training example passes no --model and documents the epoch knob"


@check("patch dumps resolve under either name and either barcode spelling", "R6")
def test_patch_dump_naming(tmp: Path) -> str:
    """Two naming disagreements have to be tolerated, and neither by copying files.

    * HEST's ``dump_patches`` writes the dataset ``barcodes``; ``image_utils`` used to
      read ``barcode``;
    * ``preprocess`` names the file ``<Sample>.h5`` while feature extraction asks for
      ``<Image_name stem>.h5``.

    Both are resolved without creating a second copy of the dump.
    """
    try:
        import h5py
        import numpy as np
    except ImportError:
        return "skipped: h5py not installed"

    from aurora.commands import _require_hest_patches
    from aurora.config import RunConfig
    from aurora.errors import ConfigError
    from aurora.paths import RunLayout
    from aurora.samples import Sample, SampleSheet
    from aurora.spec import StageContext

    project = build_project(tmp)
    layout = RunLayout.from_root(project / "aurora_runs" / "naming", project)
    layout.ensure("training")
    patches = layout.root / "processed_data" / "patches"
    patches.mkdir(parents=True)
    config = RunConfig(project_path=project, run_name="naming", mode="training")
    ctx = StageContext(
        layout=layout,
        samples=SampleSheet(
            path=None,
            samples=[
                Sample(
                    sample_name="BRCA1",
                    image_name="V1_Breast_Cancer_Block_A_Section_1_image.tif",
                )
            ],
            columns=["Sample", "Dir_name", "Image_name"],
        ),
        options=config.stage_options(),
        config=config,
    )
    stem = "V1_Breast_Cancer_Block_A_Section_1_image"

    def dump(name: str, barcode_key: str) -> None:
        with h5py.File(patches / f"{name}.h5", "w") as handle:
            handle.create_dataset(barcode_key, data=np.array([b"bc1"]))
            dataset = handle.create_dataset(
                "img", data=np.zeros((1, 4, 4, 3), dtype="uint8")
            )
            for key, value in (
                ("downsample", 1.0),
                ("patch_size_src", 224.0),
                ("patch_size_target", 224.0),
                ("pixel_size", 0.5),
            ):
                dataset.attrs[key] = value
            handle.create_dataset("coords", data=np.zeros((1, 2)))

    # HEST's spelling, named by Sample: the case that failed.
    dump("BRCA1", "barcodes")
    _require_hest_patches(ctx)
    require(
        sorted(p.name for p in patches.iterdir()) == ["BRCA1.h5"],
        f"no second file may be created: {sorted(p.name for p in patches.iterdir())}",
    )

    # The older singular spelling still works.
    dump("BRCA1", "barcode")
    _require_hest_patches(ctx)

    # A dump with no attributes on 'img' is tolerated, not refused: the reader falls
    # back to file attributes, the metadata/<name>.json sidecar, then derived values.
    with h5py.File(patches / "BRCA1.h5", "w") as handle:
        handle.create_dataset("barcodes", data=np.array([b"bc1"]))
        handle.create_dataset("img", data=np.zeros((1, 4, 4, 3), dtype="uint8"))
        handle.create_dataset("coords", data=np.zeros((1, 2)))
    _require_hest_patches(ctx)

    # A dump named by the image stem is found without a Sample-named copy.
    (patches / "BRCA1.h5").unlink()
    dump(stem, "barcodes")
    _require_hest_patches(ctx)

    # Neither name present is still refused, and says what it looked for.
    (patches / f"{stem}.h5").unlink()
    try:
        _require_hest_patches(ctx)
    except ConfigError as exc:
        require("missing" in exc.message, exc.message[:300])
        require("BRCA1.h5" in exc.message, exc.message[:300])
    else:
        require(False, "a dump that was never produced must be reported")
    return "barcodes/barcode and <Sample>/<stem> both accepted, nothing copied"


@check("training feature extraction checks the HEST patch dumps", "R6")
def test_hest_patch_preflight(tmp: Path) -> str:
    """A bad patch dump used to surface as a bare h5py ``KeyError``.

    ``code/extract_features.py`` opens ``<run>/processed_data/patches/<Image_name>.h5``
    and indexes ``barcode``/``img``/``coords`` plus four attributes on ``img``.
    Anything else raises ``KeyError: object 'barcode' doesn't exist`` from deep inside
    h5py, naming neither the file nor what it does contain.
    """
    try:
        import h5py
        import numpy as np
    except ImportError:
        return "skipped: h5py not installed"

    from aurora.commands import _require_hest_patches
    from aurora.config import RunConfig
    from aurora.errors import ConfigError
    from aurora.paths import RunLayout
    from aurora.samples import Sample, SampleSheet
    from aurora.spec import StageContext

    project = build_project(tmp)
    layout = RunLayout.from_root(project / "aurora_runs" / "patches", project)
    layout.ensure("training")
    patches = layout.root / "processed_data" / "patches"
    patches.mkdir(parents=True)
    config = RunConfig(project_path=project, run_name="patches", mode="training")
    ctx = StageContext(
        layout=layout,
        samples=SampleSheet(
            path=None,
            samples=[Sample(sample_name="BRCA1", image_name="BRCA1.tif")],
            columns=["Image_name", "Sample_name"],
        ),
        options=config.stage_options(),
        config=config,
    )
    target = patches / "BRCA1.h5"

    def write(keys=("barcode", "img", "coords"), attrs=True) -> None:
        with h5py.File(target, "w") as handle:
            if "barcode" in keys:
                handle.create_dataset("barcode", data=np.array([b"bc1"]))
            dataset = handle.create_dataset("img", data=np.zeros((1, 4, 4, 3), dtype="uint8"))
            if attrs:
                for key, value in (
                    ("downsample", 1.0),
                    ("patch_size_src", 224.0),
                    ("patch_size_target", 224.0),
                    ("pixel_size", 0.5),
                ):
                    dataset.attrs[key] = value
            if "coords" in keys:
                handle.create_dataset("coords", data=np.zeros((1, 2)))

    def expect_failure(label: str) -> str:
        try:
            _require_hest_patches(ctx)
        except ConfigError as exc:
            return exc.format()
        raise AssertionError(f"{label} should have been refused")

    # A complete dump passes.
    write()
    _require_hest_patches(ctx)

    write(keys=("img", "coords"))
    message = expect_failure("a dump without barcode")
    require("barcode" in message, message[:300])
    require("coords, img" in message, f"it must list the datasets present: {message[:300]}")

    # Missing attributes on 'img' are tolerated: HEST does not always write them and
    # the reader falls back to file attrs, a metadata sidecar, then derived values.
    write(attrs=False)
    _require_hest_patches(ctx)

    target.unlink()
    write()
    target.rename(patches / "BRCA1_rep1.h5")
    message = expect_failure("a dump named by the Sample column")
    require("missing" in message and "BRCA1.h5" in message, message[:300])
    require(
        "BRCA1_rep1.h5" in message,
        f"it must show the files that are there: {message[:300]}",
    )
    # And a dry run never fails on this: it is a plan, not a run.
    dry = StageContext(
        layout=layout,
        samples=ctx.samples,
        options=RunConfig(
            project_path=project, run_name="patches", mode="training", dry_run=True
        ).stage_options(),
        config=RunConfig(project_path=project, run_name="patches", mode="training"),
    )
    _require_hest_patches(dry)
    return "missing dataset, missing attrs and wrong name all reported with detail"


@check("a training run creates only the directories it fills", "R4")
def test_mode_scoped_layout(tmp: Path) -> str:
    """A training run used to be littered with inference directories.

    ``RunLayout.ensure()`` created every directory, and ``write_layout_stamp`` called
    it unconditionally, so ``aurora_runs/lung_train`` contained empty
    ``01_segmentation``, ``03_predictions``, ``04_histosweep`` and ``05_plots``.
    """
    from aurora.paths import RunLayout, write_layout_stamp

    project = build_project(tmp)
    training = RunLayout.from_root(project / "aurora_runs" / "train_layout", project)
    produced = {path.name for path in training.directories_for("training")}
    for unwanted in ("01_segmentation", "03_predictions", "04_histosweep", "05_plots"):
        require(
            unwanted not in produced,
            f"a training run must not create {unwanted}: {sorted(produced)}",
        )
    for needed in ("02_features", "checkpoints", "config", "logs", "state", "tmp"):
        require(needed in produced, f"a training run still needs {needed}: {sorted(produced)}")

    inference = {path.name for path in training.directories_for("inference")}
    for expected in ("01_segmentation", "03_predictions", "04_histosweep", "05_plots"):
        require(expected in inference, f"inference still needs {expected}: {sorted(inference)}")

    # The stamp must not drag the whole layout back in.
    training.ensure("training")
    write_layout_stamp(training)
    created = {path.name for path in training.root.iterdir() if path.is_dir()}
    for unwanted in ("01_segmentation", "03_predictions", "04_histosweep", "05_plots"):
        require(
            unwanted not in created,
            f"write_layout_stamp recreated {unwanted}: {sorted(created)}",
        )
    require((training.state_dir / "layout.json").is_file(), "the layout stamp is still written")

    # Without a mode the full layout is still created (used by migrate-layout).
    full = RunLayout.from_root(project / "aurora_runs" / "full_layout", project).ensure()
    full_names = {path.name for path in full.root.iterdir() if path.is_dir()}
    for expected in ("01_segmentation", "03_predictions", "04_histosweep", "05_plots"):
        require(expected in full_names, f"ensure() without a mode must create {expected}")
    return "training: 6 directories; inference: the full layout"


@check("training does not inherit inference-only prerequisites", "R6")
def test_mode_aware_prerequisites(tmp: Path) -> str:
    """`rescale` needs a tissue mask only when it runs for inference.

    The inference invocation passes ``--seg_path`` from `segment`; the training one
    works from ``processed_data`` and passes no mask. A single unconditional
    prerequisite dragged `segment` -- and its requirement for a ``wsis/`` directory
    of slides -- into every training run, which failed on data already in HEST-1K
    format.
    """
    from aurora.paths import RunLayout
    from aurora.spec import (
        INFERENCE_PIPELINE,
        TRAINING_PIPELINE,
        prerequisites_for,
        resolve_execution_order,
    )
    from aurora.state import RunState, effective_prerequisites

    require(
        prerequisites_for("rescale", "inference") == ("segment",),
        "inference rescale must still require the mask",
    )
    require(
        prerequisites_for("rescale", "training") == (),
        "training rescale passes no mask, so it must require nothing",
    )
    # Everything else keeps its prerequisites in both modes.
    for stage in ("features_lowres", "prepare_st", "train_lowres", "histosweep"):
        require(
            prerequisites_for(stage, "training") == prerequisites_for(stage, "inference")
            or stage == "features_lowres",
            f"{stage} should not differ between modes",
        )

    training = resolve_execution_order(TRAINING_PIPELINE, True, "training")
    require("segment" not in training, f"the training chain must not include segment: {training}")
    inference = resolve_execution_order(INFERENCE_PIPELINE, True, "inference")
    require("segment" in inference, f"the inference chain must include segment: {inference}")

    # The gate reads the mode recorded in the run, not a default.
    layout = RunLayout.from_root(build_project(tmp) / "aurora_runs" / "modes", build_project(tmp))
    layout.ensure()
    state = RunState.load(layout, create=True)
    state.set_mode("training")
    require(
        effective_prerequisites(state, "rescale") == [],
        f"training gate must not require segment: {effective_prerequisites(state, 'rescale')}",
    )
    state.set_mode("inference")
    require(
        effective_prerequisites(state, "rescale") == ["segment"],
        f"inference gate must require segment: {effective_prerequisites(state, 'rescale')}",
    )
    return "rescale: (segment) for inference, () for training"


@check("no stage script reaches the network at import", "R3")
def test_stage_scripts_are_lazy_about_biomart() -> str:
    """The Ensembl map is fetched on use, not at import.

    ``code/prepare_st.py`` and ``code/smooth_cnt.py`` both called BioMart before
    reading a sample, so a service that answered with an HTML error page killed the
    training chain for Visium data whose genes are already symbols and need no mapping
    at all. Every stage script is checked, so a new one cannot reintroduce it.
    """
    import ast

    def calls_biomart(node: ast.AST) -> bool:
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                target = child.func
                name = getattr(target, "attr", getattr(target, "id", ""))
                if name in ("biomart_annotations", "simple_query"):
                    return True
        return False

    checked = []
    offenders = []
    for path in sorted((REPO_ROOT / "code").glob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "biomart_annotations" not in text:
            continue
        checked.append(path.name)
        tree = ast.parse(text)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue  # deferred until called, which is the point
            if calls_biomart(node):
                offenders.append(f"{path.name}:{getattr(node, 'lineno', 0)}")
        functions = {
            child.name for child in ast.walk(tree) if isinstance(child, ast.FunctionDef)
        }
        require(
            "_load_gene_symbols" in functions,
            f"{path.name} should load the Ensembl map from a function",
        )

    require(checked, "no stage script declares a BioMart lookup any more")
    require(not offenders, f"BioMart is reached at import time in: {offenders}")
    return f"{', '.join(checked)} only query BioMart when a sample stores Ensembl IDs"


@check("a run without a checkpoint still knows its patch sizes", "R3")
def test_no_checkpoint_params_reach_the_context(tmp: Path) -> str:
    """Training from ``--gene-list``/``--cell-types``/``--norm-factors`` has no checkpoint.

    The parameters then come from ``aurora/defaults.py`` and preparation writes them
    into the run.  The stage context used to be built from an empty dict, so the first
    stage that needed a patch size died with "the combined checkpoint does not record
    the low-resolution patch sizes" -- in a run that has no checkpoint at all.
    """
    from aurora.config import RunConfig
    from aurora.defaults import DEFAULT_HIGHRES_PARAMS, DEFAULT_LOWRES_PARAMS
    from aurora.paths import RunLayout
    from aurora.pipeline import build_context
    from aurora.prepare import load_run_params, materialize_default_params
    from aurora.samples import Sample, SampleSheet

    project = build_project(tmp)
    layout = RunLayout.from_root(project / "aurora_runs" / "nockpt", project)
    layout.ensure("training")
    config = RunConfig(
        project_path=project,
        run_name="nockpt",
        mode="training",
        param_overrides={"num_epochs": 10},
    )

    # Before preparation the packaged defaults answer, with --param applied.
    lowres, _highres = load_run_params(layout, config)
    require(
        lowres.get("patch_size") == DEFAULT_LOWRES_PARAMS["patch_size"],
        f"defaults must supply the low-res patch sizes, got {lowres.get('patch_size')}",
    )
    require(
        lowres.get("num_epochs") == 10,
        f"--param must survive the fallback, got {lowres.get('num_epochs')}",
    )

    materialize_default_params(layout, config)
    sheet = SampleSheet(
        path=None,
        samples=[Sample(sample_name="BRCA1", image_name="image1.tif")],
        columns=["Sample", "Dir_name", "Image_name"],
    )
    # No checkpoint: exactly the situation that used to raise.
    context = build_context(layout, sheet, config, checkpoint=None)
    require(
        context.lowres_patch_sizes == list(DEFAULT_LOWRES_PARAMS["patch_size"]),
        f"context should see the low-res patch sizes, got {context.lowres_patch_sizes}",
    )
    require(
        context.highres_patch_sizes == list(DEFAULT_HIGHRES_PARAMS["patch_size"]),
        f"context should see the high-res patch sizes, got {context.highres_patch_sizes}",
    )
    return (
        f"lowres {context.lowres_patch_sizes}, highres {context.highres_patch_sizes}, "
        "no checkpoint needed"
    )


@check("the merge stage is handed paths it can open", "R3")
def test_combine_command_paths(tmp: Path) -> str:
    """The merge tool runs with the run directory as cwd, so asset paths must move with it.

    It was given ``--gene-list examples/BRCA_gene.csv`` verbatim, which does not exist
    from inside ``aurora_runs/<run>``: the whole training chain finished and then failed
    with "cannot read text asset from PosixPath".
    """
    from aurora.commands import build_commands
    from aurora.config import RunConfig
    from aurora.paths import RunLayout
    from aurora.pipeline import build_context
    from aurora.samples import Sample, SampleSheet

    # Real files in the repository, named the way a user would type them.
    raw = {
        "gene-list": "examples/BRCA_gene.csv",
        "cell-types": "examples/BRCA_cell_types.csv",
        "norm-factors": "examples/BRCA_gene_normalize_factor.csv",
    }
    project = build_project(tmp)
    layout = RunLayout.from_root(project / "aurora_runs" / "merge", project)
    layout.ensure("training")
    config = RunConfig(
        project_path=project,
        run_name="merge",
        mode="training",
        gene_list_path=raw["gene-list"],
        cell_type_path=raw["cell-types"],
        norm_factor_path=raw["norm-factors"],
    )
    sheet = SampleSheet(
        path=None,
        samples=[Sample(sample_name="BRCA1", image_name="image1.tif")],
        columns=["Sample", "Dir_name", "Image_name"],
    )
    context = build_context(layout, sheet, config, checkpoint=None)
    argv = build_commands("combine_checkpoints", context)[0].argv
    require(
        any(str(part).endswith("merge_aurora_checkpoints.py") for part in argv),
        f"unexpected command: {argv[:2]}",
    )

    for flag, typed in raw.items():
        value = argv[argv.index(f"--{flag}") + 1]
        opened = Path(value) if Path(value).is_absolute() else context.layout.root / value
        require(
            opened.resolve().is_file(),
            f"--{flag} {value!r} does not open from the run directory "
            f"(the user typed {typed!r})",
        )
    return "merge inputs resolve from the run directory"


@check("the ready-to-use checkpoint is called aurora.pth", "R6")
def test_combined_checkpoint_name(tmp: Path) -> str:
    """The final model and the downloaded one are both ``aurora.pth``.

    Runs that finished under the old name keep working: ``aurora_combined.pth`` is still
    accepted as this stage's output, so an existing run is not reported as unfinished.
    """
    from aurora.config import RunConfig
    from aurora.paths import RunLayout
    from aurora.pipeline import build_context
    from aurora.samples import Sample, SampleSheet
    from aurora.spec import (
        DEFAULT_COMBINED_NAME,
        LEGACY_COMBINED_NAME,
        _combined_checkpoint_outputs,
    )

    require(
        DEFAULT_COMBINED_NAME == "aurora.pth",
        f"the final model should be aurora.pth, got {DEFAULT_COMBINED_NAME}",
    )
    require(
        LEGACY_COMBINED_NAME == "aurora_combined.pth",
        f"the old name should still be recognised, got {LEGACY_COMBINED_NAME}",
    )

    project = build_project(tmp)
    layout = RunLayout.from_root(project / "aurora_runs" / "named", project)
    layout.ensure("training")
    config = RunConfig(project_path=project, run_name="named", mode="training")
    sheet = SampleSheet(
        path=None,
        samples=[Sample(sample_name="BRCA1", image_name="image1.tif")],
        columns=["Sample", "Dir_name", "Image_name"],
    )
    context = build_context(layout, sheet, config, checkpoint=None)
    checkpoints = layout.checkpoints_dir

    # Nothing written yet: the stage declares the new name.
    (name,) = _combined_checkpoint_outputs(context)
    require(
        name.name == "aurora.pth",
        f"the combine stage should write aurora.pth, got {name.name}",
    )

    # A run finished before the rename is still recognised.
    legacy = checkpoints / "aurora_combined.pth"
    legacy.write_bytes(b"old")
    (name,) = _combined_checkpoint_outputs(context)
    require(
        name.name == "aurora_combined.pth",
        f"an old run should still count as complete, got {name.name}",
    )

    # Once the new file exists it wins.
    (checkpoints / "aurora.pth").write_bytes(b"new")
    (name,) = _combined_checkpoint_outputs(context)
    require(
        name.name == "aurora.pth",
        f"the new name should take precedence, got {name.name}",
    )

    # The downloader uses the same name.
    import importlib.util

    script = REPO_ROOT / "examples" / "download_demo_data.py"
    spec = importlib.util.spec_from_file_location("downloader_named", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(
        module.COMBINED_NAME == "aurora.pth",
        f"the downloader should produce aurora.pth, got {module.COMBINED_NAME}",
    )

    # And nothing else still offers the old name as a default.
    offenders = []
    for path in sorted((REPO_ROOT / "aurora").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if LEGACY_COMBINED_NAME in text and path.name != "spec.py":
            offenders.append(path.name)
    require(not offenders, f"the old name is still defaulted in: {offenders}")
    return "aurora.pth everywhere; old runs still recognised"


@check("the training defaults match the shipped parameter files", "R6")
def test_default_epochs(tmp: Path) -> str:
    """Two phases per resolution: 400 + 200 epochs at low res, 100 + 100 at high res.

    ``num_epochs`` is the total and ``exp_epochs`` is where the expression phase ends
    and the cell-type phase begins.  These defaults mirror the ``model_parameter_*.json``
    files that ship with the checkpoint, so a run configured either way trains the same
    amount.
    """
    import json

    from aurora.config import RunConfig
    from aurora.defaults import DEFAULT_HIGHRES_PARAMS, DEFAULT_LOWRES_PARAMS
    from aurora.paths import RunLayout
    from aurora.prepare import materialize_default_params

    expected = {
        "lowres": (400, 200),
        "highres": (100, 100),
    }
    for stage, (expression, celltype) in expected.items():
        params = DEFAULT_LOWRES_PARAMS if stage == "lowres" else DEFAULT_HIGHRES_PARAMS
        require(
            params["exp_epochs"] == expression,
            f"{stage} should train {expression} expression epochs, got {params['exp_epochs']}",
        )
        require(
            params["num_epochs"] - params["exp_epochs"] == celltype,
            f"{stage} should train {celltype} cell-type epochs, got "
            f"{params['num_epochs'] - params['exp_epochs']}",
        )

    # Every other default must equal the shipped parameter file, so the two ways of
    # configuring a run cannot drift apart.
    assets = ("gene_list_file", "gene_norm_factor_file", "celltype_file")
    for stage, filename, mine in (
        ("lowres", "model_parameter_LUAD.json", DEFAULT_LOWRES_PARAMS),
        ("highres", "model_parameter_LUAD_highres.json", DEFAULT_HIGHRES_PARAMS),
    ):
        shipped_path = REPO_ROOT / "trained_model" / filename
        require(shipped_path.is_file(), f"missing {shipped_path}")
        shipped = {
            key: value
            for key, value in json.loads(shipped_path.read_text(encoding="utf-8")).items()
            if key not in assets
        }
        differing = {
            key: (shipped.get(key), mine.get(key))
            for key in set(shipped) | set(mine)
            if shipped.get(key) != mine.get(key)
        }
        require(not differing, f"{stage} defaults differ from {filename}: {differing}")

    # Written into a run, and --param still overrides.
    project = build_project(tmp)
    layout = RunLayout.from_root(project / "aurora_runs" / "epochs", project)
    layout.ensure("training")
    materialize_default_params(
        layout, RunConfig(project_path=project, run_name="epochs", mode="training")
    )
    for relative, total in (
        ("config/params_lowres.json", 600),
        ("checkpoints/params_highres.json", 200),
    ):
        data = json.loads((layout.root / relative).read_text(encoding="utf-8"))
        require(
            data["num_epochs"] == total,
            f"{relative} should hold {total} epochs, got {data['num_epochs']}",
        )

    other = RunLayout.from_root(project / "aurora_runs" / "epochs2", project)
    other.ensure("training")
    materialize_default_params(
        other,
        RunConfig(
            project_path=project,
            run_name="epochs2",
            mode="training",
            param_overrides={"num_epochs": 50},
        ),
    )
    data = json.loads((other.config_dir / "params_lowres.json").read_text(encoding="utf-8"))
    require(data["num_epochs"] == 50, f"--param must win, got {data['num_epochs']}")
    return "lowres 400+200, highres 100+100, identical to the shipped JSONs"


@check("training runs from configuration, not from someone's weights", "R6")
def test_training_without_checkpoint(tmp: Path) -> str:
    """No training stage loads the checkpoint's weights.

    ``train.py`` and ``train_highres.py`` build their networks and save their own
    ``.pth``; the only ``load_state_dict`` in the chain loads the low-resolution
    model *this run* just trained.  So a checkpoint contributes configuration only,
    and supplying it as three CSVs should be enough.
    """
    import json

    from aurora.config import RunConfig
    from aurora.defaults import DEFAULT_HIGHRES_PARAMS, DEFAULT_LOWRES_PARAMS
    from aurora.errors import ConfigError
    from aurora.paths import RunLayout
    from aurora.prepare import (
        PARAMS_ASSET_KEYS,
        materialize_asset_overrides,
        materialize_default_params,
    )

    # The defaults must cover every key the two scripts read.
    for default, expected_dim in ((DEFAULT_LOWRES_PARAMS, 1536), (DEFAULT_HIGHRES_PARAMS, 3072)):
        require(
            default["img_emb_dim"] == expected_dim,
            f"img_emb_dim should be {expected_dim}, got {default['img_emb_dim']}",
        )
        for key in ("num_epochs", "batch_size", "lr", "patch_size", "pred_celltype"):
            require(key in default, f"{key} is missing from the defaults")

    project = build_project(tmp)
    layout = RunLayout.from_root(project / "aurora_runs" / "scratch", project)
    layout.ensure()

    # Writing the parameter files needs no checkpoint.
    config = RunConfig(project_path=project, run_name="scratch", mode="training")
    written = materialize_default_params(layout, config)
    require(
        set(written) == {"params_lowres", "params_highres", "params_highres_training"},
        written,
    )
    lowres = json.loads((layout.config_dir / "params_lowres.json").read_text(encoding="utf-8"))
    highres = json.loads(
        (layout.checkpoints_dir / "params_highres.json").read_text(encoding="utf-8")
    )
    require(lowres["img_emb_dim"] == 1536 and highres["img_emb_dim"] == 3072, "per-stage dims")
    # prepare_highres_training rebuilds the low-res network, so its file must hold the
    # low-res architecture with checkpoints-relative asset paths.
    arch = json.loads(
        (layout.checkpoints_dir / "params_highres_training.json").read_text(encoding="utf-8")
    )
    require(
        arch["img_emb_dim"] == 1536,
        f"prepare_highres_training must see the low-res architecture, got {arch['img_emb_dim']}",
    )
    for _asset, param_key in PARAMS_ASSET_KEYS.items():
        require(
            "/" not in arch[param_key],
            f"its {param_key} must be checkpoints-relative, got {arch[param_key]!r}",
        )
    # Asset paths still follow each script's own base.
    for _asset, param_key in PARAMS_ASSET_KEYS.items():
        require(
            lowres[param_key].startswith("checkpoints/"),
            f"lowres {param_key} must be run-root relative, got {lowres[param_key]!r}",
        )
        require(
            "/" not in highres[param_key],
            f"highres {param_key} must be checkpoints-relative, got {highres[param_key]!r}",
        )

    # --param still reaches both.
    overridden = RunConfig(
        project_path=project, run_name="scratch", mode="training", param_overrides={"num_epochs": 3}
    )
    materialize_default_params(layout, overridden)
    for path in (layout.config_dir / "params_lowres.json", layout.checkpoints_dir / "params_highres.json"):
        require(
            json.loads(path.read_text(encoding="utf-8"))["num_epochs"] == 3,
            f"--param must reach {path.name}",
        )

    # The assets are required, and the error says which are missing.
    try:
        materialize_asset_overrides(layout, config)
    except ConfigError as exc:
        message = exc.format()
        require("--gene-list" in message and "--cell-types" in message, message[:300])
        require("--norm-factors" in message, message[:300])
    else:
        require(False, "assets must be required when there is no checkpoint")

    # Supplied, they are copied into checkpoints/ under the names the scripts read.
    assets = tmp / "my_assets"
    assets.mkdir()
    paths = {}
    for flag, filename, header in (
        ("gene_list", "genes.csv", "Genes\nKRT8\n"),
        ("cell_type", "celltypes.csv", "Celltype\nMalignant\n"),
        ("norm_factor", "norm.csv", "Gene,Factor\nKRT8,1.0\n"),
    ):
        path = assets / filename
        path.write_text(header, encoding="utf-8")
        paths[flag] = path
    filled = RunConfig(
        project_path=project,
        run_name="scratch",
        mode="training",
        gene_list_path=paths["gene_list"],
        cell_type_path=paths["cell_type"],
        norm_factor_path=paths["norm_factor"],
    )
    report = materialize_asset_overrides(layout, filled)
    require(set(report) == {"gene_list", "cell_types", "gene_norm_factor"}, report)
    for name in ("gene_list.csv", "cell_types.csv", "gene_normalize_factor.csv"):
        require((layout.checkpoints_dir / name).is_file(), f"{name} was not written")
    return "parameters from aurora/defaults.py, assets from CSV; no checkpoint involved"


@check("training parameters: genes, cell-type head, overrides", "R6")
def test_training_parameters(tmp: Path) -> str:
    """The three things a user configures without touching a JSON by hand.

    * which genes: ``--gene-list`` (copied to ``checkpoints/gene_list.csv``) or
      the checkpoint's bundled panel;
    * the cell-type head: ``--pred-celltype`` / ``--no-pred-celltype``;
    * any other parameter: ``--param KEY=VALUE``.

    Whatever route is taken has to end up in the file the script actually opens.
    """
    import json

    from aurora.config import RunConfig
    from aurora.errors import ConfigError
    from aurora.paths import RunLayout
    from aurora.prepare import (
        PARAMS_ASSET_KEYS,
        PARAMS_TYPES,
        _write_stage_params,
        parse_param_overrides,
        stage_params,
    )

    # -- parsing
    require(PARAMS_TYPES, "the overridable parameter table is empty")
    parsed = parse_param_overrides(["num_epochs=42", "lr=5e-4", "pred_celltype=false"])
    require(parsed == {"num_epochs": 42, "lr": 0.0005, "pred_celltype": False}, f"{parsed}")
    for bad, why in (("num_epochs", "no ="), ("num_epochs=abc", "not an int"), ("nope=1", "unknown")):
        try:
            parse_param_overrides([bad])
        except ConfigError:
            pass
        else:
            require(False, f"--param {bad!r} ({why}) should be rejected")
    try:
        parse_param_overrides(["pred_celltype=maybe"])
    except ConfigError:
        pass
    else:
        require(False, "a non-boolean value for pred_celltype should be rejected")

    # -- asset paths are rewritten per stage, because the two script families
    #    resolve them against different roots
    checkpoint_lowres = {"num_epochs": 600, "gene_list_file": "LUAD_gene.csv"}
    checkpoint_highres = {"num_epochs": 200, "gene_list_file": "/LUAD_gene.csv"}
    lowres = stage_params(checkpoint_lowres, "lowres", {})
    highres = stage_params(checkpoint_highres, "highres", {})
    for asset_key, param_key in PARAMS_ASSET_KEYS.items():
        require(
            lowres[param_key] == f"checkpoints/{_asset_filename(asset_key)}",
            f"lowres {param_key} must be run-root relative, got {lowres[param_key]!r}",
        )
        require(
            highres[param_key] == _asset_filename(asset_key),
            f"highres {param_key} must be checkpoints-relative, got {highres[param_key]!r}",
        )
    require(lowres["num_epochs"] == 600, "the rest of the parameters are untouched")

    # -- the file: hand edits survive, overrides are applied, refresh resets
    project = build_project(tmp)
    layout = RunLayout.from_root(project / "aurora_runs" / "params", project)
    layout.ensure()
    destination = layout.config_dir / "params_lowres.json"
    _write_stage_params(destination, stage_params(checkpoint_lowres, "lowres", {}), RunConfig())
    require(destination.is_file(), "the parameter file is written")

    edited = json.loads(destination.read_text(encoding="utf-8"))
    edited["num_epochs"] = 7
    destination.write_text(json.dumps(edited), encoding="utf-8")
    _write_stage_params(destination, stage_params(checkpoint_lowres, "lowres", {}), RunConfig())
    require(
        json.loads(destination.read_text(encoding="utf-8"))["num_epochs"] == 7,
        "a hand-edited parameter must survive a re-run",
    )

    stale = json.loads(destination.read_text(encoding="utf-8"))
    stale["gene_list_file"] = "LUAD_gene.csv"
    destination.write_text(json.dumps(stale), encoding="utf-8")
    _write_stage_params(destination, stage_params(checkpoint_lowres, "lowres", {}), RunConfig())
    repaired = json.loads(destination.read_text(encoding="utf-8"))
    require(
        repaired["gene_list_file"] == "checkpoints/gene_list.csv",
        f"a stale asset path must be repaired, got {repaired['gene_list_file']!r}",
    )
    require(repaired["num_epochs"] == 7, "repairing an asset path must not reset other values")

    config = RunConfig(param_overrides={"pred_celltype": False, "batch_size": 32})
    detail = _write_stage_params(
        destination, stage_params(checkpoint_lowres, "lowres", config.param_overrides), config
    )
    applied = json.loads(destination.read_text(encoding="utf-8"))
    require(applied["pred_celltype"] is False, "--no-pred-celltype must reach the file")
    require(applied["batch_size"] == 32, "--param must reach the file")
    require("applied" in detail, f"the run should report what it applied: {detail}")

    _write_stage_params(
        destination,
        stage_params(checkpoint_lowres, "lowres", {}),
        RunConfig(refresh_params=True),
    )
    refreshed = json.loads(destination.read_text(encoding="utf-8"))
    require(
        refreshed["num_epochs"] == 600 and refreshed.get("batch_size") != 32,
        f"--refresh-params must go back to the checkpoint's values: {refreshed}",
    )
    return f"{len(PARAMS_TYPES)} overridable parameters; asset paths fixed per stage"


def _asset_filename(asset_key: str) -> str:
    from aurora.prepare import CHECKPOINT_ASSET_FILES

    return CHECKPOINT_ASSET_FILES[asset_key]


@check("cleanup removes only intermediates, never results", "R4")
def test_cleanup() -> str:
    """`aurora clean` is destructive, so its refusals matter more than its work."""
    script = REPO_ROOT / "tests" / "test_cleanup.py"
    require(script.is_file(), f"{script} is missing")
    result = subprocess.run(
        [sys.executable, "-B", str(script)], cwd=str(REPO_ROOT), capture_output=True, text=True
    )
    require(result.returncode == 0, (result.stdout + result.stderr)[-2500:])
    return "results survive both tiers; a run with no prediction is refused"


@check("an installed-but-unimportable package is reported missing", "R3")
def test_native_binding_probe(tmp: Path) -> str:
    """`pyvips` can be installed and still fail: it dlopens libvips on import.

    `importlib.util.find_spec` locates a package without executing it, so it
    reported pyvips as available and `preprocess` then died on its first line with
    `OSError: cannot load library 'libvips.so.42'`.  Dependencies marked
    `probe_import` are now imported for real.
    """
    import aurora.deps as deps

    site = tmp / "fake_site"
    site.mkdir()
    (site / "broken_native_binding.py").write_text(
        "raise OSError('cannot load library libvips.so.42')\n", encoding="utf-8"
    )
    sys.path.insert(0, str(site))
    try:
        # Without the probe, an installed-but-broken package looks fine -- this is
        # exactly the hole that let `preprocess` run and crash.
        shallow = deps.Dependency("broken_native_binding", "broken-native", "hest")
        require(
            deps.is_module_available("broken_native_binding"),
            "find_spec should find the module (it is installed)",
        )
        require(
            deps.resolve_import_name(shallow) == "broken_native_binding",
            "the shallow check cannot see the broken import -- which is the point",
        )

        # With it, the failure is caught and the reason is kept.
        probing = deps.Dependency(
            "broken_native_binding", "broken-native", "hest", probe_import=True
        )
        require(
            deps.resolve_import_name(probing) is None,
            "a probe_import dependency that raises must resolve to None",
        )
        reason = deps.import_error_for("broken_native_binding")
        require(
            "cannot load library" in reason,
            f"the failure reason must be kept for the message, got {reason!r}",
        )
    finally:
        if str(site) in sys.path:
            sys.path.remove(str(site))
        sys.modules.pop("broken_native_binding", None)

    # The packages that actually need this are marked, and say how to fix it.
    marked = [dep for dep in deps.DEPENDENCIES if dep.probe_import]
    names = sorted(dep.distribution for dep in marked)
    require("pyvips" in names, f"pyvips must be probed by import, marked: {names}")
    require("openslide-python" in names, f"openslide-python must be probed, marked: {names}")
    for dep in marked:
        require(dep.native_hint, f"{dep.distribution} is probed but gives no native-library hint")
        _group, hint = deps.install_hint(dep.import_name)
        require(
            dep.native_hint in hint,
            f"the hint for {dep.distribution} must include the native-library fix",
        )
    return f"{', '.join(names)} probed by real import; libvips guidance in the hint"


@check("source-only dependencies are never promised to pip", "R3")
def test_source_only_dependencies() -> str:
    """`hest` has no usable PyPI release, so pip must never be told to fetch it.

    Listing it as a requirement makes pip install an unrelated project of the same
    name (or fail), and `pip install .[hest]` therefore cannot provide it.  The
    policy marks such a dependency `installable=False`, and every generated
    artefact has to respect that.
    """
    import tomllib

    from aurora.deps import DEPENDENCIES, install_hint

    source_only = [dep for dep in DEPENDENCIES if not dep.installable]
    require(source_only, "at least one dependency is expected to be source-only (hest)")
    for dep in source_only:
        require(
            dep.source_command,
            f"{dep.distribution} is marked source-only but has no source_command to show",
        )

    # 1. no requirements file may carry it as a pip line
    for path in sorted((REPO_ROOT / "requirements").glob("*.txt")):
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("-"):
                continue
            for dep in source_only:
                require(
                    not stripped.lower().startswith(dep.distribution.lower()),
                    f"{path.name} tells pip to install '{dep.distribution}', which has no "
                    "usable PyPI release, so that line cannot succeed",
                )

    # 2. no packaging extra may list it
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    for extra, lines in data["project"].get("optional-dependencies", {}).items():
        for dep in source_only:
            require(
                not any(line.strip().lower() == dep.distribution.lower() for line in lines),
                f"the '{extra}' extra lists '{dep.distribution}', so 'pip install .[{extra}]' "
                "would fetch the wrong project",
            )

    # 3. the hint must show the source command, not a requirements file
    for dep in source_only:
        _group, hint = install_hint(dep.distribution)
        require(
            dep.source_command in hint,
            f"the hint for '{dep.distribution}' must give the source command: {hint}",
        )
        require(
            "pip install -r" not in hint,
            f"the hint for '{dep.distribution}' must not point at a requirements file",
        )
    return ", ".join(f"{d.distribution} -> source install" for d in source_only)


@check("preprocess reads raw_data from the project, not the run", "R4")
def test_raw_data_resolution(tmp: Path) -> str:
    """`HEST_preprocess.py` joins `-p` and `-r`, so `-r` must be absolute.

    With `-p` pointing at the run directory (so `-o processed_data` lands in the
    run), a relative `-r raw_data` resolved to `aurora_runs/<run>/raw_data` and the
    stage died on `filtered_feature_bc_matrix.h5`.  The Visium folders are an
    *input* and live in the project.
    """
    import subprocess as sp

    from aurora.commands import build_commands
    from aurora.config import RunConfig
    from aurora.errors import ConfigError
    from aurora.paths import RunLayout
    from aurora.samples import Sample, SampleSheet
    from aurora.spec import StageContext

    project = build_project(tmp)
    raw = project / "raw_data"
    (raw / "BRCA1").mkdir(parents=True)
    (raw / "BRCA1" / "filtered_feature_bc_matrix.h5").write_bytes(b"")
    config = RunConfig(project_path=project, run_name="rawd", mode="training")
    require(
        config.resolved_raw_data() == raw.resolve(),
        f"resolved_raw_data should find {raw}, got {config.resolved_raw_data()}",
    )

    layout = RunLayout.from_root(project / "aurora_runs" / "rawd", project)
    layout.ensure()

    def context(**options) -> StageContext:
        cfg = RunConfig(project_path=project, run_name="rawd", mode="training", **options)
        return StageContext(
            layout=layout,
            samples=SampleSheet(
                path=None,
                samples=[Sample(sample_name="BRCA1", image_name="BRCA1.tif")],
                columns=["Image_name", "Sample_name"],
            ),
            options=cfg.stage_options(),
            config=cfg,
        )

    argv = [str(a) for a in build_commands("preprocess", context())[0].argv]
    given_p = Path(argv[argv.index("-p") + 1])
    given_r = argv[argv.index("-r") + 1]
    given_m = argv[argv.index("-m") + 1]
    given_o = argv[argv.index("-o") + 1]
    # The script concatenates every path onto -p:
    #   f"{project_path}/{raw_data_path}", and the same for -m and -o.
    # So assert the *join*, not the individual values: an absolute value here is
    # what produced "<run>/abs/path/raw_data".
    require(
        (given_p / given_r).resolve() == raw.resolve(),
        f"-p/-r must resolve to the project's raw_data; "
        f"got {(given_p / given_r).resolve()} vs {raw.resolve()}",
    )
    require(
        (given_p / given_o).resolve() == (layout.root / "processed_data").resolve(),
        f"-p/-o must resolve into the run, got {(given_p / given_o).resolve()}",
    )
    require(
        (given_p / given_m).resolve().is_file() or (given_p / given_m).name == "train_samples.csv",
        f"-p/-m must resolve to the run's training sheet, got {(given_p / given_m)}",
    )
    for flag, value in (("-r", given_r), ("-m", given_m), ("-o", given_o)):
        require(
            not Path(value).is_absolute(),
            f"{flag} must be relative to -p, otherwise the script doubles the path: {value}",
        )
    require(
        "raw_data" not in layout.root.as_posix(),
        "raw_data must never be fabricated inside the run directory",
    )

    # An absolute --raw-data outside the project is still reached, via ..
    other = tmp / "elsewhere_visium"
    (other / "S1").mkdir(parents=True, exist_ok=True)
    elsewhere = [
        str(a) for a in build_commands("preprocess", context(raw_data_path=str(other)))[0].argv
    ]
    require(
        (Path(elsewhere[elsewhere.index("-p") + 1]) / elsewhere[elsewhere.index("-r") + 1])
        .resolve()
        == other.resolve(),
        f"an absolute --raw-data must still resolve, got {elsewhere[elsewhere.index('-r') + 1]!r}",
    )

    # Missing data is a clear error naming where it looked, not a FileNotFoundError
    # from deep inside the script.
    empty = tmp / "no_visium"
    empty.mkdir(exist_ok=True)
    cfg = RunConfig(project_path=empty, run_name="rawd", mode="training")
    layout2 = RunLayout.from_root(empty / "aurora_runs" / "rawd", empty)
    layout2.ensure()
    ctx2 = StageContext(
        layout=layout2,
        samples=SampleSheet(
            path=None,
            samples=[Sample(sample_name="S1", image_name="S1.tif")],
            columns=["Image_name", "Sample_name"],
        ),
        options=cfg.stage_options(),
        config=cfg,
    )
    try:
        build_commands("preprocess", ctx2)
    except ConfigError as exc:
        message = exc.format()
        require("raw Visium data directory" in message, message[:200])
        require("--raw-data" in message, "the hint must name the option")
    else:
        require(False, "a missing raw_data directory should be refused")
    # A dry run still plans, so it can show the commands.
    dry = [str(a) for a in build_commands("preprocess", context(dry_run=True))[0].argv]
    require("-r" in dry, "a dry run must still build the command")
    return f"-p={given_p.name} -r={given_r} -o={given_o} (concatenation verified)"


@check("--model is only required by stages that read it", "R3")
def test_model_optional_for_preprocess(tmp: Path) -> str:
    """`preprocess` converts raw Visium and uses nothing from the checkpoint.

    Demanding a `.pth` for it was friction with no purpose, so the requirement is
    now per stage.  Everything that reads the checkpoint -- model parameters,
    bundled gene list / cell types / normalisation factors, or weights -- keeps
    it.
    """
    from aurora.spec import requires_model, stages_needing_model

    require(not requires_model(["preprocess"]), "preprocess must not require a checkpoint")
    require(
        requires_model(["prepare_st"]),
        "prepare_st reads the gene list from the checkpoint, so it must require one",
    )
    require(
        "segment" in stages_needing_model(["prepare_st"]),
        f"prerequisites count too: {stages_needing_model(['prepare_st'])}",
    )
    for stage in ("rescale", "features_lowres", "predict_lowres", "predict_highres", "train_lowres"):
        require(requires_model([stage]), f"{stage} reads the checkpoint and must require one")

    project = build_project(tmp)
    (project / "raw_data").mkdir(parents=True, exist_ok=True)
    sheet = project / "train.csv"
    sheet.write_text("Sample_name,Image_name\nS1,S1.svs\n", encoding="utf-8")
    no_checkpoint = tmp / "no_checkpoints"
    no_checkpoint.mkdir(exist_ok=True)

    def run(stage: str) -> subprocess.CompletedProcess:
        environment = dict(os.environ)
        # Hide the repo's trained_model/ so nothing can be auto-discovered.
        environment["AURORA_REPO_ROOT"] = str(no_checkpoint)
        return subprocess.run(
            [sys.executable, "-B", "-m", "aurora", "run", "--mode", "training",
             "--run", "nocheck", "--project", str(project), "--samples", str(sheet),
             "--stage", stage, "--dry-run", "--quiet"],
            cwd=str(REPO_ROOT), capture_output=True, text=True, env=environment,
        )

    allowed = run("preprocess")
    require(allowed.returncode == 0, (allowed.stdout + allowed.stderr)[-1200:])
    require(
        "No checkpoint needed" in (allowed.stdout + allowed.stderr),
        "it should say the checkpoint is not needed",
    )
    refused = run("prepare_st")
    require(refused.returncode != 0, "prepare_st must still refuse without a checkpoint")
    return "preprocess runs without --model; prepare_st still refuses"


@check("a resume command need not repeat --samples", "R5")
def test_resume_without_samples(tmp: Path) -> str:
    """`--stage plot` on an existing run must not demand --samples again.

    run.sh and the README both show the plot stage resumed by run name only; the
    earlier behaviour failed with "--samples is required", which made a
    documented command unrunnable.  Parsing was fine, so only running it catches
    this.
    """
    import pickle

    import numpy as np  # noqa: PLC0415

    project = build_project(tmp)
    from aurora.paths import RunLayout

    run_root = project / "aurora_runs" / "resume"
    layout = RunLayout.from_root(run_root, project)
    layout.ensure()

    # A run that has already recorded its sheet and produced predictions.
    sample = "TCGA-TEST-01"
    (layout.config_dir / "samples.csv").write_text(
        "Image_name,Sample_name\nTCGA-TEST-01.svs,TCGA-TEST-01\n", encoding="utf-8"
    )
    spots = 3
    payload = {
        "exp_pred": {224: np.zeros((spots, 2), dtype=np.float32)},
        "log_prop_pred": {224: np.zeros((spots, 2), dtype=np.float32)},
        "pos": {224: np.array([[0, 0], [1, 0], [0, 1]])},
        "patches_loc_before_pad": {224: np.array([[0, 0], [1, 0], [0, 1]])},
        "genes": ["IL7R", "KRT8"],
        "cell_types": ["Malignant", "T_cell"],
        "resolutions": [224],
    }
    pred_dir = layout.prediction_output_dir("lowres")
    pred_dir.mkdir(parents=True, exist_ok=True)
    with open(pred_dir / f"{sample}-pred.pickle", "wb") as handle:
        pickle.dump(payload, handle)

    model = find_combined_model()
    require(model is not None, "no combined checkpoint to test with")
    common_args = [
        sys.executable, "-B", "-m", "aurora", "run",
        "--project", str(project), "--run", "resume",
        "--model", str(model),
        "--stage", "plot", "--gene", "IL7R",
    ]

    # No --samples: the run's recorded sheet is reused.
    result = subprocess.run(
        common_args + ["--dry-run", "--quiet"],
        cwd=str(REPO_ROOT), capture_output=True, text=True,
    )
    output = result.stdout + result.stderr
    require(result.returncode == 0, output[-1500:])
    require(
        "recorded by the run" in output,
        f"the reuse should be reported, got: {output[-500:]}",
    )
    require("--samples is required" not in output, "it must not demand --samples")

    # A run that has recorded nothing must still fail, with a useful hint.
    fresh = build_project(tmp / "fresh")
    result = subprocess.run(
        [sys.executable, "-B", "-m", "aurora", "run",
         "--project", str(fresh), "--run", "never_started",
         "--model", str(model), "--stage", "plot", "--gene", "IL7R",
         "--dry-run", "--quiet"],
        cwd=str(REPO_ROOT), capture_output=True, text=True,
    )
    output = result.stdout + result.stderr
    require(result.returncode != 0, "a run with no recorded sheet must still fail")
    require("--samples is required" in output, output[-800:])
    require("Inference_samples.csv" in output or "samples.csv" in output, output[-800:])
    return "the recorded sheet is reused; an unknown run still errors clearly"


@check("the new example script covers inference and training", "R5")
def test_new_example() -> str:
    candidates = [REPO_ROOT / "run.sh", REPO_ROOT / "command_new.sh"]
    found = [p for p in candidates if p.is_file()]
    if not found:
        return SKIP
    text = "\n".join(p.read_text(encoding="utf-8") for p in found)
    require("python -m aurora run" in text or "-m aurora run" in text, "no 'aurora run' example")
    require("--mode training" in text, "the example does not show a training run")
    require("predict_highres" in text or "--all" in text, "the example does not show the prediction chain")
    require("merge_aurora_checkpoints" in text, "the example does not show how to build the combined checkpoint")
    return f"{', '.join(p.name for p in found)}"


# --------------------------------------------------------------------------- #
# Requirement 6 -- README structure
# --------------------------------------------------------------------------- #


@check("README has an inference-only part and a training part", "R6")
def test_readme() -> str:
    path = REPO_ROOT / "README.md"
    require(path.is_file(), "README.md is missing")
    text = path.read_text(encoding="utf-8")
    # Match the parts by what they are for, not by an exact title: the headings have
    # been reworded more than once ("Inference only", "Inference from existing
    # checkpoints").
    parts = [line for line in text.splitlines() if line.startswith("# Part")]
    require(
        any("nference" in line for line in parts),
        f"no inference part among {parts}",
    )
    require(
        any("raining" in line for line in parts),
        f"no training part among {parts}",
    )
    require("-m aurora run" in text, "the README does not show the new CLI")
    # The old, long-form instructions must still be available.
    legacy = REPO_ROOT / "docs" / "legacy" / "README_original.md"
    require(legacy.is_file(), f"the original README was not preserved at {legacy}")
    return "inference-only + training sections, original preserved"


# --------------------------------------------------------------------------- #
# Nothing was deleted or broken
# --------------------------------------------------------------------------- #


@check("every original script is still present and parses", "keep-old")
def test_legacy_intact() -> str:
    required = [
        "code/create_patches_fp.py",
        "code/rescale.py",
        "code/extract_features.py",
        "code/extract_features_highres.py",
        "code/segment_image.py",
        "code/predict.py",
        "code/predict_highres.py",
        "code/train.py",
        "code/train_highres.py",
        "code/prepare_highres_training.py",
        "code/prepare_st.py",
        "code/smooth_cnt.py",
        "code/finetune.py",
        "code/test.py",
        "code/test_highres.py",
        "code/plot.py",
        "code/model.py",
        "code/train_utils.py",
        "code/datasets.py",
        "code/image_utils.py",
        "code/utils.py",
        "code/run_istar.py",
        "code/run_HistoSweep.py",
        "code/run_chunked_tcga.py",
        "command.sh",
        "README.md",
    ]
    missing = [name for name in required if not (REPO_ROOT / name).is_file()]
    require(not missing, f"missing files: {missing}")
    broken = []
    for name in required:
        if not name.endswith(".py"):
            continue
        path = REPO_ROOT / name
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            broken.append(f"{name}: {exc}")
    require(not broken, f"files that no longer parse: {broken}")
    return f"{len(required)} files present, {sum(1 for n in required if n.endswith('.py'))} parse"


@check("the legacy scripts no longer import training packages eagerly", "keep-old")
def test_legacy_imports_deferred() -> str:
    import tools.audit_inference_deps as audit  # type: ignore

    index = audit.build_index()
    expectations = {
        "code/train_utils.py": "sklearn",
        "code/model.py": "torch_geometric",
        "code/rescale.py": "scanpy",
        "code/extract_features.py": "sklearn",
        "code/extract_features_highres.py": "sklearn",
    }
    for relative, module in expectations.items():
        info = audit.analyse_file(REPO_ROOT / relative)
        require(
            module not in info.eager,
            f"{relative} still imports {module} eagerly (it must be deferred)",
        )
    return f"{len(expectations)} modules defer their training-only imports"


@check("old and new code coexist on sys.path", "keep-old")
def test_coexistence() -> str:
    sys.path.insert(0, str(REPO_ROOT))
    aurora = importlib.import_module("aurora")
    require(aurora.__version__, "aurora has no version")
    from aurora._legacy import code_dir, legacy_script

    directory = code_dir()
    require(directory.is_dir(), f"{directory} is not a directory")
    for script in ("predict.py", "predict_highres.py", "train.py", "train_highres.py"):
        require(legacy_script(script).is_file(), f"{script} not found")
    return f"aurora {aurora.__version__} + {directory.name}/"


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def _console_safe(text: str) -> str:
    """``text`` with anything the current stdout encoding cannot encode replaced."""
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        text.encode(encoding)
        return text
    except (UnicodeEncodeError, LookupError):
        return text.encode(encoding, "replace").decode(encoding, "replace")


def main(argv: Optional[List[str]] = None) -> int:
    global _verbose
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-v", "--verbose", action="store_true", help="Print tracebacks for failures")
    parser.add_argument("--keep", action="store_true", help="Keep the temporary project directory")
    parser.add_argument(
        "--tmp",
        type=str,
        default=None,
        help=(
            "Parent directory for the scratch project. Defaults to the repository root, "
            "because a sandboxed session may not be allowed to write to the system temp folder."
        ),
    )
    args = parser.parse_args(argv)
    _verbose = args.verbose

    parent = Path(args.tmp).resolve() if args.tmp else REPO_ROOT
    parent.mkdir(parents=True, exist_ok=True)
    # NOTE: tempfile.mkdtemp() creates the directory with mode 0700, which some
    # sandboxed Windows sessions then refuse to write into.  A plain mkdir with
    # the default mode behaves everywhere.
    temp_dir = parent / f".aurora_selftest_{os.getpid()}"
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir()
    try:
        _run_registered(temp_dir)
    finally:
        if not args.keep:
            shutil.rmtree(temp_dir, ignore_errors=True)
        else:
            print(f"temporary project kept at {temp_dir}")

    width = max(len(name) for name, _, _ in _results)
    failures = 0
    print()
    print("AURORA code-system self-test")
    print("=" * (width + 40))
    for name, status, detail in _results:
        if status == FAIL:
            failures += 1
        if status == SKIP and not args.verbose:
            continue
        line = f"  [{status}] {name:<{width}}"
        if detail and (status != PASS or args.verbose):
            line += f"  {detail}"
        # This runs on consoles that are not UTF-8 (a GBK terminal on Windows, for
        # one) and a detail line may quote the README, so never let the report itself
        # raise UnicodeEncodeError.
        print(_console_safe(line))
    print()
    total = len(_results)
    skipped = sum(1 for _, status, _ in _results if status == SKIP)
    print(f"{total - failures - skipped} passed, {failures} failed, {skipped} skipped")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
