#!/usr/bin/env python
"""
Pure-stdlib (no torch) recon of torch.save archives.

Strategy
--------
torch.save with the "new zipfile serialization" layout writes a ZIP archive
containing (at least):
    <prefix>/data.pkl      -> the pickle stream for the object graph
    <prefix>/version       -> ASCII "3" (or similar) format version
    <prefix>/byteorder     -> "little" (only when the pickle is byte-swapped)
    <prefix>/data/<n>      -> raw storage bytes, one zip entry per storage

We do two independent passes over data.pkl:

  PASS A - opcode census with pickletools.genops: enumerate every GLOBAL /
           STACK_GLOBAL (module, name) pair and every opcode, so we can see
           which torch rebuild helpers are referenced. This needs no
           class resolution at all.

  PASS B - real object-graph reconstruction with a pickle.Unpickler
           subclass. find_class() returns:
             * the genuine class for a small safelist (collections.OrderedDict,
               builtins.*) so the top level and nested dicts come back as real
               dicts with real string keys, and torch.Size -> tuple;
             * a dynamically generated *stub class* for everything else
               (torch._utils._rebuild_tensor_v2, torch.float32,
               torch.storage.*, numpy dtypes, ...). The stub records its
               qualified name plus the positional/keyword arguments it was
               called with, and tolerates __setstate__ (BUILD opcode) and
               __new__ (NEWOBJ / NEWOBJ_EX opcodes).
           persistent_load() receives the out-of-band storage descriptors
           (torch emits ('storage', StorageClass, key, location, numel)) and
           returns a StorageRef stub. Because storages are referenced by
           persistent id, torch never pickles the storage bytes inline.

Afterwards we walk the reconstructed graph and interpret the stub call
signatures:
    _rebuild_tensor_v2(storage, storage_offset, size, stride,
                       requires_grad, backward_hooks)
    _rebuild_tensor(storage, storage_offset, size, stride)      # v1
    _rebuild_parameter(data, requires_grad, backward_hooks)
    _rebuild_parameter_with_state(data, requires_grad, backward_hooks, state)
    _rebuild_tensor_v3 / _rebuild_wrapper_subclass / _rebuild_from_type_v2
to recover (type, shape, dtype, numel, storage bytes) per key.
"""

import io
import json
import os
import pickle
import pickletools
import sys
import zipfile
from collections import OrderedDict

# --------------------------------------------------------------------------
# dtype -> itemsize (bytes per element), stdlib-only
# --------------------------------------------------------------------------
DTYPE_ITEMSIZE = {
    "torch.float64": 8, "torch.double": 8,
    "torch.float32": 4, "torch.float": 4,
    "torch.float16": 2, "torch.half": 2,
    "torch.bfloat16": 2,
    "torch.complex128": 16, "torch.cdouble": 16,
    "torch.complex64": 8, "torch.cfloat": 8,
    "torch.int64": 8, "torch.long": 8,
    "torch.int32": 4, "torch.int": 4,
    "torch.int16": 2, "torch.short": 2,
    "torch.int8": 1, "torch.uint8": 1, "torch.bool": 1,
}
STORAGE_DTYPE = {
    "FloatStorage": "torch.float32", "DoubleStorage": "torch.float64",
    "HalfStorage": "torch.float16", "BFloat16Storage": "torch.bfloat16",
    "LongStorage": "torch.int64", "IntStorage": "torch.int32",
    "ShortStorage": "torch.int16", "CharStorage": "torch.int8",
    "ByteStorage": "torch.uint8", "BoolStorage": "torch.bool",
    "ComplexFloatStorage": "torch.complex64",
    "ComplexDoubleStorage": "torch.complex128",
    "UntypedStorage": "torch.uint8",
}


# --------------------------------------------------------------------------
# Stub machinery
# --------------------------------------------------------------------------
class Stub:
    """Generic stand-in for any class we refuse to actually import."""

    _qualname = "<unknown>"

    def __new__(cls, *args, **kwargs):
        obj = object.__new__(cls)
        obj.qualname = cls._qualname
        obj.args = args
        obj.kwargs = kwargs
        obj.state = None
        return obj

    def __init__(self, *args, **kwargs):
        # object.__new__ already captured them; keep a no-op so that a
        # subsequent default __init__ call with args does not explode.
        self.args = args
        self.kwargs = kwargs

    def __setstate__(self, state):
        self.state = state

    def __repr__(self):
        return "<Stub %s args=%d>" % (self.qualname, len(self.args))


_stub_cache = {}


def stub_class(qualname):
    if qualname not in _stub_cache:
        _stub_cache[qualname] = type(
            "Stub_" + qualname.replace(".", "_"), (Stub,), {"_qualname": qualname}
        )
    return _stub_cache[qualname]


class StorageRef(Stub):
    """Result of Unpickler.persistent_load for a torch storage descriptor."""

    _qualname = "torch.Storage"

    def __init__(self, pid):
        self.pid = pid
        # torch emits ('storage', StorageClass, key, location, numel)
        self.kind = pid[0] if len(pid) > 0 else None
        self.storage_class = pid[1] if len(pid) > 1 else None
        self.key = pid[2] if len(pid) > 2 else None
        self.location = pid[3] if len(pid) > 3 else None
        self.numel = pid[4] if len(pid) > 4 else None
        self.qualname = "torch.storage.%s" % (
            getattr(self.storage_class, "__name__", self.storage_class)
            if not isinstance(self.storage_class, str) else self.storage_class
        )


SAFE_REAL = {
    ("collections", "OrderedDict"): OrderedDict,
    ("collections", "defaultdict"): dict,
    ("builtins", "dict"): dict,
    ("builtins", "list"): list,
    ("builtins", "tuple"): tuple,
    ("builtins", "set"): set,
    ("builtins", "frozenset"): frozenset,
    ("builtins", "int"): int,
    ("builtins", "float"): float,
    ("builtins", "complex"): complex,
    ("builtins", "str"): str,
    ("builtins", "bytes"): bytes,
    ("builtins", "bytearray"): bytearray,
    ("builtins", "bool"): bool,
    ("builtins", "object"): object,
    ("builtins", "slice"): slice,
    ("torch", "Size"): tuple,          # torch.Size subclasses tuple
    ("torch", "device"): str,          # only ever used as an argument
}


class ReconUnpickler(pickle.Unpickler):
    def __init__(self, file, module_census, global_census):
        super().__init__(file, fix_imports=False, encoding="latin1")
        self.module_census = module_census
        self.global_census = global_census
        self.persistent_ids = []

    def find_class(self, module, name):
        key = (module, name)
        self.global_census["%s.%s" % (module, name)] = (
            self.global_census.get("%s.%s" % (module, name), 0) + 1
        )
        self.module_census[module] = self.module_census.get(module, 0) + 1
        if key in SAFE_REAL:
            return SAFE_REAL[key]
        return stub_class("%s.%s" % (module, name))

    def persistent_load(self, pid):
        self.persistent_ids.append(pid)
        return StorageRef(pid)


# --------------------------------------------------------------------------
# Interpretation of the reconstructed graph
# --------------------------------------------------------------------------
def _dtype_from(value):
    """Extract a dtype string from a stub / string / dict argument."""
    if isinstance(value, Stub):
        q = value.qualname
        if q.startswith("torch.") and q.split(".")[-1] in (
            "float64", "double", "float32", "float", "float16", "half",
            "bfloat16", "complex128", "cdouble", "complex64", "cfloat",
            "int64", "long", "int32", "int", "int16", "short", "int8",
            "uint8", "bool",
        ):
            return q
        if q == "numpy.dtype":
            a = value.args[0] if value.args else None
            return "numpy.%s" % a
        return q
    if isinstance(value, str):
        return value
    return None


def _shape_of(value):
    if isinstance(value, (tuple, list)):
        try:
            return [int(x) for x in value]
        except (TypeError, ValueError):
            return None
    if isinstance(value, Stub) and value.qualname == "torch.Size":
        return _shape_of(value.args[0] if value.args else None)
    return None


def describe_node(node):
    """Turn one reconstructed value into a metadata record."""
    rec = {"kind": None, "shape": None, "dtype": None, "storage": None,
           "requires_grad": None, "stride": None, "storage_offset": None,
           "children": None}

    if isinstance(node, StorageRef):
        rec["kind"] = "storage_ref"
        scname = storage_class_name(node.storage_class)
        rec["dtype"] = STORAGE_DTYPE.get(scname, "torch.%s" % scname)
        rec["storage_key"] = node.key
        rec["numel"] = node.numel
        return rec

    if isinstance(node, Stub):
        q = node.qualname
        a = list(node.args)
        rec["kind"] = q
        if q in ("torch._utils._rebuild_tensor_v2", "torch._utils._rebuild_tensor_v3"):
            # (storage, storage_offset, size, stride, requires_grad, hooks[, ...])
            if len(a) >= 1:
                rec["storage"] = a[0]
            if len(a) >= 2:
                rec["storage_offset"] = a[1]
            if len(a) >= 3:
                rec["shape"] = _shape_of(a[2])
            if len(a) >= 4:
                rec["stride"] = _shape_of(a[3])
            if len(a) >= 5:
                rec["requires_grad"] = a[4]
        elif q == "torch._utils._rebuild_tensor":
            if len(a) >= 1:
                rec["storage"] = a[0]
            if len(a) >= 2:
                rec["storage_offset"] = a[1]
            if len(a) >= 3:
                rec["shape"] = _shape_of(a[2])
            if len(a) >= 4:
                rec["stride"] = _shape_of(a[3])
        elif q in ("torch._utils._rebuild_parameter",
                   "torch._utils._rebuild_parameter_with_state"):
            inner = describe_node(a[0]) if a else {}
            rec.update({k: v for k, v in inner.items() if k != "kind"})
            rec["kind"] = "parameter(%s)" % inner.get("kind")
            if len(a) >= 2:
                rec["requires_grad"] = a[1]
        elif q == "torch._utils._rebuild_from_type_v2":
            # (func, new_type, args, state)
            if len(a) >= 3 and isinstance(a[2], (tuple, list)):
                inner = describe_node(a[2][0]) if a[2] else {}
                rec.update({k: v for k, v in inner.items() if k != "kind"})
            rec["kind"] = "from_type_v2(%s)" % rec.get("kind")
        elif q == "torch._utils._rebuild_wrapper_subclass":
            if len(a) >= 1:
                rec["dtype"] = _dtype_from(a[0])
                rec["shape"] = _shape_of(a[1]) if len(a) > 1 else None
        elif q == "numpy.core.multiarray._reconstruct" or q == "numpy._core.multiarray._reconstruct":
            rec["kind"] = "numpy_array"
            for x in a:
                if isinstance(x, tuple) and x and x[0] is not None:
                    pass
            rec["shape"] = _shape_of(a[1]) if len(a) > 1 else None
            rec["dtype"] = _dtype_from(a[2]) if len(a) > 2 else None
        elif q == "numpy.ndarray":
            rec["kind"] = "numpy_ndarray"
            st = node.state
            if isinstance(st, tuple) and len(st) >= 3:
                rec["shape"] = _shape_of(st[0])
                rec["dtype"] = _dtype_from(st[1])
        elif q == "numpy.dtype":
            rec["kind"] = "numpy_dtype"
            rec["dtype"] = _dtype_from(node)
        else:
            # unknown torch rebuild: try to salvage a size tuple
            for x in a:
                s = _shape_of(x)
                if s is not None:
                    rec["shape"] = s
                    break
        if rec["shape"] is not None and rec["dtype"] is None and rec["storage"] is not None:
            rec["dtype"] = describe_node(rec["storage"]).get("dtype")
        if rec["shape"] is not None:
            n = 1
            for d in rec["shape"]:
                n *= d
            rec["numel"] = n
        return rec

    if isinstance(node, (OrderedDict, dict)):
        rec["kind"] = "dict"
        rec["children"] = list(node.keys())
        return rec
    if isinstance(node, (list, tuple, set, frozenset)):
        rec["kind"] = type(node).__name__
        rec["children"] = len(node)
        return rec
    rec["kind"] = type(node).__name__
    return rec


def storage_class_name(sc):
    """torch.FloatStorage shows up as a stub class -> read its qualname."""
    if isinstance(sc, type) and issubclass(sc, Stub):
        return sc._qualname.rsplit(".", 1)[-1]
    if isinstance(sc, Stub):
        return sc.qualname.rsplit(".", 1)[-1]
    if isinstance(sc, str):
        return sc
    return getattr(sc, "__name__", str(sc))


def itemsize(dtype):
    if dtype in DTYPE_ITEMSIZE:
        return DTYPE_ITEMSIZE[dtype]
    return None


# --------------------------------------------------------------------------
# Main per-file analysis
# --------------------------------------------------------------------------
def opcode_census(data):
    globals_seen = {}
    opcodes = {}
    strings = set()
    n_persid = 0
    try:
        for op, arg, pos in pickletools.genops(io.BytesIO(data)):
            opcodes[op.name] = opcodes.get(op.name, 0) + 1
            if op.name in ("GLOBAL", "STACK_GLOBAL") and arg:
                globals_seen[arg] = globals_seen.get(arg, 0) + 1
            elif op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE",
                             "BINUNICODE8") and isinstance(arg, str):
                strings.add(arg)
            elif op.name == "BINPERSID":
                n_persid += 1
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc), "opcodes": opcodes,
                "globals": globals_seen, "n_persid": n_persid, "strings": strings}
    return {"error": None, "opcodes": opcodes, "globals": globals_seen,
            "n_persid": n_persid, "strings": strings}


def analyze(path):
    out = {"path": path, "entries": [], "root": None, "root_type": None,
           "keys": [], "records": [], "opcode": None, "persistent": [],
           "notes": [], "size_on_disk": os.path.getsize(path),
           "zip_total_uncompressed": 0}

    with zipfile.ZipFile(path, "r") as zf:
        infos = zf.infolist()
        for i in infos:
            out["entries"].append(
                {"name": i.filename, "size": i.file_size,
                 "compressed": i.compress_size, "method": i.compress_type}
            )
            out["zip_total_uncompressed"] += i.file_size

        names = {i.filename for i in infos}
        # locate the format-version / byteorder sidecar entries
        for i in infos:
            base = i.filename.rsplit("/", 1)[-1]
            if base in ("version", "byteorder"):
                raw = zf.read(i.filename)
                out["notes"].append("%s = %r" % (i.filename, raw.decode("utf-8", "replace")))
        if not any(i.filename.endswith("version") for i in infos):
            out["notes"].append("NO 'version' entry present")

        pkl_name = None
        for i in infos:
            if i.filename.endswith("data.pkl") or i.filename.endswith(".pkl"):
                pkl_name = i.filename
                break
        if pkl_name is None:
            out["notes"].append("NO data.pkl entry found")
            return out
        out["data_pkl_entry"] = pkl_name
        data = zf.read(pkl_name)

    out["data_pkl_bytes"] = len(data)
    cen = opcode_census(data)
    out["opcode"] = {
        "error": cen["error"],
        "opcodes": cen["opcodes"],
        "globals": cen["globals"],
        "n_persid": cen["n_persid"],
        "n_distinct_strings": len(cen["strings"]),
        "strings_sample": sorted(cen["strings"])[:400],
    }

    module_census, global_census = {}, {}
    up = ReconUnpickler(io.BytesIO(data), module_census, global_census)
    try:
        root = up.load()
    except Exception as exc:  # noqa: BLE001
        out["notes"].append("Unpickler failed: %r" % (exc,))
        out["modules"] = module_census
        out["globals"] = global_census
        return out
    out["modules"] = module_census
    out["globals"] = global_census
    out["root_type"] = type(root).__name__ + (
        " (stub %s)" % root.qualname if isinstance(root, Stub) else ""
    )

    for pid in up.persistent_ids:
        sc = pid[1] if len(pid) > 1 else None
        scname = storage_class_name(sc)
        dt = STORAGE_DTYPE.get(scname, "torch.%s" % scname)
        nel = pid[4] if len(pid) > 4 else None
        out["persistent"].append({
            "tuple": [storage_class_name(x) if isinstance(x, Stub) else str(x) for x in pid],
            "storage_class": scname, "key": str(pid[2]) if len(pid) > 2 else None,
            "location": str(pid[3]) if len(pid) > 3 else None,
            "numel": nel, "dtype": dt,
            "bytes": (nel * itemsize(dt)) if (nel is not None and itemsize(dt)) else None,
        })

    # Decide what the "state_dict" level is
    target = root
    out["root_is_dict"] = isinstance(root, (dict, OrderedDict))
    if isinstance(root, (dict, OrderedDict)):
        out["root_keys"] = [str(k) for k in root.keys()]
        # heuristic: if there is exactly one dict-valued entry with many tensor
        # values, treat that as the state dict
        dict_children = [(k, v) for k, v in root.items() if isinstance(v, (dict, OrderedDict))]
        if len(root) == 1 and len(dict_children) == 1:
            target = dict_children[0][1]
            out["state_dict_container_key"] = str(dict_children[0][0])
        else:
            tensorish = sum(1 for v in root.values()
                            if isinstance(v, Stub) or isinstance(v, StorageRef))
            if tensorish > 0 and not dict_children:
                target = root
    else:
        out["root_keys"] = []

    if isinstance(target, StorageRef):
        out["keys"] = ["<bare storage>"]
        out["records"] = [{"key": "<bare storage>", **describe_node(target)}]
        return out
    if isinstance(target, Stub):
        out["keys"] = []
        out["root_stub_args"] = len(target.args)
        return out
    if not isinstance(target, (dict, OrderedDict)):
        out["records"] = [{"key": "<root>", **describe_node(target)}]
        return out

    for k in target.keys():
        v = target[k]
        rec = describe_node(v)
        rec["key"] = str(k)
        out["keys"].append(str(k))
        out["records"].append(rec)

    # storage accounting (storages are shared by persistent id -> dedupe)
    seen = {}
    keymap = {}
    for r in out["records"]:
        st = r.get("storage")
        if isinstance(st, StorageRef):
            seen[st.key] = st
            keymap.setdefault(str(st.key), []).append(r["key"])
            r["dtype"] = STORAGE_DTYPE.get(storage_class_name(st.storage_class),
                                           r.get("dtype"))
    total = 0
    for key, st in seen.items():
        dt = STORAGE_DTYPE.get(storage_class_name(st.storage_class),
                               "torch.%s" % storage_class_name(st.storage_class))
        sz = itemsize(dt)
        if st.numel is not None and sz:
            total += st.numel * sz
    for r in out["records"]:
        if r.get("numel") is not None and itemsize(r.get("dtype")):
            r["bytes"] = r["numel"] * itemsize(r["dtype"])
    out["unique_storages"] = len(seen)
    out["storage_key_map"] = keymap
    out["total_storage_bytes"] = total
    return out


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    targets = [
        os.path.join(root, "trained_model", "AURORA_LUAD_V2.pth"),
        os.path.join(root, "trained_model", "AURORA_highres_model_weights.pth"),
    ]
    results = {}
    for t in targets:
        print("=" * 78)
        print("FILE:", t)
        print("=" * 78)
        r = analyze(t)
        results[os.path.basename(t)] = r

        print("\n-- ZIP ENTRIES --")
        for e in r["entries"][:20]:
            print("  %-40s size=%-12d comp=%d" % (e["name"], e["size"], e["compressed"]))
        if len(r["entries"]) > 20:
            print("  ... (%d entries total)" % len(r["entries"]))
        print("\n-- NOTES --")
        for n in r["notes"]:
            print("  ", n)
        print("\n-- ROOT --")
        print("  root_type:", r.get("root_type"))
        print("  root_keys:", r.get("root_keys"))
        print("  state_dict_container_key:", r.get("state_dict_container_key"))
        print("  n keys:", len(r.get("keys", [])))
        print("  unique_storages:", r.get("unique_storages"),
              "total_storage_bytes:", r.get("total_storage_bytes"))
        print("\n-- KEY TABLE --")
        for rec in r.get("records", []):
            print("  %-52s %-26s %-16s %-10s %s" % (
                rec.get("key"), str(rec.get("kind")),
                str(rec.get("shape")), str(rec.get("numel")),
                str(rec.get("dtype"))))
        print("\n-- GLOBALS (pickletools) --")
        g = r.get("opcode", {}).get("globals", {})
        for k in sorted(g):
            print("  %-6d %s" % (g[k], k))
        print("\n-- persistent ids (first 3) --")
        for p in r.get("persistent", [])[:3]:
            print("  ", p)
        print()

    outp = os.path.join(here, "recon.json")
    with open(outp, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1, default=str)
    print("wrote", outp)

    a = results["AURORA_LUAD_V2.pth"]
    b = results["AURORA_highres_model_weights.pth"]
    sa, sb = set(a.get("keys", [])), set(b.get("keys", []))
    print("\n=== OVERLAP ===")
    print("low keys:", len(sa), " high keys:", len(sb))
    print("intersection:", len(sa & sb))
    print("low only:", len(sa - sb), " high only:", len(sb - sa))


if __name__ == "__main__":
    main()
