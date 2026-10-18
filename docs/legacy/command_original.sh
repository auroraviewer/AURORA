# Training
export CODE_PATH=./AURORA/code/
export PROJECT_PATH=./
python -u $CODE_PATH/rescale.py -p $PROJECT_PATH -d hest_data -m Training_samples.csv -o debug --plot
python -u $CODE_PATH/extract_features.py -p $PROJECT_PATH -d hest_data -m Training_samples.csv -o debug -b 128 --plot --UNI_local_path ./uni2h
python -u $CODE_PATH/prepare_st.py -p $PROJECT_PATH -d hest_data -m Training_samples.csv -o debug
python -u $CODE_PATH/smooth_cnt.py -p $PROJECT_PATH -d hest_data -m Breast/all_samples.csv -o Breast
python -u $CODE_PATH/train.py -p $PROJECT_PATH -d hest_data -i Breast -t Breast -j Breast/model_parameter_BRCA.json --train_samples Breast/train_samples.csv --train_sample_col 0 --log --smooth_cnts
python -u $CODE_PATH/extract_features_highres.py -p $PROJECT_PATH -c $CODE_PATH -m Training_samples.csv -o debug --plot --UNI_local_path ./uni2h --CATCH_config_templet_path $CODE_PATH/CATCH_config_templet.yaml --catch_path $CODE_PATH/CATCH-main
python -u $CODE_PATH/prepare_highres_training.py -p $PROJECT_PATH -t debug -d hest_data -j model_parameter_BRCA.json --model_pth AURORA_BRCA.pth --sample_sheet debug/Training_samples.csv
python -u $CODE_PATH/train_highres.py -p $PROJECT_PATH -t debug -d hest_data -i debug -j model_parameter_BRCA_highres.json --train_samples Training_samples.csv --log

# Test (only for developer)
python -u $CODE_PATH/test.py -p $PROJECT_PATH -d hest_data -i Breast -t Breast -j Breast/model_parameter_BRCA.json --model_pth AURORA_model_weights_smooth.pth --test_samples Breast/test_samples.csv --output_dir test_smooth_pred --smooth_cnts
python -u $CODE_PATH/test_highres.py -p $PROJECT_PATH -t debug -d hest_data -i debug -j model_parameter_BRCA_highres.json --test_samples Test_samples.csv --model_pth AURORA_highres_model_weights.pth --plot

## Apply existing model to new samples (step wise)
export CODE_PATH=../AURORA/code/
export PROJECT_PATH=./

python -u $CODE_PATH/segment_image.py -p $PROJECT_PATH -d wsis -s wsis_segmentation --code_path $CODE_PATH --preset lung.csv # For lung use lung.csv
python -u $CODE_PATH/rescale.py --inference -p $PROJECT_PATH -m Inference_samples.csv -o AURORA_interim --image_path wsis --seg_path wsis_segmentation --plot
python -u $CODE_PATH/extract_features.py --inference -p $PROJECT_PATH -m Inference_samples.csv -o AURORA_interim --image_path wsis --seg_path wsis_segmentation -b 128 --plot --UNI_local_path ../uni2h

python -u $CODE_PATH/extract_features_highres.py -p $PROJECT_PATH -c $CODE_PATH -m Inference_samples.csv --sample_name_col 1 -o AURORA_interim --plot --UNI_local_path ../uni2h --CATCH_config_templet_path $CODE_PATH/CATCH_config_templet.yaml --catch_path ../

python -u $CODE_PATH/predict.py -p $PROJECT_PATH -i AURORA_interim -t AURORA_pred -j model_parameter_LUAD.json --model_pth AURORA_LUAD.pth --sample_sheet Inference_samples.csv --bulk TCGA_LUAD_expression.csv --save_embedding
python -u $CODE_PATH/predict_highres.py -p $PROJECT_PATH -i AURORA_interim -t AURORA_pred -j model_parameter_LUAD_highres.json --model_pth AURORA_highres_model_weights.pth --sample_sheet Inference_samples.csv --batch_size 100

## Apply existing model to new samples (batch mode)
export CODE_PATH=./AURORA/code/
export PROJECT_PATH=./

python -u $CODE_PATH/segment_image.py -p $PROJECT_PATH/debug_pred_kidney -d wsis -s wsis_segmentation --code_path $CODE_PATH --preset tcga.csv # For lung use lung.csv
python -u $CODE_PATH/rescale.py --inference -p $PROJECT_PATH -m debug_pred_kidney/Inference_samples.csv -o debug_pred_kidney --image_path debug_pred_kidney/wsis --seg_path debug_pred_kidney/wsis_segmentation --plot
python -u $CODE_PATH/run_chunked_tcga.py -p $PROJECT_PATH -c $CODE_PATH \
        -m debug_pred_kidney/Inference_samples.csv --sample_name_col 1 --inference \
        -i debug_pred_kidney --UNI_local_path ./uni2h --batch_size 128 --num_workers 4 --stride 112 \
        --compile-mode off --aggregation-mode auto \
        --strip-temp-dir $PROJECT_PATH/debug_pred_kidney/UNI_multiscale_patches/strip_tmp \
        --smoothing-workers 4 --smoothing-row-chunk 64 --smoothing-channel-chunk 128 \
            --filter-prop 0.5 --patch_sizes 56 14 \
                --histosweep_tmp HistoSweep_tmp --histosweep_path ../HistoSweep/HistoSweep \
                --histosweep_patch_size 14 \
                    -t debug_pred_kidney/AURORA_pred  -j debug_pred_kidney/model_parameter_KIRC_highres.json \
                    --model_pth debug_pred_kidney/AURORA_highres_model_weights.pth \
                    --predict_batch_size 100 \
                        --chunk_size 5 \
                            --uint8_output \
                                --h5_layout spatial_gene_delta \
                                    --h5_compression zstd \
                                        --h5_compression_level 9
