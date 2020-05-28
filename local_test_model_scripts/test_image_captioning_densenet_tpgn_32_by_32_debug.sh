#!/bin/sh

cd ..

export DATASET_DIR="./data/"

# Folder with data files saved by create_input_files.py
export DATA_FOLDER=${DATASET_DIR}32By32/

python eval.py --experiment_name "image_caption_densenet_tpgn_32_by_32_debug_exp" \
               --dataset_name "coco" \
               --data_name "coco_5_cap_per_img_5_min_word_freq" \
               --run_full_dataset "partial" \
               --best_val_model_idx_for_test 3 \
               --seed 0 \
               --encoder_type "densenet" \
               --decoder_type "tpgn" \
               --use_gpu "False" \
               --gpu_id "None" \
               --beam_size 5
