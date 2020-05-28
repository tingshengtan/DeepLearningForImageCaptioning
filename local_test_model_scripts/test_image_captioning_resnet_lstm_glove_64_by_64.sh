#!/bin/sh

cd ..

export DATASET_DIR="./data/"

# Folder with data files saved by create_input_files.py
export DATA_FOLDER=${DATASET_DIR}64By64/

python eval.py --experiment_name "image_caption_resnet_lstm_glove_64_by_64_exp" \
               --dataset_name "coco" \
               --data_name "coco_5_cap_per_img_5_min_word_freq" \
               --best_val_model_idx_for_test 4 \
               --seed 0 \
               --encoder_type "resnet" \
               --decoder_type "lstm" \
               --use_gpu "False" \
               --gpu_id "None" \
               --beam_size 5
