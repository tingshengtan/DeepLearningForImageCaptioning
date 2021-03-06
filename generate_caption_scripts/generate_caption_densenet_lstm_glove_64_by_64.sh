#!/bin/sh

cd ..

export DATASET_DIR="./data/"

# Folder with data files saved by create_input_files.py
export DATA_FOLDER=${DATASET_DIR}64By64/

python caption.py --img "./generate_caption_scripts/images/" \
                  --img_length 64 \
                  --img_width 64 \
                  --model "./image_caption_densenet_lstm_glove_64_by_64_exp/saved_models/train_model_15" \
                  --encoder_type "densenet" \
                  --decoder_type "lstm" \
                  --word_map "./data/64By64/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json" \
                  --output_file "./generate_caption_scripts/generate_caption_densenet_lstm_glove_64_by_64.txt" \
                  --beam_size 5
