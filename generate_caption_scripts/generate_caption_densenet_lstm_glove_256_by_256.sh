#!/bin/sh

cd ..

export DATASET_DIR="./data/"

# Folder with data files saved by create_input_files.py
export DATA_FOLDER=${DATASET_DIR}256By256/

python caption.py --img "./generate_caption_scripts/images/" \
                  --img_length 256 \
                  --img_width 256 \
                  --model "./image_caption_densenet_lstm_glove_256_by_256_exp/saved_models/train_model_16" \
                  --encoder_type "densenet" \
                  --decoder_type "lstm" \
                  --word_map "./data/256By256/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json" \
                  --output_file "./generate_caption_scripts/generate_caption_densenet_lstm_glove_256_by_256.txt" \
                  --beam_size 5
