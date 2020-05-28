#!/bin/sh

cd ..

export DATASET_DIR="./data/"

# Folder with data files saved by create_input_files.py
export DATA_FOLDER=${DATASET_DIR}32By32/

python train_image_captioning_system.py --dataset_name "coco" \
                                        --data_name "coco_5_cap_per_img_5_min_word_freq" \
                                        --batch_size 100 \
                                        --seed 0 \
                                        --encoder_type "resnet" \
                                        --decoder_type "tpgn" \
                                        --encoder_dim 2048 \
                                        --emb_dim 512 \
                                        --decoder_dim 25 \
                                        --experiment_name "image_caption_resnet_tpgn_32_by_32_exp" \
                                        --num_epochs 50 \
                                        --continue_from_epoch -1 \
                                        --use_gpu "False" \
                                        --gpu_id "None" \
                                        --encoder_lr 1e-4 \
                                        --decoder_lr 4e-4 \
                                        --fine_tune_encoder "True" \
                                        --grad_clip 5. \
                                        --beam_size 1
