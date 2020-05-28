#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=Teach-LongJobs
#SBATCH --gres=gpu:1
#SBATCH --mem=12000  # memory in Mb
#SBATCH --time=3-08:00:00

export CUDA_HOME=/opt/cuda-9.0.176.1/

export CUDNN_HOME=/opt/cuDNN-7.0/

export STUDENT_ID=$(whoami)

export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH

export CPATH=${CUDNN_HOME}/include:$CPATH

export PATH=${CUDA_HOME}/bin:${PATH}

export PYTHON_PATH=$PATH

mkdir -p /disk/scratch/${STUDENT_ID}

export TMPDIR=/disk/scratch/${STUDENT_ID}/
export TMP=/disk/scratch/${STUDENT_ID}/

mkdir -p ${TMP}/datasets/

export DATASET_DIR=${TMP}/datasets/

mkdir -p ${DATASET_DIR}64By64/

# Folder with data files saved by create_input_files.py
export DATA_FOLDER=${DATASET_DIR}64By64/

rsync -avu /home/${STUDENT_ID}/EdinburghMlpGroupProject/EdinburghMlpClusterTutorial/data/64By64/TEST_CAPLENS_coco_5_cap_per_img_5_min_word_freq.json ${DATA_FOLDER}
rsync -avu /home/${STUDENT_ID}/EdinburghMlpGroupProject/EdinburghMlpClusterTutorial/data/64By64/TEST_CAPTIONS_coco_5_cap_per_img_5_min_word_freq.json ${DATA_FOLDER}
rsync -avu /home/${STUDENT_ID}/EdinburghMlpGroupProject/EdinburghMlpClusterTutorial/data/64By64/TEST_IMAGES_coco_5_cap_per_img_5_min_word_freq.hdf5 ${DATA_FOLDER}
rsync -avu /home/${STUDENT_ID}/EdinburghMlpGroupProject/EdinburghMlpClusterTutorial/data/64By64/TRAIN_CAPLENS_coco_5_cap_per_img_5_min_word_freq.json ${DATA_FOLDER}
rsync -avu /home/${STUDENT_ID}/EdinburghMlpGroupProject/EdinburghMlpClusterTutorial/data/64By64/TRAIN_CAPTIONS_coco_5_cap_per_img_5_min_word_freq.json ${DATA_FOLDER}
rsync -avu /home/${STUDENT_ID}/EdinburghMlpGroupProject/EdinburghMlpClusterTutorial/data/64By64/TRAIN_IMAGES_coco_5_cap_per_img_5_min_word_freq.hdf5 ${DATA_FOLDER}
rsync -avu /home/${STUDENT_ID}/EdinburghMlpGroupProject/EdinburghMlpClusterTutorial/data/64By64/VAL_CAPLENS_coco_5_cap_per_img_5_min_word_freq.json ${DATA_FOLDER}
rsync -avu /home/${STUDENT_ID}/EdinburghMlpGroupProject/EdinburghMlpClusterTutorial/data/64By64/VAL_CAPTIONS_coco_5_cap_per_img_5_min_word_freq.json ${DATA_FOLDER}
rsync -avu /home/${STUDENT_ID}/EdinburghMlpGroupProject/EdinburghMlpClusterTutorial/data/64By64/VAL_IMAGES_coco_5_cap_per_img_5_min_word_freq.hdf5 ${DATA_FOLDER}
rsync -avu /home/${STUDENT_ID}/EdinburghMlpGroupProject/EdinburghMlpClusterTutorial/data/64By64/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json ${DATA_FOLDER}

# Activate the relevant virtual environment:
source /home/${STUDENT_ID}/miniconda3/bin/activate mlp

cd ..
python train_image_captioning_system.py --dataset_name "coco" \
                                        --data_name "coco_5_cap_per_img_5_min_word_freq" \
                                        --batch_size 28 \
                                        --seed 0 \
                                        --encoder_type "densenet" \
                                        --decoder_type "tpgn" \
                                        --encoder_dim 1024 \
                                        --emb_dim 512 \
                                        --decoder_dim 25 \
                                        --experiment_name "image_caption_densenet_tpgn_64_by_64_exp" \
                                        --num_epochs 50 \
                                        --continue_from_epoch -1 \
                                        --use_gpu "True" \
                                        --gpu_id "0" \
                                        --encoder_lr 1e-4 \
                                        --decoder_lr 4e-4 \
                                        --fine_tune_encoder "True" \
                                        --grad_clip 5. \
                                        --beam_size 1
