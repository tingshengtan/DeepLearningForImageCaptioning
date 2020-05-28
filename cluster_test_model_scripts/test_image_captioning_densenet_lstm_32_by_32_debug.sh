#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=Teach-Standard
#SBATCH --gres=gpu:1
#SBATCH --mem=12000  # memory in Mb
#SBATCH --time=0-08:00:00

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

mkdir -p ${DATASET_DIR}32By32/

# Folder with data files saved by create_input_files.py
export DATA_FOLDER=${DATASET_DIR}32By32/

rsync -avu /home/${STUDENT_ID}/EdinburghMlpGroupProject/EdinburghMlpClusterTutorial/data/32By32/TEST_CAPLENS_coco_5_cap_per_img_5_min_word_freq.json ${DATA_FOLDER}
rsync -avu /home/${STUDENT_ID}/EdinburghMlpGroupProject/EdinburghMlpClusterTutorial/data/32By32/TEST_CAPTIONS_coco_5_cap_per_img_5_min_word_freq.json ${DATA_FOLDER}
rsync -avu /home/${STUDENT_ID}/EdinburghMlpGroupProject/EdinburghMlpClusterTutorial/data/32By32/TEST_IMAGES_coco_5_cap_per_img_5_min_word_freq.hdf5 ${DATA_FOLDER}
rsync -avu /home/${STUDENT_ID}/EdinburghMlpGroupProject/EdinburghMlpClusterTutorial/data/32By32/TRAIN_CAPLENS_coco_5_cap_per_img_5_min_word_freq.json ${DATA_FOLDER}
rsync -avu /home/${STUDENT_ID}/EdinburghMlpGroupProject/EdinburghMlpClusterTutorial/data/32By32/TRAIN_CAPTIONS_coco_5_cap_per_img_5_min_word_freq.json ${DATA_FOLDER}
rsync -avu /home/${STUDENT_ID}/EdinburghMlpGroupProject/EdinburghMlpClusterTutorial/data/32By32/TRAIN_IMAGES_coco_5_cap_per_img_5_min_word_freq.hdf5 ${DATA_FOLDER}
rsync -avu /home/${STUDENT_ID}/EdinburghMlpGroupProject/EdinburghMlpClusterTutorial/data/32By32/VAL_CAPLENS_coco_5_cap_per_img_5_min_word_freq.json ${DATA_FOLDER}
rsync -avu /home/${STUDENT_ID}/EdinburghMlpGroupProject/EdinburghMlpClusterTutorial/data/32By32/VAL_CAPTIONS_coco_5_cap_per_img_5_min_word_freq.json ${DATA_FOLDER}
rsync -avu /home/${STUDENT_ID}/EdinburghMlpGroupProject/EdinburghMlpClusterTutorial/data/32By32/VAL_IMAGES_coco_5_cap_per_img_5_min_word_freq.hdf5 ${DATA_FOLDER}
rsync -avu /home/${STUDENT_ID}/EdinburghMlpGroupProject/EdinburghMlpClusterTutorial/data/32By32/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json ${DATA_FOLDER}

# Activate the relevant virtual environment:
source /home/${STUDENT_ID}/miniconda3/bin/activate mlp

cd ..
python eval.py --experiment_name "image_caption_densenet_lstm_32_by_32_debug_exp" \
               --dataset_name "coco" \
               --data_name "coco_5_cap_per_img_5_min_word_freq" \
               --best_val_model_idx_for_test 2 \
               --seed 0 \
               --encoder_type "densenet" \
               --decoder_type "lstm" \
               --use_gpu "True" \
               --gpu_id "0" \
               --beam_size 5
