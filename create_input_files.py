from utils import create_input_files

# Steps to prepare data:
# 1) Download COCO datasets according to README file in SgrvinodShowAttendAndTell, and store the folders in './data/'.
# 2) Download Karpathy split file according to README file in SgrvinodShowAttendAdnTell, and only store 'dataset_coco.json' in './data'.
# 3) Create './data/256By256/' './data/64By64', and './data/32By32/' directories.
# 4) Run this python file.

if __name__ == '__main__':
    # Create input files (along with word map)
    # create_input_files(dataset='coco',
    #                    karpathy_json_path='./data/dataset_coco.json',
    #                    image_folder='./data/',
    #                    image_length=256,
    #                    image_width=256,
    #                    captions_per_image=5,
    #                    min_word_freq=5,
    #                    output_folder='./data/256By256/',
    #                    max_len=50)
    
    # create_input_files(dataset='coco',
    #                    karpathy_json_path='./data/dataset_coco.json',
    #                    image_folder='./data/',
    #                    image_length=32,
    #                    image_width=32,
    #                    captions_per_image=5,
    #                    min_word_freq=5,
    #                    output_folder='./data/32By32/',
    #                    max_len=50)
                       
    create_input_files(dataset='coco',
                       karpathy_json_path='./data/dataset_coco.json',
                       image_folder='./data/',
                       image_length=64,
                       image_width=64,
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder='./data/64By64/',
                       max_len=50)
