# Deep Learning for Image Captioning

This is a group project that I had done, together with [@yashvi1109](https://github.com/yashvi1109), when we were taking the 'Machine Learning Practical' course at the University of Edinburgh.

## About

Image captioning is a challenging field of research that involves two domains: computer vision and natural language processing. Many approaches have been proposed over the last few years in order to achieve better performance when generating a description for an image. The aim of this project is to implement an image captioning model based on a combination of DenseNet and [Tensor Product Generation Network (TPGN)](https://arxiv.org/abs/1709.09118) which has never been explored before. With only limited available resources, we have implemented the TPGN language model from scratch. Then, by using the standard MS COCO dataset, we have trained and evaluated ResNet-LSTM, DenseNet-LSTM, ResNet-TPGN, and DenseNet-TPGN models. Specifically, we have compared the new DenseNet-TPGN model against our ResNet-LSTM baseline model, to examine if the new model can perform equally well or even better on the image captioning task. Moreover, we have also validated the performance of DenseNet-based models against ResNet-based model in image captioning. A key finding of this project is that an attention mechanism in image captioning can outperform the sole usage of TPRs in TPGN.

## Directories and Files

* './create_input_files.py' is used to prepare training, validation, and test sets.
* './glove_embeds.py' is used to prepare GloVe embeddings.
* './model_architectures.py' contains the codes of our image captioning models. The TPGN language model that we implemented based on ['Tensor Product Generation Networks for Deep NLP Mmodeling'](https://arxiv.org/abs/1709.09118) can be found here.
* './utils.py' contains all the utility functions.
* './experiment_builder.py' and 'train_image_captioning_system.py' are used to train our models.
* './cluster_experiment_scripts/', './cluster_test_model_scripts/', './local_experiment_scripts/', and './local_test_model_scripts/' contain the scripts that we used to train and evalutate our models.
* './eval.py' is used to compute validation and test scores.
* './image_caption_*_exp/' and './experiment_results/' contain the training and evaluation results of our models.
* './generate_caption_scripts' contain the scripts that we used to generate captions for some example images.
* './report.pdf' is our group project report.

## Code References

Our codes have been built on top of the [Edinburgh Machine Learning Practical course repository](https://github.com/CSTR-Edinburgh/mlpractical/tree/mlp2019-20/mlp_cluster_tutorial) and an [image captioning tutorial](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning). The highlight of this project in terms of codes is our implementation of the TPGN language model.
