import data_providers as data_providers
import numpy as np
from arg_extractor import get_args
from experiment_builder import ExperimentBuilder
from model_architectures import Encoder, DensenetEncoder, DecoderWithAttention, TpgnDecoder
from torchvision import transforms
from utils import *
import torch
import torch.utils.data
import os
import pickle

args, device = get_args()                    # get arguments from command line
rng = np.random.RandomState(seed=args.seed)  # set the seeds for the experiment
torch.manual_seed(seed=args.seed)            # sets pytorch's seed

data_folder = os.environ['DATA_FOLDER']
data_name = args.data_name

# if args.dataset_name == 'emnist':
#     train_data = data_providers.EMNISTDataProvider('train', batch_size=args.batch_size, rng=rng, flatten=False)  # initialize our rngs using the argument set seed
#     val_data = data_providers.EMNISTDataProvider('valid', batch_size=args.batch_size, rng=rng, flatten=False)    # initialize our rngs using the argument set seed
#     test_data = data_providers.EMNISTDataProvider('test', batch_size=args.batch_size, rng=rng, flatten=False)    # initialize our rngs using the argument set seed
#     num_output_classes = train_data.num_classes
# 
# elif args.dataset_name == 'cifar10':
#     transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
#                                           transforms.RandomHorizontalFlip(),
#                                           transforms.ToTensor(),
#                                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
# 
#     transform_test = transforms.Compose([transforms.ToTensor(),
#                                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
# 
#     trainset = data_providers.CIFAR10(root='data', set_name='train', download=True, transform=transform_train)
#     train_data = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)
# 
#     valset = data_providers.CIFAR10(root='data', set_name='val', download=True, transform=transform_test)
#     val_data = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=False, num_workers=2)
# 
#     testset = data_providers.CIFAR10(root='data', set_name='test', download=True, transform=transform_test)
#     test_data = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
# 
#     classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#     num_output_classes = 10
# 
# elif args.dataset_name == 'cifar100':
#     transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
#                                           transforms.RandomHorizontalFlip(),
#                                           transforms.ToTensor(),
#                                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
# 
#     transform_test = transforms.Compose([transforms.ToTensor(),
#                                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
# 
#     trainset = data_providers.CIFAR100(root='data', set_name='train', download=True, transform=transform_train)
#     train_data = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
# 
#     valset = data_providers.CIFAR100(root='data', set_name='val', download=True, transform=transform_test)
#     val_data = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=False, num_workers=2)
# 
#     testset = data_providers.CIFAR100(root='data', set_name='test', download=True, transform=transform_test)
#     test_data = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
# 
#     num_output_classes = 100

if args.dataset_name == 'coco':
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
                                     
    trainset = data_providers.CaptionDataset(data_folder, data_name, 'TRAIN', transform=transforms.Compose([normalize]))
    train_data = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    
    valset = data_providers.CaptionDataset(data_folder, data_name, 'VAL', transform=transforms.Compose([normalize]))
    val_data = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    
    testset = data_providers.CaptionDataset(data_folder, data_name, 'TEST', transform=transforms.Compose([normalize]))
    test_data = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

# Read word map
word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
with open(word_map_file, 'r') as j:
    word_map = json.load(j)

# Read glove embeddings
if args.emb_type == "glove":
    glove_embeddings = pickle.load(open(os.path.join(data_folder, 'glove_words.pkl'), 'rb'))
    glove_embeddings = torch.tensor(glove_embeddings)

# custom_conv_net = ConvolutionalNetwork(  # initialize our network object, in this case a ConvNet
#     input_shape=(args.batch_size, args.image_num_channels, args.image_height, args.image_height),
#     dim_reduction_type=args.dim_reduction_type, num_filters=args.num_filters, num_layers=args.num_layers, use_bias=False,
#     num_output_classes=num_output_classes)

# Initialize encoder and decoder objects
if args.encoder_type == 'resnet' and args.decoder_type == 'lstm':
    print('resnet, lstm')
    encoder_model = Encoder(fine_tune=args.fine_tune_encoder)
    decoder_model = DecoderWithAttention(attention_dim=args.attention_dim,
                                         embed_dim=args.emb_dim,
                                         encoder_dim=args.encoder_dim,
                                         decoder_dim=args.decoder_dim,
                                         vocab_size=len(word_map),
                                         dropout=args.dropout)

elif args.encoder_type == 'densenet' and args.decoder_type == 'lstm':
    print('densenet, lstm')
    encoder_model = DensenetEncoder(fine_tune=args.fine_tune_encoder)
    decoder_model = DecoderWithAttention(attention_dim=args.attention_dim,
                                         embed_dim=args.emb_dim,
                                         encoder_dim=args.encoder_dim,
                                         decoder_dim=args.decoder_dim,
                                         vocab_size=len(word_map),
                                         dropout=args.dropout)
    
elif args.encoder_type == 'resnet' and args.decoder_type == 'tpgn':
    print('resnet, tpgn')
    encoder_model = Encoder(fine_tune=args.fine_tune_encoder)
    decoder_model = TpgnDecoder(embed_dim=args.emb_dim,
                                encoder_dim=args.encoder_dim,
                                decoder_dim=args.decoder_dim,
                                vocab_size=len(word_map))

elif args.encoder_type == 'densenet' and args.decoder_type == 'tpgn':
    print('densenet, tpgn')
    encoder_model = DensenetEncoder(fine_tune=args.fine_tune_encoder)
    decoder_model = TpgnDecoder(embed_dim=args.emb_dim,
                                encoder_dim=args.encoder_dim,
                                decoder_dim=args.decoder_dim,
                                vocab_size=len(word_map))
    
else:
    print('ERROR: Encoder-decoder combination is invalid.')
    exit()

if args.emb_type == "glove":
    decoder_model.load_pretrained_embeddings(glove_embeddings)
decoder_model.fine_tune_embeddings(fine_tune=True)

image_caption_experiment = ExperimentBuilder(encoder_model=encoder_model,
                                             decoder_model=decoder_model,
                                             experiment_name=args.experiment_name,
                                             num_epochs=args.num_epochs,
                                             continue_from_epoch=args.continue_from_epoch,
                                             encoder_lr=args.encoder_lr,
                                             decoder_lr=args.decoder_lr,
                                             fine_tune_encoder=args.fine_tune_encoder,
                                             grad_clip=args.grad_clip,
                                             beam_size=args.beam_size,
                                             device=device,
                                             train_data=train_data,
                                             val_data=val_data,
                                             test_data=test_data,
                                             word_map=word_map,
                                             run_full_dataset=args.run_full_dataset)  # build an experiment object
experiment_metrics, test_metrics = image_caption_experiment.run_experiment()  # run experiment and return experiment metrics
