'''
This code is mainly taken from the following github repositories:
1.  parksunwoo/show_attend_and_tell_pytorch
Link: https://github.com/parksunwoo/show_attend_and_tell_pytorch/blob/master/prepro.py
2. sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning
Link: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning
3. This code was written by following the following tutorial:
Link: https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76
This script processes the COCO dataset. Besides, it also processes and generates GloVe embeddings
''' 

'''
Steps to prepare data:
1) Download the glove.6B dataset and place it in glove.6B (https://nlp.stanford.edu/projects/glove/).
2) Run glove_embeds.py.
3) cp data/glove.6B/glove_words.pkl ../32By32
   cp data/glove.6B/glove_words.pkl ../64By64
   cp data/glove.6B/glove_words.pkl ../256By256
   cp data/glove.6B/glove_words.pkl ..
'''

import os
import pickle
from collections import Counter
import nltk
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
import json
from scipy import misc
import bcolz
 
class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

# def build_vocab(json, threshold):
#     coco = COCO(json)
#     counter = Counter()
#     ids = coco.anns.keys()
#     for i, id in enumerate(ids):
#         caption = str(coco.anns[id]['caption'])
#         tokens = nltk.tokenize.word_tokenize(caption.lower())
#         counter.update(tokens)
# 
#     # ommit non-frequent words
#     words = [word for word, cnt in counter.items() if cnt >= threshold]
# 
#     vocab = Vocabulary()
#     vocab.add_word('<pad>') # 0
#     vocab.add_word('<start>') # 1
#     vocab.add_word('<end>') # 2
#     vocab.add_word('<unk>') # 3
# 
#     for i, word in enumerate(words):
#         vocab.add_word(word)
#     return vocab
    
def build_vocab(json_path, threshold):
    with open(json_path, 'r') as j:
        data = json.load(j)
    
    counter = Counter()
    
    for img in data['images']:
        for c in img['sentences']:
            counter.update(c['tokens'])

    # ommit non-frequent words
    words = [word for word, cnt in counter.items() if cnt > threshold]

    vocab = Vocabulary()
    vocab.add_word('<pad>') # 0
    vocab.add_word('<start>') # 1
    vocab.add_word('<end>') # 2
    vocab.add_word('<unk>') # 3

    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab

def resize_image(image):
    width, height = image.size
    if width > height:
        left = (width - height) / 2
        right = width - left
        top = 0
        bottom = height
    else:
        top = (height - width) / 2
        bottom = height - top
        left = 0
        right = width
    image = image.crop((left, top, right, bottom))
    image = image.resize([224, 224], Image.ANTIALIAS)
    return image

def main(caption_path,vocab_path,threshold):
    vocab = build_vocab(json_path=caption_path,threshold=threshold)
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)

    # print("resizing images...")
    # splits = ['val','train']
    # 
    # for split in splits:
    #     folder = './data/%s2014' %split
    #     resized_folder = './data/%s2014_resized/' %split
    #     if not os.path.exists(resized_folder):
    #         os.makedirs(resized_folder)
    #     image_files = os.listdir(folder)
    #     num_images = len(image_files)
    #     for i, image_file in enumerate(image_files):
    #         with open(os.path.join(folder, image_file), 'r+b') as f:
    #             with Image.open(f) as image:
    #                 image = resize_image(image)
    #                 image.save(os.path.join(resized_folder, image_file), image.format)
    # 
    # print("done resizing images...")

nltk.download('punkt')

# caption_path = './data/annotations/captions_train2014.json'
caption_path = './data/dataset_coco.json'
vocab_path   = './data/vocab.pkl'
threshold    = 5

main(caption_path,vocab_path,threshold)

words = []
idx = 0
word2idx = {}
vectors = bcolz.carray(np.zeros(1), rootdir='data/glove.6B/6B.300.dat', mode='w')

with open('data/glove.6B/glove.6B.300d.txt', 'rb') as f:
    for l in f:
        line = l.decode().split()
        word = line[0]
        words.append(word)
        word2idx[word] = idx
        idx += 1
        vect = np.array(line[1:]).astype(np.float)
        vectors.append(vect)
    
vectors = bcolz.carray(vectors[1:].reshape((400000, 300)), rootdir='data/glove.6B/6B.300.dat', mode='w')
vectors.flush()
pickle.dump(words, open('data/glove.6B/6B.300_words.pkl', 'wb'))
pickle.dump(word2idx, open('data/glove.6B/6B.300_idx.pkl', 'wb'))

with open('data/vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

print('Loading vocab...')

vectors = bcolz.open('data/glove.6B/6B.300.dat')[:]
words = pickle.load(open('data/glove.6B/6B.300_words.pkl', 'rb'))
word2idx = pickle.load(open('data/glove.6B/6B.300_idx.pkl', 'rb'))

print('glove is loaded...')

glove = {w: vectors[word2idx[w]] for w in words}
matrix_len = len(vocab)
weights_matrix = np.zeros((matrix_len, 300))
words_found = 0

for i, word in enumerate(vocab.idx2word):
    try: 
        weights_matrix[i] = glove[word]
        words_found += 1
    except KeyError:
        weights_matrix[i] = np.random.normal(scale=0.6, size=(300, ))

pickle.dump(weights_matrix, open('data/glove.6B/glove_words.pkl', 'wb'), protocol=2)

print('weights_matrix is created')