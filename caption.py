import torch
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
import argparse
import imageio
import os
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def caption_image_beam_search(encoder, decoder, image_path, image_length, image_width, word_map, encoder_type, decoder_type, beam_size=3):
    """
    Reads an image and captions it with beam search.

    :param encoder: encoder model
    :param decoder: decoder model
    :param image_path: path to image
    :param image_lentgh: image length
    :param image_width: image width
    :param word_map: word map
    :param beam_size: number of sequences to consider at each decode-step
    :return: caption, weights for visualization
    """
    
    k          = beam_size
    vocab_size = len(word_map)

    # Read image and process
    img = imageio.imread(image_path)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
    img = np.array(Image.fromarray(img).resize((image_length, image_width)))
    img = img.transpose(2, 0, 1)
    img = img / 255.
    img = torch.FloatTensor(img).to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])
    image = transform(img)  # (3, 256, 256)

    # Encode
    image          = image.unsqueeze(0)  # (1, 3, 256, 256)
    encoder_out    = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
    enc_image_size = encoder_out.size(1)
    encoder_dim    = encoder_out.size(3)

    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    num_pixels  = encoder_out.size(1)

    # We'll treat the problem as having a batch size of k
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

    if decoder_type == "lstm":
        # Tensor to store top k sequences' alphas; now they're just 1s
        seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)  # (k, 1, enc_image_size, enc_image_size)

    # Lists to store completed sequences, their alphas and scores
    complete_seqs        = []
    complete_seqs_alpha  = []
    complete_seqs_scores = []
    
    # Start decoding
    step = 1
    if decoder_type == "lstm":
        h, c = decoder.init_hidden_state(encoder_out)
    
    elif decoder_type == "tpgn":
        h_s, c_s, h_u, c_u = decoder.init_hidden_state(encoder_out)    # (k, decoder_dim)
        decoder_dim        = h_s.shape[1]
    
    else:
        raise Exception("ERROR: Invalid decoder type.")
        
    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:
        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)
        
        if decoder_type == "lstm":
            awe, alpha = decoder.attention(encoder_out, h)                                   # (s, encoder_dim), (s, num_pixels)
            alpha      = alpha.view(-1, enc_image_size, enc_image_size)                      # (s, enc_image_size, enc_image_size)
            gate       = decoder.sigmoid(decoder.f_beta(h))                                  # gating scalar, (s, encoder_dim)
            awe        = gate * awe                                                          # (s, encoder_dim)
            h, c       = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))    # (s, decoder_dim)
            scores     = decoder.fc(h)                                                       # (s, vocab_size)
            
        elif decoder_type == "tpgn":
            h_s, c_s = decoder.lstm_cell_s(embeddings, h_s, h_u, c_s)    # (s, decoder_dim, decoder_dim)
            h_u, c_u = decoder.lstm_cell_u(embeddings, h_u, h_s, c_u)    # (s, decoder_dim)
            
            encoded_sentence = torch.zeros(k, decoder_dim ** 2, decoder_dim ** 2).to(device)    # (s, hidden_d ** 2, hidden_d **2)
            for d in range(decoder_dim):
                encoded_sentence[:, d * decoder_dim:d * decoder_dim + h_s.shape[1], d * decoder_dim:d * decoder_dim + h_s.shape[2]] += h_s
            
            unbinding_vector = decoder.tanh(decoder.unbind(h_u))                                           # (s, hidden_d **2)
            filler_vector    = torch.matmul(encoded_sentence, unbinding_vector.unsqueeze(2)).squeeze(2)    # (s, hidden_d ** 2)
            scores           = decoder.fc(filler_vector)                                                   # (s, vocab_size)
            
        else:
            raise Exception("ERROR: Invalid decoder type.")
        
        scores = F.log_softmax(scores, dim=1)

        # Add
        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            
        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words / vocab_size  # (s)
        next_word_inds = top_k_words % vocab_size  # (s)

        # Add new words to sequences, alphas
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
        if decoder_type == "lstm":
            seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],
                                   dim=1)  # (s, step+1, enc_image_size, enc_image_size)

        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            if decoder_type == "lstm":
                complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break
            
        seqs = seqs[incomplete_inds]    
            
        if decoder_type == "lstm":
            seqs_alpha = seqs_alpha[incomplete_inds]
            h          = h[prev_word_inds[incomplete_inds]]
            c          = c[prev_word_inds[incomplete_inds]]
            
        elif decoder_type == "tpgn":
            h_s = h_s[prev_word_inds[incomplete_inds]]
            c_s = c_s[prev_word_inds[incomplete_inds]]
            h_u = h_u[prev_word_inds[incomplete_inds]]
            c_u = c_u[prev_word_inds[incomplete_inds]]
        
        else:
            raise Exception("ERROR: Invalid decoder type.")    
        
        encoder_out  = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)
        
        # Break if things have been going on too long
        if step > 50:
            break
        step += 1

    i   = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]
    if decoder_type == "lstm":
        alphas = complete_seqs_alpha[i]
    else:
        alphas = None

    return seq, alphas

def visualize_att(image_path, seq, alphas, rev_word_map, smooth=True):
    """
    Visualizes caption with weights at every word.

    Adapted from paper authors' repo: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb

    :param image_path: path to image that has been captioned
    :param seq: caption
    :param alphas: weights
    :param rev_word_map: reverse word mapping, i.e. ix2word
    :param smooth: smooth weights?
    """
    image = Image.open(image_path)
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

    words = [rev_word_map[ind] for ind in seq]

    for t in range(len(words)):
        if t > 50:
            break
            
        plt.subplot(np.ceil(len(words) / 5.), 5, t + 1)
        plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)
        plt.imshow(image)
        
        current_alpha = alphas[t, :]
        
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha.numpy(), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha.numpy(), [14 * 24, 14 * 24])
        
        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)
            
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
        
    plt.show()

def save_captions(image_name, seq, rev_word_map, output_file):
    output_file.write("%s:\n" % image_name)
    output_file.write("Hypothesis:\n")
    words = [rev_word_map[ind] for ind in seq]
    for word in words:
        output_file.write("%s " % word)
    output_file.write("\n\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show, Attend, and Tell - Tutorial - Generate Caption')

    parser.add_argument('--img', '-i', help='path to images')
    parser.add_argument('--img_length', default=256, type=int, help='image length')
    parser.add_argument('--img_width', default=256, type=int, help='image width')
    parser.add_argument('--model', '-m', help='path to model')
    parser.add_argument('--encoder_type', type=str, default="resnet", help='resnet or densenet')
    parser.add_argument('--decoder_type', type=str, default="lstm", help='lstm or tpgn')
    parser.add_argument('--word_map', '-wm', help='path to word map JSON')
    parser.add_argument('--output_file', help='output file file store captions generated')
    parser.add_argument('--beam_size', '-b', default=5, type=int, help='beam size for beam search')
    parser.add_argument('--dont_smooth', dest='smooth', action='store_false', help='do not smooth alpha overlay')

    args = parser.parse_args()
    
    if device == torch.device('cpu'):
        state = torch.load(f=args.model, map_location=torch.device('cpu'))
    else:
        state = torch.load(f=args.model)
    
    encoder = state['encoder_model']
    decoder = state['decoder_model']
    
    encoder.eval()
    decoder.eval()
    
    # Load word map (word2ix)
    with open(args.word_map, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}  # ix2word
    
    output_file = open(args.output_file, 'w')
    
    # images = ["COCO_val2014_000000000785.jpg",
    #           "COCO_val2014_000000001584.jpg",
    #           "COCO_val2014_000000008548.jpg",
    #           "COCO_val2014_000000011796.jpg",
    #           "COCO_val2014_000000012764.jpg",
    #           "COCO_val2014_000000012927.jpg",
    #           "COCO_val2014_000000029573.jpg",
    #           "COCO_val2014_000000029727.jpg",
    #           "COCO_val2014_000000035807.jpg",
    #           "COCO_val2014_000000049133.jpg"]
    
    images = ["COCO_val2014_000000018534.jpg",
              "COCO_val2014_000000021284.jpg",
              "COCO_val2014_000000036149.jpg",
              "COCO_val2014_000000037325.jpg",
              "COCO_val2014_000000230240.jpg",
              "COCO_val2014_000000257219.jpg"]
    
    output_file.write("Using bean size of %d.\n\n" % args.beam_size)    
    for image in images:
        seq, alphas = caption_image_beam_search(encoder, decoder, os.path.join(args.img, image), args.img_length, args.img_width, word_map, args.encoder_type, args.decoder_type, args.beam_size)
        save_captions(image, seq, rev_word_map, output_file)
        print("Processed %s." % image)
    
    # Encode, decode with attention and beam search
    # seq, alphas = caption_image_beam_search(encoder, decoder, args.img, word_map, args.beam_size)
    # alphas = torch.FloatTensor(alphas)
    
    # Visualize caption and attention of best sequence
    # visualize_att(args.img, seq, alphas, rev_word_map, args.smooth)
