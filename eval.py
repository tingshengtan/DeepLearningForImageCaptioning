import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data
import tqdm as tqdm
import os
import numpy as np
import time
import csv
import data_providers as data_providers
from torchvision import transforms
from arg_extractor import get_args
from torch.nn.utils.rnn import pack_padded_sequence
from nltk.translate.bleu_score import corpus_bleu
from utils import *
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

# References: http://cocodataset.org/#captions-eval
#             http://cocodataset.org/#format-results
#             https://github.com/salaniz/pycocoevalcap
#             https://www.nltk.org/api/nltk.translate.html
#             https://github.com/yunjey/show-attend-and-tell/blob/master/core/bleu.py
#             https://github.com/tylin/coco-caption/blob/master/pycocoevalcap/bleu/bleu.py
#             https://gist.github.com/kracwarlock/c979b10433fe4ac9fb97
#             https://github.com/ruotianluo/ImageCaptioning.pytorch
#             https://github.com/salaniz/pycocoevalcap

class COCOEvalCap:
    def __init__(self,images,gts,res):
        self.evalImgs  = []
        self.eval      = {}
        self.imgToEval = {}
        self.params    = {'image_id': images}
        self.gts       = gts
        self.res       = res

    def evaluate(self):
        imgIds = self.params['image_id']
        gts    = self.gts
        res    = self.res

        # =================================================
        # Set up scorers
        # =================================================
        print('tokenization...')
        tokenizer = PTBTokenizer()
        gts       = tokenizer.tokenize(gts)
        res       = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = [(Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
                   (Meteor(),"METEOR"),
                   (Rouge(), "ROUGE_L"),
                   (Cider(), "CIDEr")]

        # =================================================
        # Compute scores
        # =================================================
        eval = {}
        for scorer, method in scorers:
            print('computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, imgIds, m)
                    print("%s: %0.3f"%(m, sc))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, imgIds, method)
                print("%s: %0.3f"%(method, score))
        self.setEvalImgs()

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId]             = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]

class ModelTester(nn.Module):
    def __init__(self):
        super(ModelTester, self).__init__()
        
        self.args, self.device = get_args()
        
        self.rng = np.random.RandomState(seed=self.args.seed)
        torch.manual_seed(seed=self.args.seed)

        self.experiment_name  = self.args.experiment_name
        self.data_folder      = os.environ['DATA_FOLDER']
        self.dataset_name     = self.args.dataset_name
        self.data_name        = self.args.data_name
        self.run_full_dataset = self.args.run_full_dataset
        
        self.experiment_folder       = os.path.abspath(self.experiment_name)
        self.experiment_logs         = os.path.abspath(os.path.join(self.experiment_folder, "result_outputs"))
        self.experiment_saved_models = os.path.abspath(os.path.join(self.experiment_folder, "saved_models"))
        print(self.experiment_folder, self.experiment_logs, self.experiment_saved_models)
        
        self.encoder_type       = self.args.encoder_type
        self.decoder_type       = self.args.decoder_type
        self.best_val_model_idx = self.args.best_val_model_idx_for_test
        self.beam_size          = self.args.beam_size

    def load_model(self, model_save_dir, model_save_name, model_idx):
        if self.device == torch.device('cpu'):
            state = torch.load(f=os.path.join(model_save_dir, "{}_{}".format(model_save_name, str(model_idx))), map_location=torch.device('cpu'))
        else:
            state = torch.load(f=os.path.join(model_save_dir, "{}_{}".format(model_save_name, str(model_idx))))
        
        self.encoder_model     = state['encoder_model']
        self.decoder_model     = state['decoder_model']
        self.encoder_optimizer = state['encoder_optimizer']
        self.decoder_optimizer = state['decoder_optimizer']
            
        return state

    # TODO: Batched Beam Search
    # Therefore, do not use a batch_size greater than 1 - IMPORTANT!
    def run_test_evaluation_iter(self, image, caps, caplens, allcaps):
        k = self.beam_size
        
        # Move to GPU device, if available
        image = image.to(self.device)    # (1, 3, 256, 256)
        
        # Encode
        encoder_out    = self.encoder_model(image)    # (1, enc_image_size, enc_image_size, encoder_dim)
        enc_image_size = encoder_out.size(1)
        encoder_dim    = encoder_out.size(3)
        
        # Flatten encoding
        encoder_out = encoder_out.view(1, -1, encoder_dim)    # (1, num_pixels, encoder_dim)
        num_pixels  = encoder_out.size(1)
        
        # We'll treat the problem as having a batch size of k
        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)    # (k, num_pixels, encoder_dim)
        
        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[self.word_map['<start>']]] * k).to(self.device)    # (k, 1)
        
        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words    # (k, 1)
        
        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(self.device)    # (k, 1)
        
        # Lists to store completed sequences and scores
        complete_seqs        = []
        complete_seqs_scores = []
        
        # Start decoding
        step = 1
        if self.decoder_type == "lstm":
            h, c = self.decoder_model.init_hidden_state(encoder_out)
        
        elif self.decoder_type == "tpgn":
            h_s, c_s, h_u, c_u = self.decoder_model.init_hidden_state(encoder_out)    # (k, decoder_dim)
            self.decoder_dim   = h_s.shape[1]
        
        else:
            raise Exception("ERROR: Invalid decoder type.")
        
        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:            
            embeddings = self.decoder_model.embedding(k_prev_words).squeeze(1)    # (s, embed_dim)
            
            if self.decoder_type == "lstm":
                awe, _ = self.decoder_model.attention(encoder_out, h)                                   # (s, encoder_dim), (s, num_pixels)
                gate   = self.decoder_model.sigmoid(self.decoder_model.f_beta(h))                       # gating scalar, (s, encoder_dim)
                awe    = gate * awe                                                                     # (s, encoder_dim)
                h, c   = self.decoder_model.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))    # (s, decoder_dim)
                scores = self.decoder_model.fc(h)                                                       # (s, vocab_size)
                
                
            elif self.decoder_type == "tpgn":
                h_s, c_s = self.decoder_model.lstm_cell_s(embeddings, h_s, h_u, c_s)    # (s, decoder_dim, decoder_dim)
                h_u, c_u = self.decoder_model.lstm_cell_u(embeddings, h_u, h_s, c_u)    # (s, decoder_dim)
                
                encoded_sentence = torch.zeros(k, self.decoder_dim ** 2, self.decoder_dim ** 2).to(self.device)    # (s, hidden_d ** 2, hidden_d **2)
                for d in range(self.decoder_dim):
                    encoded_sentence[:, d * self.decoder_dim:d * self.decoder_dim + h_s.shape[1], d * self.decoder_dim:d * self.decoder_dim + h_s.shape[2]] += h_s
                
                unbinding_vector = self.decoder_model.tanh(self.decoder_model.unbind(h_u))                     # (s, hidden_d **2)
                filler_vector    = torch.matmul(encoded_sentence, unbinding_vector.unsqueeze(2)).squeeze(2)    # (s, hidden_d ** 2)
                scores           = self.decoder_model.fc(filler_vector)                                        # (s, vocab_size)
                
            else:
                raise Exception("ERROR: Invalid decoder type.")
            
            scores = F.log_softmax(scores, dim=1)
            
            # Add
            scores = top_k_scores.expand_as(scores) + scores    # (s, vocab_size)
            
            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)

            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words / self.vocab_size  # (s)
            next_word_inds = top_k_words % self.vocab_size  # (s)

            # Add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
                        
            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != self.word_map['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
                        
            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            
            seqs = seqs[incomplete_inds]
            
            if self.decoder_type == "lstm":
                h = h[prev_word_inds[incomplete_inds]]
                c = c[prev_word_inds[incomplete_inds]]
                
            elif self.decoder_type == "tpgn":
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

        if self.dataset_under_eval == "test":
            # References
            img_caps     = allcaps[0].tolist()
            img_captions = list(map(lambda c: [w for w in c if w not in {self.word_map['<start>'], self.word_map['<end>'], self.word_map['<pad>']}], img_caps))  # remove <start> and pads
            self.test_references.append(img_captions)

            # Hypotheses
            self.test_hypotheses.append([w for w in seq if w not in {self.word_map['<start>'], self.word_map['<end>'], self.word_map['<pad>']}])

            assert len(self.test_references) == len(self.test_hypotheses)
            
        elif self.dataset_under_eval == "val":
            # References
            img_caps     = allcaps[0].tolist()
            img_captions = list(map(lambda c: [w for w in c if w not in {self.word_map['<start>'], self.word_map['<end>'], self.word_map['<pad>']}], img_caps))  # remove <start> and pads
            self.val_references.append(img_captions)

            # Hypotheses
            self.val_hypotheses.append([w for w in seq if w not in {self.word_map['<start>'], self.word_map['<end>'], self.word_map['<pad>']}])

            assert len(self.val_references) == len(self.val_hypotheses)
            
        else:
            raise Exception("ERROR: self.dataset_under_eval has an invalid value.")

    def calculate_metrics(self, rng, datasetGTS, datasetRES):
        imgIds = rng
        gts    = {}
        res    = {}

        imgToAnnsGTS = {ann['image_id']: [] for ann in datasetGTS['annotations']}
        for ann in datasetGTS['annotations']:
            imgToAnnsGTS[ann['image_id']] += [ann]

        imgToAnnsRES = {ann['image_id']: [] for ann in datasetRES['annotations']}
        for ann in datasetRES['annotations']:
            imgToAnnsRES[ann['image_id']] += [ann]

        for imgId in imgIds:
            gts[imgId] = imgToAnnsGTS[imgId]
            res[imgId] = imgToAnnsRES[imgId]

        evalObj = COCOEvalCap(imgIds,gts,res)
        evalObj.evaluate()
        
        return evalObj.eval

    def evaluate_model(self):
        if self.dataset_name == 'coco':
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            
            valset        = data_providers.CaptionDataset(self.data_folder, self.data_name, 'VAL', transform=transforms.Compose([normalize]))
            self.val_data = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=True, num_workers=1, pin_memory=True)
            
            testset        = data_providers.CaptionDataset(self.data_folder, self.data_name, 'TEST', transform=transforms.Compose([normalize]))
            self.test_data = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=1, pin_memory=True)
        
        word_map_file = os.path.join(self.data_folder, 'WORDMAP_' + self.data_name + '.json')
        with open(word_map_file, 'r') as j:
            self.word_map = json.load(j)
        self.rev_word_map = {v: k for k, v in self.word_map.items()}    # ix2word
        self.vocab_size   = len(self.word_map)
        
        # Load best validation mdoel
        self.load_model(model_save_dir=self.experiment_saved_models,
                        model_idx=self.best_val_model_idx,
                        model_save_name="train_model")
        
        if torch.cuda.device_count() > 1:
            self.encoder_model.to(self.device)
            self.decoder_model.to(self.device)
            self.encoder_model = nn.DataParallel(module=self.encoder_model)
            self.decoder_model = nn.DataParallel(module=self.decoder_model)
        else:
            # sends the model from the cpu to the gpu
            self.encoder_model.to(self.device)
            self.decoder_model.to(self.device)

        self.encoder_model.eval()
        self.decoder_model.eval()
        
        # Lists to store references (true captions), and hypothesis (prediction) for each image
        # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
        # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
        self.dataset_under_eval = "test"
        self.test_references    = []
        self.test_hypotheses    = []
        self.test_ground_truths = {'annotations': []}
        self.test_results       = {'annotations': []}
        self.image_count        = 0
        
        with tqdm(total=len(self.test_data), desc="EVALUATING TEST DATASET AT BEAM SIZE " + str(self.beam_size)) as pbar_test:  # init a progress bar
            # For each image
            for idx, (image, caps, caplens, allcaps) in enumerate(self.test_data):
                if self.run_full_dataset == "partial" and idx == 10:
                    break
                self.run_test_evaluation_iter(image=image, caps=caps, caplens=caplens, allcaps=allcaps)
                pbar_test.update(1)
                self.image_count += 1
        
        self.image_ids = range(self.image_count)
        
        for image_id, captions in enumerate(self.test_references):
            for caption in captions:
                words    = [self.rev_word_map[ind] for ind in caption]
                sentence = " ".join(words)
                self.test_ground_truths['annotations'].append({'image_id': image_id, 'caption': sentence})
            
        for image_id, caption in enumerate(self.test_hypotheses):
            words    = [self.rev_word_map[ind] for ind in caption]
            sentence = " ".join(words)
            self.test_results['annotations'].append({'image_id': image_id, 'caption': sentence})
        
        save_to_stats_pkl_file(self.experiment_logs, "train_model_%d_test_ground_truths" % self.best_val_model_idx, self.test_ground_truths)
        save_to_stats_pkl_file(self.experiment_logs, "train_model_%d_test_results" % self.best_val_model_idx, self.test_results)
        
        # Calculate scores using MS COCO caption evaluation
        self.test_scores = self.calculate_metrics(self.image_ids, self.test_ground_truths, self.test_results)
        print("MS COCO test dataset caption evaluation with beam size of %d:" % self.beam_size)
        print(self.test_scores)
        
        # Calculate BLEU-4 scores using NLTK
        self.test_bleu4 = corpus_bleu(self.test_references, self.test_hypotheses)
        print("NLTK test dataset caption evaluation with beam size of %d:" % self.beam_size)
        print("Bleu_4 is %.4f" % self.test_bleu4)
        
        with open(os.path.join(self.experiment_logs, "train_model_%d_test_summary.csv" % self.best_val_model_idx), 'w') as f:
            writer = csv.writer(f)
            
            writer.writerow(["Best_val_model_idx",
                             "Beam_size",
                             "Bleu_1",
                             "Bleu_2",
                             "Bleu_3",
                             "Bleu_4",
                             "Bleu_4_NLTK",
                             "METEOR",
                             "ROUGE_L",
                             "CIDEr"])
            
            writer.writerow([self.best_val_model_idx,
                             self.beam_size,
                             self.test_scores["Bleu_1"],
                             self.test_scores["Bleu_2"],
                             self.test_scores["Bleu_3"],
                             self.test_scores["Bleu_4"],
                             self.test_bleu4,
                             self.test_scores["METEOR"],
                             self.test_scores["ROUGE_L"],
                             self.test_scores["CIDEr"]])
        
        self.dataset_under_eval = "val"
        self.val_references    = []
        self.val_hypotheses    = []
        self.val_ground_truths = {'annotations': []}
        self.val_results       = {'annotations': []}
        self.image_count        = 0
        
        with tqdm(total=len(self.val_data), desc="EVALUATING VAL DATASET AT BEAM SIZE " + str(self.beam_size)) as pbar_val:  # init a progress bar
            # For each image
            for idx, (image, caps, caplens, allcaps) in enumerate(self.val_data):
                if self.run_full_dataset == "partial" and idx == 10:
                    break
                self.run_test_evaluation_iter(image=image, caps=caps, caplens=caplens, allcaps=allcaps)
                pbar_val.update(1)
                self.image_count += 1
        
        self.image_ids = range(self.image_count)
        
        for image_id, captions in enumerate(self.val_references):
            for caption in captions:
                words    = [self.rev_word_map[ind] for ind in caption]
                sentence = " ".join(words)
                self.val_ground_truths['annotations'].append({'image_id': image_id, 'caption': sentence})
            
        for image_id, caption in enumerate(self.val_hypotheses):
            words    = [self.rev_word_map[ind] for ind in caption]
            sentence = " ".join(words)
            self.val_results['annotations'].append({'image_id': image_id, 'caption': sentence})

        save_to_stats_pkl_file(self.experiment_logs, "train_model_%d_val_ground_truths" % self.best_val_model_idx, self.val_ground_truths)
        save_to_stats_pkl_file(self.experiment_logs, "train_model_%d_val_results" % self.best_val_model_idx, self.val_results)
        
        # Calculate scores using MS COCO caption evaluation
        self.val_scores = self.calculate_metrics(self.image_ids, self.val_ground_truths, self.val_results)
        print("MS COCO val dataset caption evaluation with beam size of %d:" % self.beam_size)
        print(self.val_scores)
        
        # Calculate BLEU-4 scores using NLTK
        self.val_bleu4 = corpus_bleu(self.val_references, self.val_hypotheses)
        print("NLTK val dataset caption evaluation with beam size of %d:" % self.beam_size)
        print("Bleu_4 is %.4f" % self.val_bleu4)
        
        with open(os.path.join(self.experiment_logs, "train_model_%d_val_summary.csv" % self.best_val_model_idx), 'w') as f:
            writer = csv.writer(f)
            
            writer.writerow(["Best_val_model_idx",
                             "Beam_size",
                             "Bleu_1",
                             "Bleu_2",
                             "Bleu_3",
                             "Bleu_4",
                             "Bleu_4_NLTK",
                             "METEOR",
                             "ROUGE_L",
                             "CIDEr"])
            
            writer.writerow([self.best_val_model_idx,
                             self.beam_size,
                             self.val_scores["Bleu_1"],
                             self.val_scores["Bleu_2"],
                             self.val_scores["Bleu_3"],
                             self.val_scores["Bleu_4"],
                             self.val_bleu4,
                             self.val_scores["METEOR"],
                             self.val_scores["ROUGE_L"],
                             self.val_scores["CIDEr"]])

if __name__ == '__main__':
    model_tester = ModelTester()
    model_tester.evaluate_model()