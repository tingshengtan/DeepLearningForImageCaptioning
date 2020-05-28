import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data
import tqdm as tqdm
import os
import numpy as np
import time
from torch.nn.utils.rnn import pack_padded_sequence
from nltk.translate.bleu_score import corpus_bleu
from utils import *

class ExperimentBuilder(nn.Module):
    def __init__(self, encoder_model, decoder_model, experiment_name, num_epochs,
                 encoder_lr, decoder_lr, fine_tune_encoder, grad_clip, beam_size, device,
                 train_data, val_data, test_data, word_map, run_full_dataset, continue_from_epoch=-1):
        """
        Initializes an ExperimentBuilder object. Such an object takes care of running training and evaluation of a deep net
        on a given dataset. It also takes care of saving per epoch models and automatically inferring the best val model
        to be used for evaluating the test set metrics.
        :param network_model: A pytorch nn.Module which implements a network architecture.
        :param experiment_name: The name of the experiment. This is used mainly for keeping track of the experiment and creating and directory structure that will be used to save logs, model parameters and other.
        :param num_epochs: Total number of epochs to run the experiment
        :param train_data: An object of the DataProvider type. Contains the training set.
        :param val_data: An object of the DataProvider type. Contains the val set.
        :param test_data: An object of the DataProvider type. Contains the test set.
        :param weight_decay_coefficient: A float indicating the weight decay to use with the adam optimizer.
        :param use_gpu: A boolean indicating whether to use a GPU or not.
        :param continue_from_epoch: An int indicating whether we'll start from scrach (-1) or whether we'll reload a previously saved model of epoch 'continue_from_epoch' and continue training from there.
        """
        super(ExperimentBuilder, self).__init__()

        self.experiment_name = experiment_name
        self.encoder_model   = encoder_model
        self.decoder_model   = decoder_model
                
        # TODO: Experiment whether or not to do this.
        # self.encoder_model.reset_parameters()
        # self.decoder_model.reset_parameters()
        
        self.encoder_lr        = encoder_lr
        self.decoder_lr        = decoder_lr
        self.fine_tune_encoder = fine_tune_encoder
        self.grad_clip         = grad_clip
        self.beam_size         = beam_size
        self.run_full_dataset  = run_full_dataset
        self.device            = device

        if torch.cuda.device_count() > 1:
            self.encoder_model.to(self.device)
            self.decoder_model.to(self.device)
            self.encoder_model = nn.DataParallel(module=self.encoder_model)
            self.decoder_model = nn.DataParallel(module=self.decoder_model)
        else:
            # sends the model from the cpu to the gpu
            self.encoder_model.to(self.device)
            self.decoder_model.to(self.device)
         
        # re-initialize network parameters
        self.train_data   = train_data
        self.val_data     = val_data
        self.test_data    = test_data
        self.word_map     = word_map
        self.rev_word_map = {v: k for k, v in self.word_map.items()}
        self.vocab_size   = len(self.word_map)
        
        if self.fine_tune_encoder:
            self.encoder_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, self.encoder_model.parameters()), lr=self.encoder_lr)
        else:
            self.encoder_optimizer = None
        self.decoder_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, self.decoder_model.parameters()), lr=self.decoder_lr)
                    
        # Generate the directory names
        self.experiment_folder = os.path.abspath(experiment_name)
        self.experiment_logs = os.path.abspath(os.path.join(self.experiment_folder, "result_outputs"))
        self.experiment_saved_models = os.path.abspath(os.path.join(self.experiment_folder, "saved_models"))
        print(self.experiment_folder, self.experiment_logs, self.experiment_saved_models)
                
        # Set best models to be at 0 since we are just starting
        self.best_val_model_idx = 0
        self.best_val_model_bleu4 = 0.

        if not os.path.exists(self.experiment_folder):  # If experiment directory does not exist
            os.mkdir(self.experiment_folder)  # create the experiment directory

        if not os.path.exists(self.experiment_logs):
            os.mkdir(self.experiment_logs)  # create the experiment log directory

        if not os.path.exists(self.experiment_saved_models):
            os.mkdir(self.experiment_saved_models)  # create the experiment saved models directory

        self.num_epochs               = num_epochs
        self.epochs_since_improvement = 0               # keeps track of number of epochs since there's been an improvement in validation BLEU
        self.train_batch_time         = AverageMeter()  # forward prop. + back prop. time
        self.train_data_time          = AverageMeter()  # data loading time
        self.train_losses             = AverageMeter()  # loss (per word decoded)
        self.train_top5accs           = AverageMeter()  # top5 accuracy
        self.val_batch_time           = AverageMeter()
        self.val_losses               = AverageMeter()
        self.val_top5accs             = AverageMeter()
        
        self.criterion = nn.CrossEntropyLoss().to(self.device)  # send the loss computation to the GPU
        
        if continue_from_epoch == -2:
            try:
                self.state = self.load_model(model_save_dir=self.experiment_saved_models,
                                             model_save_name="train_model",
                                             model_idx='latest')
                self.starting_epoch = self.state['current_epoch_idx'] + 1
                self.epochs_since_improvement = self.state['epochs_since_improvement']
                self.bleu4 = self.state['bleu4']
                self.best_val_model_bleu4 = self.state['best_val_model_bleu4']
                self.best_val_model_idx = self.state['best_val_model_idx']
                
                if torch.cuda.device_count() > 1:
                    self.encoder_model.to(self.device)
                    self.decoder_model.to(self.device)
                    self.encoder_model = nn.DataParallel(module=self.encoder_model)
                    self.decoder_model = nn.DataParallel(module=self.decoder_model)
                else:
                    # sends the model from the cpu to the gpu
                    self.encoder_model.to(self.device)
                    self.decoder_model.to(self.device)
                
                if self.fine_tune_encoder is True and self.encoder_optimizer is None:
                    self.encoder_model.fine_tune(self.fine_tune_encoder)
                    self.encoder_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, self.encoder_model.parameters()), lr=self.encoder_lr)
                elif self.fine_tune_encoder is False and self.encoder_optimizer is not None:
                    self.encoder_model.fine_tune(self.fine_tune_encoder)
                    self.encoder_optimizer = None
                    
            except:
                print("Model objects cannot be found, initializing a new model and starting from scratch")
                self.starting_epoch = 0
                self.state = dict()
        
        elif continue_from_epoch != -1:  # if continue from epoch is not -1 then
            self.state = self.load_model(model_save_dir=self.experiment_saved_models,
                                         model_save_name="train_model",
                                         model_idx=continue_from_epoch)
            self.starting_epoch = self.state['current_epoch_idx'] + 1
            self.epochs_since_improvement = self.state['epochs_since_improvement']
            self.bleu4 = self.state['bleu4']
            self.best_val_model_bleu4 = self.state['best_val_model_bleu4']
            self.best_val_model_idx = self.state['best_val_model_idx']
            
            if torch.cuda.device_count() > 1:
                self.encoder_model.to(self.device)
                self.decoder_model.to(self.device)
                self.encoder_model = nn.DataParallel(module=self.encoder_model)
                self.decoder_model = nn.DataParallel(module=self.decoder_model)
            else:
                # sends the model from the cpu to the gpu
                self.encoder_model.to(self.device)
                self.decoder_model.to(self.device)
            
            if self.fine_tune_encoder is True and self.encoder_optimizer is None:
                self.encoder_model.fine_tune(self.fine_tune_encoder)
                self.encoder_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, self.encoder_model.parameters()), lr=self.encoder_lr)
            elif self.fine_tune_encoder is False and self.encoder_optimizer is not None:
                self.encoder_model.fine_tune(self.fine_tune_encoder)
                self.encoder_optimizer = None
        
        else:
            self.starting_epoch = 0
            self.state = dict()

    def get_num_parameters(self):
        total_num_params = 0
        for param in self.parameters():
            total_num_params += np.prod(param.shape)

        return total_num_params

    def run_train_iter(self, imgs, caps, caplens):
        self.train_data_time.update(time.time() - self.train_batch_start)
        
        # Move to GPU, if available
        imgs    = imgs.to(self.device)
        caps    = caps.to(self.device)
        caplens = caplens.to(self.device)
        
        # Forward prop.
        imgs = self.encoder_model(imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = self.decoder_model(imgs, caps, caplens)
        
        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]
        
        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)
        
        # Calculate loss
        loss = self.criterion(scores.data, targets.data)
        
        if self.decoder_model.get_decoder_type() == "lstm":
            # Add doubly stochastic attention regularization
            alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
        
        # Back prop.
        self.decoder_optimizer.zero_grad()
        if self.encoder_optimizer is not None:
            self.encoder_optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients
        if self.grad_clip is not None:
            clip_gradient(self.decoder_optimizer, self.grad_clip)
            if self.encoder_optimizer is not None:
                clip_gradient(self.encoder_optimizer, self.grad_clip)
        
        # Update weights
        self.decoder_optimizer.step()
        if self.encoder_optimizer is not None:
            self.encoder_optimizer.step()
        
        # Keep track of metrics
        top5 = accuracy(scores.data, targets.data, 5)
        self.train_losses.update(loss.item(), sum(decode_lengths))
        self.train_top5accs.update(top5, sum(decode_lengths))
        self.train_batch_time.update(time.time() - self.train_batch_start)
        self.train_batch_start = time.time()
    
    def run_val_evaluation_iter(self, imgs, caps, caplens, allcaps):
        # Move to device, if available
        imgs    = imgs.to(self.device)
        caps    = caps.to(self.device)
        caplens = caplens.to(self.device)
        
        # Forward prop.
        if self.encoder_model is not None:
            imgs = self.encoder_model(imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = self.decoder_model(imgs, caps, caplens)
        
        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]
        
        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores_copy = scores.clone()
        scores  = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)
        
        # Calculate loss
        loss = self.criterion(scores.data, targets.data)
        
        if self.decoder_model.get_decoder_type() == "lstm":
            # Add doubly stochastic attention regularization
            alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
        
        # Keep track of metrics
        self.val_losses.update(loss.item(), sum(decode_lengths))
        top5 = accuracy(scores.data, targets.data, 5)
        self.val_top5accs.update(top5, sum(decode_lengths))
        self.val_batch_time.update(time.time() - self.val_batch_start)
        self.val_batch_start = time.time()

        # Store references (true captions), and hypothesis (prediction) for each image
        # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
        # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
        
        # References
        allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
        for j in range(allcaps.shape[0]):
            img_caps = allcaps[j].tolist()
            img_captions = list(map(lambda c: [w for w in c if w not in {self.word_map['<start>'], self.word_map['<pad>']}], img_caps))  # remove <start> and pads
            self.val_references.append(img_captions)

        # Hypotheses
        _, preds = torch.max(scores_copy, dim=2)
        preds = preds.tolist()
        temp_preds = list()
        for j, p in enumerate(preds):
            temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
        preds = temp_preds
        self.val_hypotheses.extend(preds)
        
        assert len(self.val_references) == len(self.val_hypotheses)

    # TODO: Batched Beam Search
    # Therefore, do not use a batch_size greater than 1 - IMPORTANT!
    def run_test_evaluation_iter(self, image, caps, caplens, allcaps):
        k = self.beam_size
        
        # Move to GPU device, if available
        image = image.to(self.device)  # (1, 3, 256, 256)

        # Encode
        encoder_out = self.encoder_model(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
        enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(3)
        
        # Flatten encoding
        encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)
        
        # We'll treat the problem as having a batch size of k
        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)
        
        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[self.word_map['<start>']]] * k).to(self.device)  # (k, 1)
        
        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)
        
        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(self.device)  # (k, 1)
        
        # Lists to store completed sequences and scores
        complete_seqs = []
        complete_seqs_scores = []
        
        # TODO: Need to modify the codes for TPGN because its init_hidden_state returns 4 values
        # Start decoding
        step = 1
        h, c = self.decoder_model.init_hidden_state(encoder_out)
        
        # TODO: Need to modify the codes for TPGN
        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:
            embeddings = self.decoder_model.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

            awe, _ = self.decoder_model.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

            gate = self.decoder_model.sigmoid(self.decoder_model.f_beta(h))  # gating scalar, (s, encoder_dim)
            awe = gate * awe

            h, c = self.decoder_model.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

            scores = self.decoder_model.fc(h)  # (s, vocab_size)
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
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > 50:
                break
            step += 1
        
        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]

        # References
        img_caps = allcaps[0].tolist()
        img_captions = list(map(lambda c: [w for w in c if w not in {self.word_map['<start>'], self.word_map['<end>'], self.word_map['<pad>']}], img_caps))  # remove <start> and pads
        self.test_references.append(img_captions)

        # Hypotheses
        self.test_hypotheses.append([w for w in seq if w not in {self.word_map['<start>'], self.word_map['<end>'], self.word_map['<pad>']}])

        assert len(self.test_references) == len(self.test_hypotheses)

    def save_model(self, model_save_dir, model_save_name, model_idx, state):
        # TODO: Using state_dict() in the following way does not lead to deterministic results. Fix this in future.
        # state['encoder_model'] = self.encoder_model.state_dict()
        # state['decoder_model'] = self.decoder_model.state_dict()        
        # state['encoder_optimizer'] = self.encoder_optimizer
        # state['decoder_optimizer'] = self.decoder_optimizer.state_dict()
        
        state['encoder_model'] = self.encoder_model
        state['decoder_model'] = self.decoder_model
        state['encoder_optimizer'] = self.encoder_optimizer
        state['decoder_optimizer'] = self.decoder_optimizer
        
        torch.save(state, f=os.path.join(model_save_dir, "{}_{}".format(model_save_name, str(model_idx))))  # save state at prespecified filepath

    def load_model(self, model_save_dir, model_save_name, model_idx):
        state = torch.load(f=os.path.join(model_save_dir, "{}_{}".format(model_save_name, str(model_idx))))
        # TODO: Using state_dict() in the following way does not lead to deterministic results. Fix this in future.
        # TODO: Verify wheher 'strict=Flase' is behaving correctly.
        # self.load_state_dict(state_dict=state['encoder_model'], strict=False)
        # self.load_state_dict(state_dict=state['decoder_model'], strict=False)
        # self.encoder_optimizer = state['encoder_optimizer']
        # self.load_state_dict(state_dict=state['decoder_optimizer'], strict=False)
        
        self.encoder_model = state['encoder_model']
        self.decoder_model = state['decoder_model']
        self.encoder_optimizer = state['encoder_optimizer']
        self.decoder_optimizer = state['decoder_optimizer']
        
        return state

    def run_experiment(self):
        """
        Runs experiment train and evaluation iterations, saving the model and best val model and val model accuracy after each epoch
        :return: The summary current_epoch_losses from starting epoch to total_epochs.
        """
        total_losses = {"curr_epoch": [], "train_acc": [], "train_loss": [],
                        "val_acc": [], "val_loss": [], "bleu4": []}    # initialize a dict to keep the per-epoch metrics
                    
        for i, epoch_idx in enumerate(range(self.starting_epoch, self.num_epochs)):            
            # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
            if self.epochs_since_improvement == 50:
                break
            if self.epochs_since_improvement > 0 and self.epochs_since_improvement % 8 == 0:
                adjust_learning_rate(self.decoder_optimizer, 0.8)
                if self.fine_tune_encoder:
                    adjust_learning_rate(self.encoder_optimizer, 0.8)
            
            epoch_start_time = time.time()
            
            self.decoder_model.train()  # train mode (dropout and batchnorm is used)
            self.encoder_model.train()
            self.train_batch_time.reset()
            self.train_data_time.reset()
            self.train_losses.reset()
            self.train_top5accs.reset()
            self.train_batch_start = time.time()
            with tqdm(total=len(self.train_data)) as pbar_train:  # create a progress bar for training
                for idx, (imgs, caps, caplens) in enumerate(self.train_data):  # get data batches
                    if self.run_full_dataset == "partial" and idx == 100:
                        break
                    self.run_train_iter(imgs=imgs, caps=caps, caplens=caplens)  # take a training iter step
                    pbar_train.update(1)
                    pbar_train.set_description("loss {loss.val:.4f} ({loss.avg:.4f}), top-5 accuracy {top5.val:.3f} ({top5.avg:.3f})".format(loss=self.train_losses, top5=self.train_top5accs))
            
            self.decoder_model.eval()  # eval mode (no dropout or batchnorm)
            if self.encoder_model is not None:
                self.encoder_model.eval()
            self.val_batch_time.reset()
            self.val_losses.reset()
            self.val_top5accs.reset()
            self.val_references = []    # references (true captions) for calculating BLEU-4 score
            self.val_hypotheses = []    # hypotheses (predictions)
            self.val_batch_start = time.time()
            with tqdm(total=len(self.val_data)) as pbar_val:  # create a progress bar for validation
                # explicitly disable gradient calculation to avoid CUDA memory error
                with torch.no_grad():
                    for idx, (imgs, caps, caplens, allcaps) in enumerate(self.val_data):    # get data batches 
                        if self.run_full_dataset == "partial" and idx == 100:
                           break
                        self.run_val_evaluation_iter(imgs=imgs, caps=caps, caplens=caplens, allcaps=allcaps)    # run a validation iter
                        pbar_val.update(1)  # add 1 step to the progress bar
                        pbar_val.set_description("loss {loss.val:.4f} ({loss.avg:.4f}), top-5 accuracy {top5.val:.3f} ({top5.avg:.3f})".format(loss=self.val_losses, top5=self.val_top5accs))
            
            self.bleu4 = corpus_bleu(self.val_references, self.val_hypotheses)
            print('\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(loss=self.val_losses, top5=self.val_top5accs, bleu=self.bleu4))
            
            if self.bleu4 > self.best_val_model_bleu4:
                self.best_val_model_bleu4 = self.bleu4
                self.best_val_model_idx = epoch_idx
                self.epochs_since_improvement = 0
            else:
                self.epochs_since_improvement += 1
                print("\nEpochs since last improvement: %d\n" % self.epochs_since_improvement)

            # get mean of all metrics of current epoch metrics dict, to get them ready for storage and output on the terminal.
            total_losses['curr_epoch'].append(epoch_idx)
            total_losses['train_acc'].append(self.train_top5accs.avg)
            total_losses['train_loss'].append(self.train_losses.avg)
            total_losses['val_acc'].append(self.val_top5accs.avg)
            total_losses['val_loss'].append(self.val_losses.avg)
            total_losses['bleu4'].append(self.bleu4)

            save_statistics(experiment_log_dir=self.experiment_logs, filename='summary.csv',
                            stats_dict=total_losses, current_epoch=i,
                            continue_from_mode=True if (self.starting_epoch != 0 or i > 0) else False) # save statistics to stats file.

            # load_statistics(experiment_log_dir=self.experiment_logs, filename='summary.csv') # How to load a csv file if you need to

            # create a string to use to report our epoch metrics
            current_epoch_losses = {"train_acc": self.train_top5accs.avg, "train_loss": self.train_losses.avg,
                                    "val_acc": self.val_top5accs.avg, "val_loss": self.val_losses.avg}
            out_string = "_".join(["{}_{:.4f}".format(key, value) for key, value in current_epoch_losses.items()])
            epoch_elapsed_time = time.time() - epoch_start_time  # calculate time taken for epoch
            epoch_elapsed_time = "{:.4f}".format(epoch_elapsed_time)
            print("Epoch {}:".format(epoch_idx), out_string, "epoch time", epoch_elapsed_time, "seconds")
            
            self.state['current_epoch_idx'] = epoch_idx
            self.state['epochs_since_improvement'] = self.epochs_since_improvement
            self.state['bleu4'] = self.bleu4
            self.state['best_val_model_bleu4'] = self.best_val_model_bleu4
            self.state['best_val_model_idx'] = self.best_val_model_idx
            self.save_model(model_save_dir=self.experiment_saved_models,
                            # save model and best val idx and best val acc, using the model dir, model name and model idx
                            model_save_name="train_model", model_idx=epoch_idx, state=self.state)
            self.save_model(model_save_dir=self.experiment_saved_models,
                            # save model and best val idx and best val acc, using the model dir, model name and model idx
                            model_save_name="train_model", model_idx='latest', state=self.state)

        print("Model training completed")

        # TODO: Move test set evaluation to 'test_image_captioning_system.py'
        # Disable test set evaluation
        return total_losses, None

        print("Generating test set evaluation metrics")
        self.load_model(model_save_dir=self.experiment_saved_models,
                        model_idx=self.best_val_model_idx,
                        model_save_name="train_model")    # load best validation model
        
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
        self.test_references = []
        self.test_hypotheses = []
        
        with tqdm(total=len(self.test_data), desc="EVALUATING AT BEAM SIZE " + str(self.beam_size)) as pbar_test:  # init a progress bar
            # For each image
            for idx, (image, caps, caplens, allcaps) in enumerate(self.test_data):
                self.run_test_evaluation_iter(image=image, caps=caps, caplens=caplens, allcaps=allcaps)
                
        # Calculate BLEU-4 scores
        self.bleu4 = corpus_bleu(self.test_references, self.test_hypotheses)
        print("\nBLEU-4 score @ beam size of %d is %.4f." % (self.beam_size, self.bleu4))
        
        test_losses = {"bleu4": self.bleu4}
        save_statistics(experiment_log_dir=self.experiment_logs, filename='test_summary.csv',
                        stats_dict=test_losses, current_epoch=0, continue_from_mode=False)    # save test set metrics on disk in .csv format

        return total_losses, test_losses
