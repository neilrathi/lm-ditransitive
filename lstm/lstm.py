import argparse
import logging
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dictionary_corpus import Corpus, Dictionary, tokenize
from utils import batchify

parser = argparse.ArgumentParser(add_help=False)

parser.add_argument('--cuda', action='store_true',
                       help='use CUDA')
parser.add_argument('--train', action='store_true',
                       help='training mode?')
parser.add_argument('--test', action='store_true',
                       help='testing mode?')

args = parser.parse_args()

emsize = 200
nhid = 200
nlayers = 2
dropout = 0.2

lr = 20
clip = 0.25
epochs = 10
batch_size = 20

seed = 1111
log_interval = 5000
log = 'log.txt'

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(), logging.FileHandler(log)])

logging.info(args)

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder.
        ntoken: vocab size
        nip: embedding size
    """

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        #print("hidden", hidden, hidden[0].size())

        # take last output of the sequence
        output = output[-1]

        #print(output)
        #decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        #print(output.size())
        decoded = self.decoder(output.view(-1, output.size(1)))
        #print(output.view(output.size(0)*output.size(1), output.size(2)))
        #print(decoded)
        return decoded

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (weight.new(self.nlayers, bsz, self.nhid).zero_(),
                    weight.new(self.nlayers, bsz, self.nhid).zero_())
        else:
            return weight.new(self.nlayers, bsz, self.nhid).zero_()




def get_batch(source, i, seq_length):
    seq_len = min(seq_length, len(source) - 1 - i)
    #print("Sequence length", seq_len)
    #print(source)
    data = source[i:i+seq_len]
    #print(data)
    #> predict the sequences shifted by one word
    target = source[i+seq_len].view(-1)
    #print(target)
    return data, target

def create_target_mask(test_file, gold_file, index_col):
    sents = open(test_file, "r", encoding="utf8").readlines()
    golds = open(gold_file, "r", encoding="utf8").readlines()
    #TODO optimize by initializaing np.array of needed size and doing indexing
    targets = []
    for sent, gold in zip(sents, golds):
        # constr_id, sent_id, word_id, pos, morph
        target_idx = int(gold.split()[index_col])
        len_s = len(sent.split(" "))
        t_s = [0] * len_s
        t_s[target_idx] = 1
        #print(sent.split(" ")[target_idx])
        targets.extend(t_s)
    return np.array(targets)


def evaluate_perplexity(data_source, exclude_oov=False):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    len_data = 0
    unk_idx = corpus.dictionary.word2idx["<unk>"]

    if args.cuda:
        torch_range =  torch.cuda.LongTensor()
    else:
        torch_range = torch.LongTensor()
    
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1):
            hidden = model.init_hidden(eval_batch_size)
            data, targets = get_batch(data_source, i, seq_length)
            #> output has size seq_length x batch_size x vocab_size
            output = model(data, hidden)
            output_flat = output.view(-1, ntokens)

            # excluding OOV
            if exclude_oov:
                subset = targets != unk_idx
                subset = subset.data
                targets = targets[subset]
                output_flat = output_flat[torch.arange(0, output_flat.size(0), out=torch_range)[subset]]

            total_loss += targets.size(0) * nn.CrossEntropyLoss()(output_flat, targets).data
            len_data += targets.size(0)

    return total_loss.item() / len_data


def evaluate_on_mask(data_source, mask):

    model.eval()

    idx2word = dictionary.idx2word

    cur_surp = 0
    cur_sent = []

    for i in range(0, data_source.size(0) - 1):
        hidden = model.init_hidden(eval_batch_size)
        data, targets = get_batch(data_source, i, seq_length)
        _, targets_mask = get_batch(mask, i, seq_length)
        #print(targets_mask.size())
        #> output has size seq_length x batch_size x vocab_size
        output = model(data, hidden)
        output_flat = output.view(-1, ntokens)

        log_probs = F.log_softmax(output_flat)
        # print("Log probs size", log_probs.size())
        # print("Target size", targets.size())

        log_probs_np = log_probs.data.cpu().numpy()
        subset = targets_mask.data.numpy().astype(bool)

        for scores, correct_label in zip(log_probs_np[subset], targets.data.cpu().numpy()[subset]):
            cur_surp += scores[correct_label]
            cur_sent.append(idx2word[correct_label])
            if idx2word[correct_label] == '<eos>':
                f_output.write(' '.join(cur_sent) + '\t' + str(cur_surp) + '\n')
                cur_surp = 0
                cur_sent = []

            # print(idx2word[correct_label], scores[correct_label])
            # f_output.write("\t".join(str(s) for s in scores) + "\n")

    #return total_loss[0] /len(data_source)



###############################################################################
# Training code
###############################################################################

def train():

    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    start_time = time.time()

    criterion = nn.CrossEntropyLoss()

    for batch, i in enumerate(range(0, train_data.size(0) - 1)):
        #> i is the starting index of the batch
        #> batch is the number of the batch
        #> data is a tensor of size seq_length x batch_size, where each element is an index from input vocabulary
        #> targets is a vector of length seq_length x batch_size
        data, targets = get_batch(train_data, i, seq_length)

        hidden = model.init_hidden(batch_size)
        model.zero_grad()

        output = model(data, hidden)

        #> output.view(-1, ntokens) transforms a tensor to a longer tensor of size
        #> (seq_length x batch_size) x output_vocab_size
        #> which matches targets in length
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm(model.parameters(), clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.data

        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss.item() / log_interval
            elapsed = time.time() - start_time
            #logging.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
            #        'loss {:5.2f} | ppl {:8.2f}'.format(
            #    epoch, batch, len(train_data) // seq_length, lr,
            #    elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)))
            logging.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                         'loss {:5.2f}'.format(epoch, batch, len(train_data), lr,
                              elapsed * 1000 / log_interval, cur_loss))
            total_loss = 0
            start_time = time.time()



# Set the random seed manually for reproducibility.
torch.manual_seed(seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(seed)

###############################################################################
# Load data
###############################################################################

if args.train:
    for k in range(1, 5):
        for seq_length in range(1, 7):
            logging.info(f'----------- new value of n = {seq_length}, k = {k} -----------')
            logging.info("Loading data")
            corpus = Corpus(f'../data/{k}/')
            # logging.info(corpus.train)

            ntokens = len(corpus.dictionary)
            logging.info("Vocab size %d", ntokens)

            logging.info("Batchying..")
            eval_batch_size = 256

            train_data = batchify(corpus.train, batch_size, args.cuda)
            # logging.info("Train data size", train_data.size())
            val_data = batchify(corpus.valid, eval_batch_size, args.cuda)
            test_data = batchify(corpus.test, eval_batch_size, args.cuda)

            logging.info("Building the model")

            model = RNNModel('LSTM', ntokens, emsize, nhid, nlayers, dropout)
            if args.cuda:
                model.cuda()

            # Loop over epochs.
            best_val_loss = None

            try:
                for epoch in range(1, epochs+1):
                    epoch_start_time = time.time()

                    train()

                    val_loss = evaluate_perplexity(val_data)
                    logging.info('-' * 89)
                    logging.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                            'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                                    val_loss, math.exp(val_loss)))
                    logging.info('-' * 89)
                    # Save the model if the validation loss is the best we've seen so far.
                    if not best_val_loss or val_loss < best_val_loss:
                        with open(f'models/{k}/model-{seq_length}.pt', 'wb') as f:
                            torch.save(model, f)
                        best_val_loss = val_loss
                    else:
                        # Anneal the learning rate if no improvement has been seen in the validation dataset.
                        lr /= 4.0
            except KeyboardInterrupt:
                logging.info('-' * 89)
                logging.info('Exiting from training early')

            # Load the best saved model.
            with open(f'models/{k}/model-{seq_length}.pt', 'rb') as f:
                model = torch.load(f)

            # Run on valid data with OOV excluded
            test_loss = evaluate_perplexity(val_data, exclude_oov=True)
            logging.info('=' * 89)
            logging.info('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(test_loss, math.exp(test_loss)))
            logging.info('=' * 89)

#####################################
#   Testing                         #
#####################################


if args.test:
    for k in range(1, 5):
        for seq_length in range(1, 7):
            print(f'----------- new value of n = {seq_length}, k = {k} -----------')
            with open(f'models/{k}/model-{seq_length}.pt', 'rb') as f:
                if args.cuda:
                    model = torch.load(f)
                    model.cuda()
                else:
                    # to convert model trained on cuda to cpu model
                    model = torch.load(f, map_location=lambda storage, loc: storage)
                    model.cpu()

            model.eval()

            eval_batch_size = 1

            corpus = Corpus(f'../data/{k}/')
            dictionary = Dictionary(f'../data/{k}/') 
            ntokens = len(corpus.dictionary)
            
            # print("Vocab size", ntokens)
            # print("TESTING")

            mask = create_target_mask(f'../data/{k}/test-clean.txt', f'../data/{k}/test.gold', 0)
            mask_data = batchify(torch.LongTensor(mask), eval_batch_size, False)
            test_data = batchify(tokenize(corpus.dictionary, f'../data/{k}/test-clean.txt'), eval_batch_size, args.cuda)

            f_output = open(f'../data/{k}/output/test-{seq_length}.txt', 'w')
            evaluate_on_mask(test_data, mask_data)
            f_output.close()