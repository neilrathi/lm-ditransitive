import os
from collections import defaultdict

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
parser.add_argument('--save', type=str, default='model.pt',
                       help='path to save the final model')

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
log_interval = 200
log = 'log.txt'
seq_length = 5

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(), logging.FileHandler(log)])

logging.info(args)

class RNNModel(nn.Module):
    """
        RNN module, with an encoder, decoder, and recurrent module.
            - ntoken: vocab size
            - ninp: embedding size
            - nhid: # hidden units per layer
            - nlayers: # layers
    """
    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        
        self.rnn = getattr(nn, 'LSTM')(ninp, nhid, nlayers, dropout=dropout)
        
        self.decoder = nn.Linear(nhid, ntoken)

        self.init_weights()

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

        # take last output of the sequence
        output = output[-1]

        decoded = self.decoder(output.view(-1, output.size(1)))

        return decoded

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (weight.new(self.nlayers, bsz, self.nhid).zero_(),
                weight.new(self.nlayers, bsz, self.nhid).zero_())

def get_batch(source, i, seq_length):
    seq_len = min(seq_length, len(source) - 1 - i)
    
    data = source[i:i+seq_len]
    target = source[i+seq_len].view(-1)

    return data, target

def evaluate_perplexity(data_source, seq_length, exclude_oov=False):
    model.eval()

    total_loss = 0
    ntokens = len(corpus.dictionary)
    len_data = 0
    unk_idx = corpus.dictionary.word2idx["<unk>"]

    if args.cuda:
        torch_range =  torch.cuda.LongTensor()
    else:
        torch_range = torch.LongTensor()
    
    print('data_source', data_source.size(0) - 1)
    
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

def train(seq_length):
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

logging.info("Loading data")
corpus = Corpus('data/')

ntokens = len(corpus.dictionary)
logging.info("Vocab size %d", ntokens)

logging.info("Batchying..")
eval_batch_size = 256

train_data = batchify(corpus.train, batch_size, args.cuda)
# logging.info("Train data size", train_data.size())
val_data = batchify(corpus.valid, eval_batch_size, args.cuda)
test_data = batchify(corpus.test, eval_batch_size, args.cuda)

print(val_data.size(0))

logging.info("Building the model")

model = RNNModel(ntokens, emsize, nhid, nlayers, dropout)
if args.cuda:
    model.cuda()

# Loop over epochs.
best_val_loss = None

try:
    for epoch in range(1, epochs+1):
        epoch_start_time = time.time()

        train(seq_length)

        val_loss = evaluate_perplexity(val_data, seq_length)
        logging.info('-' * 89)
        logging.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                            val_loss, math.exp(val_loss)))
        logging.info('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0

except KeyboardInterrupt:
    logging.info('-' * 89)
    logging.info('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb', encoding="utf8") as f:
    model = torch.load(f)

# Run on valid data with OOV excluded
test_loss = evaluate_perplexity(val_data, seq_length, exclude_oov=True)
logging.info('=' * 89)
logging.info('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(test_loss, math.exp(test_loss)))
logging.info('=' * 89)