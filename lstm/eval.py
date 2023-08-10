import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import repackage_hidden, batchify, get_batch
from dictionary_corpus import Dictionary, tokenize


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

        return decoded, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (weight.new(self.nlayers, bsz, self.nhid).zero_(),
                weight.new(self.nlayers, bsz, self.nhid).zero_())

def create_target_mask(test_file):
    sents = open(test_file, "r").readlines()
    targets = []
    for sent in sents:
        t_s = [0] * len(sent.split(' '))
        if len(t_s) > 1:
            t_s[2] = 1
        else:
            t_s[0] = 1
        targets.extend(t_s)
    """for sent in sents:
        for i in range(len(sent.split(' '))):
            t_s = [0] * len(sent.split(' '))
            t_s[i] = 1
            targets.extend(t_s)"""
    return np.array(targets)

def evaluate(data_source, mask):
    model.eval()
    #total_loss = 0

    hidden = model.init_hidden(eval_batch_size)

    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, seq_len):
            # keep continuous hidden state across all sentences in the input file
            data, targets = get_batch(data_source, i, seq_len)
            _, targets_mask = get_batch(mask, i, seq_len)
            print('data:', data)
            print('targets:', targets)
            print('targets_mask:', targets_mask)
            output, hidden = model(data, hidden)
            print('output:', output)
            output_flat = output.view(-1, vocab_size)
            # total_loss += len(data) * nn.CrossEntropyLoss()(output_flat, targets)

            output_candidates_probs(output_flat, targets, targets_mask)

            hidden = repackage_hidden(hidden)

    #return total_loss.item() / (len(data_source) - 1)

def output_candidates_probs(output_flat, targets, mask):
    log_probs = F.log_softmax(output_flat, dim=1)

    log_probs_np = log_probs.cpu().numpy()
    subset = mask.cpu().numpy().astype(bool)
    # subset = np.array([1 if i == 2 else 0 for i, j in enumerate(log_probs_np[0])]).astype(bool)

    print(log_probs_np)
    print(subset)

    print('Scores:')

    for scores, correct_label in zip(log_probs_np[subset], targets.cpu().numpy()[subset]):
        #print(idx2word[correct_label], scores[correct_label])
        print("\t".join(str(s) for s in scores) + "\n")

torch.manual_seed(1111)

with open('model.pt', 'rb') as f:
    model = torch.load(f, map_location = lambda storage, loc: storage)

model.eval()

model.cpu()
eval_batch_size = 1
seq_len = 5

dictionary = Dictionary('data')
vocab_size = len(dictionary)

mask = create_target_mask('data/test.txt')
print('Mask:', mask)
mask_data = batchify(torch.LongTensor(mask), eval_batch_size, False)
test_data = batchify(tokenize(dictionary, "data/test.txt"), eval_batch_size, False)

print('mask_data:', mask_data)
print('test_data:', test_data)

evaluate(test_data, mask_data)
