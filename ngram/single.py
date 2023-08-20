import nltk
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE

import csv

from eval import mean_confidence_interval, sent_surp, surp

with open(f'../data/2/train.txt', 'r') as f:
	train_sents = f.readlines()
	train_data = [s.split(' ')[:-1] for s in train_sents[:8]]
	test_data = [s.split(' ')[:-1] for s in train_sents[:8]]

train, padded_sents = padded_everygram_pipeline(8, train_data)

model = MLE(8)
model.fit(train, padded_sents)

a = []
b = []

for i, w in enumerate(train_data[2]):
    a.append(surp(model, i, train_data[4], 5))
    b.append(surp(model, i, train_data[5], 5))

for i, x in enumerate(zip(a, b)):
    print(train_data[4][i], '\t', x[0], '\t\t', train_data[5][i], '\t', x[1])