import nltk
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE

import csv

from data import create_vocab, create_data
from eval import mean_confidence_interval, sent_surp

vocab = create_vocab(2000)


all_surps_all_k_all_n = dict()

for k in range(1, 4):
	all_surps_all_n = dict()
	for n in range(1, 7):
		print(f'----------- new value of n = {n}, k = {k} -----------')
		data = create_data(vocab, 200, k)
		train_data = [y for x in data for y in data[x]]

		train, padded_sents = padded_everygram_pipeline(n, train_data)

		model = MLE(n)
		model.fit(train, padded_sents)

		all_surps = {0 : [], 1 : [], 2 : [], 3 : []}

		for i in data:
			item = data[i]
			for j in all_surps:
				all_surps[j].append(sent_surp(model, item[2*j + 1], n) - sent_surp(model, item[2*j], n))

		all_surps_all_n[n] = [[x, *mean_confidence_interval(all_surps[x])] for x in all_surps]

	all_surps_all_k_all_n[k] = all_surps_all_n

with open('../data/ngram.csv', 'w') as f:
	writer = csv.writer(f, delimiter = '\t')
	writer.writerow(['n', 'k', 'theme', 'recipient', 'pref', 'lower', 'upper'])
	for k in all_surps_all_k_all_n:
		for j in all_surps_all_k_all_n[k]:
			item = all_surps_all_k_all_n[k][j]
			for i, lengths in enumerate([[x, y] for x in ['short', 'long'] for y in ['short', 'long']]):
				writer.writerow([f'n = {j}', f'k = {k}', lengths[0], lengths[1], item[i][1], item[i][2], item[i][3]])