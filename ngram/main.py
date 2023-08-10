import nltk
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE

import csv

from eval import mean_confidence_interval, sent_surp

all_surps_all_k_all_n = dict()

for k in range(1, 5):
	all_surps_all_n = dict()
	with open(f'../data/{k}/train.txt', 'r') as f:
		train_sents = f.readlines()
		train_data = [s.split(' ')[:-1] for s in train_sents]
		train_data = train_data[:-1]
	with open(f'../data/{k}/test.txt', 'r') as f:
		train_sents = f.readlines()
		train_data = [s.split(' ')[:-1] for s in train_sents]
		train_data = train_data[:-1]
	
	for n in range(1, 7):
		print(f'----------- new value of n = {n}, k = {k} -----------')

		train, padded_sents = padded_everygram_pipeline(n, train_data)

		model = MLE(n)
		model.fit(train, padded_sents)

		all_surps = {0 : [], 1 : [], 2 : [], 3 : []}

		j = 0
		for i in range(0, len(train_data), 8):
			for j in range(0, 4):
				all_surps[j].append(sent_surp(model, train_data[i + 2 * j + 1], n) - sent_surp(model, train_data[i + 2 * j], n))

		all_surps_all_n[n] = [[x, *mean_confidence_interval(all_surps[x])] for x in all_surps]

	all_surps_all_k_all_n[k] = all_surps_all_n

with open('../analysis/ngram.csv', 'w') as f:
	writer = csv.writer(f, delimiter = '\t')
	writer.writerow(['n', 'k', 'theme', 'recipient', 'pref', 'lower', 'upper'])
	for k in all_surps_all_k_all_n:
		for j in all_surps_all_k_all_n[k]:
			item = all_surps_all_k_all_n[k][j]
			for i, lengths in enumerate([[x, y] for x in ['short', 'long'] for y in ['short', 'long']]):
				writer.writerow([f'n = {j}', f'k = {k}', lengths[0], lengths[1], item[i][1], item[i][2], item[i][3]])