import nltk
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE

import csv

from eval import mean_confidence_interval, sent_surp, surp

all_surps_all_k_all_n = dict()

for k in range(1, 5):
	print(f'----------- new value of k = {k} -----------')
	all_surps_all_n = dict()
	with open(f'../data/{k}/train.txt', 'r') as f:
		train_sents = f.readlines()
		train_data = [s.split(' ')[:-1] for s in train_sents]
		test_data = [s.split(' ')[:-1] for s in train_sents]
	
	"""with open(f'../data/{k}/train.txt', 'r') as f:
		train_sents = f.readlines()
		test_data = [s.split(' ')[:-1] for s in test_sents]"""
	
	for n in range(1, 9):
		print(f'----------- new value of n = {n} -----------')

		train, padded_sents = padded_everygram_pipeline(n, train_data)

		model = MLE(n)
		model.fit(train, padded_sents)

		all_surps = {0 : [], 1 : [], 2 : [], 3 : []}

		j = 0
		for i in range(0, len(test_data), 8):
			for j in range(0, 4):
				interaction = sent_surp(model, test_data[i + 2 * j + 1], n) - sent_surp(model, test_data[i + 2 * j], n)
				if interaction != interaction:
					print(test_data[i + 2 * j + 1], test_data[i + 2 * j])
					print(sent_surp(model, test_data[i + 2 * j + 1], n))
					print(sent_surp(model, test_data[i + 2 * j], n))
					for w_index, w in enumerate(test_data[i + 2 * j + 1]):
						print(f'Surprisal of {w} = {surp(model, w_index, test_data[i + 2 * j + 1], n)}')
					raise ValueError(f'Probability of items {i + 2 * j + 1} and {i + 2 * j} are 0! Check your model.')
				all_surps[j].append(sent_surp(model, test_data[i + 2 * j + 1], n) - sent_surp(model, test_data[i + 2 * j], n))

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