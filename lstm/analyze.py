import csv
import scipy.stats
import numpy as np

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

all_surps_all_k_all_n = dict()

for k in range(1, 5):
    all_surps_all_n = dict()
    
    for n in range(1, 9):
        print(f'----------- new value of n = {n}, k = {k} -----------')

        data = []
        
        with open(f'../data/{k}/output/test-{n}.txt', 'r') as f:
            reader = csv.reader(f, delimiter = '\t')
            for line in reader:
                if line[0] != '<unk>' and line[0] != '':
                    data.append(float(line[1]))

        data = data[:(len(data) // 8 * 8)]

        all_surps = {0 : [], 1 : [], 2 : [], 3 : []}

        j = 0
        for i in range(0, len(data), 8):
            for j in range(0, 4):
                all_surps[j].append(data[i + 2 * j + 1] - data[i + 2 * j])

        all_surps_all_n[n] = [[x, *mean_confidence_interval(all_surps[x])] for x in all_surps]

    all_surps_all_k_all_n[k] = all_surps_all_n

with open('../analysis/lstm.csv', 'w') as f:
    writer = csv.writer(f, delimiter = '\t')
    writer.writerow(['n', 'k', 'theme', 'recipient', 'pref', 'lower', 'upper'])
    for k in all_surps_all_k_all_n:
        for j in all_surps_all_k_all_n[k]:
            item = all_surps_all_k_all_n[k][j]
            for i, lengths in enumerate([[x, y] for x in ['short', 'long'] for y in ['short', 'long']]):
                writer.writerow([f'n = {j}', f'k = {k}', lengths[0], lengths[1], item[i][1], item[i][2], item[i][3]])