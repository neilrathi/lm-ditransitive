import numpy as np
import scipy.stats

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def surp(model, w_index, s, n):
    if w_index < n:
        return -model.logscore(s[w_index], s[0:w_index])
    else:
        return -model.logscore(s[w_index], s[(w_index-n+1):w_index])

def sent_surp(model, s, n):
    total_surp = 0
    for i, w in enumerate(s):
        total_surp += surp(model, i, s, n)
    return total_surp