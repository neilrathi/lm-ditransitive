import random
import csv

def create_vocab(size):
    vocab = {x : [] for x in ['N', 'A']}
    targets = {x[0] : int(size * x[1]) for x in zip(['N', 'A'], [0.8, 0.2])}
    unique_words = set()

    while len(unique_words) < size:
        unique_words.add(''.join(random.choices(list('abcdefghijklmnopqrstuvwxyz'), k = 4)))

    for word in list(unique_words):
        if len(vocab['N']) < targets['N']:
            vocab['N'].append(word)
        elif len(vocab['A']) < targets['A']:
            vocab['A'].append(word)
    
    return vocab

def create_sents(theme_noun, theme_adj, recipient_noun, recipient_adj):
    data = []
    for lengths in [[x, y] for x in ['short', 'long'] for y in ['short', 'long']]:
        for condition in ['double', 'prep']:
            if lengths[0] == 'short':
                theme = theme_noun
            else:
                theme = f'{" ".join(theme_adj)} {theme_noun}'

            if lengths[1] == 'short':
                recipient = recipient_noun
            else:
                recipient = f'{" ".join(recipient_adj)} {recipient_noun}'

            if condition == 'double':
                data.append(f'the a gave the {recipient} the {theme}')
            else:
                data.append(f'the a gave the {theme} to the {recipient}')
    
    return data

def create_data(vocab, size, num_adj):
    if size % 5 != 0:
        raise ValueError('Size must be a multiple of 5!')

    data = []
    train_vocab = {
        'N' : [],
        'A' : []
    }
    while len(data) < (8 * size * 0.8):
        theme_noun = random.choice(vocab['N'])
        theme_adj = random.sample(vocab['A'], num_adj)

        recipient_noun = theme_noun 
        recipient_adj = theme_adj

        while recipient_noun == theme_noun:
            recipient_noun = random.choice(vocab['N'])

        while recipient_adj == theme_adj:
            recipient_adj = random.sample(vocab['A'], num_adj)
        
        train_vocab['N'].append(theme_noun)
        train_vocab['N'].append(recipient_noun)
        train_vocab['A'] += theme_adj + recipient_adj

        data += create_sents(theme_noun, theme_adj, recipient_noun, recipient_adj)

    with open(f'{num_adj}/train.txt', 'w') as f:
        for s in data:
            f.write(s + ' <eos>\n')
    
    test_data = []
    
    while len(test_data) < (8 * size * 0.2):
        theme_noun = random.choice(train_vocab['N'])
        theme_adj = random.sample(train_vocab['A'], num_adj)

        recipient_noun = theme_noun 
        recipient_adj = theme_adj

        while recipient_noun == theme_noun:
            recipient_noun = random.choice(train_vocab['N'])

        while recipient_adj == theme_adj:
            recipient_adj = random.sample(train_vocab['A'], num_adj)
        
        if f'the a gave the {recipient_noun} the {theme_noun}' in data:
            continue
            
        test_data += create_sents(theme_noun, theme_adj, recipient_noun, recipient_adj)
    
    with open(f'{num_adj}/valid.txt', 'w') as f:
        for s in test_data[0:int(8 * 0.1 * size)]:
            f.write(s + ' <eos>\n')
    with open(f'{num_adj}/test.txt', 'w') as f:
        for s in test_data[int(8 * 0.1 * size):]:
            f.write(s + ' <eos>\n')
    
    return ['the', 'a', 'gave', 'to', '<unk>', '<eos>'] + [word for words in train_vocab.values() for word in words]

vocab = create_vocab(50000)
for num_adj in range(1, 5):
    print(num_adj)
    tot_vocab = create_data(vocab, 10000, num_adj)

    with open(f'{num_adj}/test.txt', 'r') as f:
        lines = f.readlines()[:-1]
    with open(f'{num_adj}/test.gold', 'w') as f:
        for line in lines:
            i = 0
            for word in line.split(' '):
                f.write(str(i) + '\n')
                i += 1
    with open(f'{num_adj}/test-clean.txt', 'w') as f:
        for line in lines:
            for word in line.split(' '):
                f.write(line)
    with open(f'{num_adj}/vocab.txt', 'w') as f:
        for w in tot_vocab:
            f.write(w + '\n')