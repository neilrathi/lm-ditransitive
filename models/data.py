import random

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

def create_data(vocab, size, num_adj):
    data = []
    while len(data) < (8 * size):
        theme_noun = random.choice(vocab['N'])
        theme_adj = random.sample(vocab['A'], num_adj)

        recipient_noun = theme_noun 
        recipient_adj = theme_adj

        while recipient_noun == theme_noun:
            recipient_noun = random.choice(vocab['N'])

        while recipient_adj == theme_adj:
            recipient_adj = random.sample(vocab['A'], num_adj)

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

    data_dict = {i : [] for i in range(0, size)}
    j = -1
    for i, x in enumerate(data):
        if i % 8 == 0:
            j += 1
        data_dict[j].append(x.split())
    return data_dict
