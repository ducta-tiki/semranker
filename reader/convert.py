

def convert_tokens(arr_tokens, token_2_idx, zero_idx, max_seq_len, unknown_map_func):
    """
    :param tokens: Array of tokens
    :param token_2_idx: Token to index
    :param zero_idx: Padding zero_idx
    :param max_seq_len: Maximum sequence length
    :return: Array of indices
    """
    arr_indices = []
    base_unknown = len(token_2_idx)

    for tokens in arr_tokens:
        z = tokens[:max_seq_len] + ['<zero>',]*(max(max_seq_len, len(tokens)) - len(tokens))
        indices = []
        for t in z:
            if t in token_2_idx:
                indices.append(token_2_idx[t])
            elif t == '<zero>':
                indices.append(zero_idx)
            else:
                indices.append(base_unknown + unknown_map_func(t))
        
        arr_indices.append(indices)
    
    return arr_indices


def create_ngrams(s):
    unigrams = []
    bigrams = []
    char_trigrams = []

    tokens = s.split()
    for t in tokens:
        if len(t):
            unigrams.append(t)
            
            z = "#" + t + "#"
            for i in range(0, max(len(z)-2, 1)):
                v = z[i:i+3]
                char_trigrams.append(v)

    for i in range(0, max(len(tokens) - 1, 0)):
        t = "%s#%s" % (tokens[i], tokens[i+1])
        bigrams.append(t)
    
    return unigrams, bigrams, char_trigrams


def convert_strings(
    arr_strings, token_2_idx, zero_idx, 
    unigram_max_seq_len, bigram_max_seq_len, char_trigram_max_seq_len,
    unknown_map_func):
    
    arr_tokens = []
    unigram_tokens = []
    bigram_tokens = []
    char_trigram_tokens = []

    for s in arr_strings:
        unigrams, bigrams, char_trigrams = create_ngrams(s)
        unigram_tokens.append(unigrams)
        bigram_tokens.append(bigrams)
        char_trigram_tokens.append(char_trigrams)
    
    unigram_indices = convert_tokens(
        unigram_tokens, token_2_idx, zero_idx, unigram_max_seq_len, unknown_map_func)
    bigram_indices = convert_tokens(
        bigram_tokens, token_2_idx, zero_idx, bigram_max_seq_len, unknown_map_func)
    char_trigram_indices = convert_tokens(
        char_trigram_tokens, token_2_idx, zero_idx, char_trigram_max_seq_len, unknown_map_func)

    return unigram_indices, bigram_indices, char_trigram_indices
