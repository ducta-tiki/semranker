from vn_lang import query_preprocessing
import numpy as np


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

    return np.asarray(unigram_indices, dtype=np.int32), \
           np.asarray(bigram_indices, dtype=np.int32), \
           np.asarray(char_trigram_indices, dtype=np.int32)


def convert_cats(
    arr_cats, token_2_idx, cat_2_idx, 
    token_zero_idx, cat_zero_idx, unknown_map_func,
    unigram_max_seq_len, bigram_max_seq_len, char_trigram_max_seq_len):

    cat_indices = []
    cat_in_product = []

    unigram_indices = []
    bigram_indices = []
    char_trigram_indices = []

    for cat_str in arr_cats:
        zz = cat_str.split("|")
        if len(cat_str.strip()) == 0:
            cat_in_product.append(1)
            cat_indices.append(cat_zero_idx)
            unigram_indices.append([token_zero_idx,] * unigram_max_seq_len)
            bigram_indices.append([token_zero_idx,] * bigram_max_seq_len)
            char_trigram_indices.append([token_zero_idx,] * char_trigram_max_seq_len)
            continue

        count = 0
        for t in zz:
            cat_token = "#".join(t.split("#")[:2])
            if cat_token in cat_2_idx:
                count += 1
                cat_indices.append(cat_2_idx[cat_token])
                cat_name = query_preprocessing(t.split("#")[-1])
                ui, bi, ci = convert_strings(
                    [cat_name], token_2_idx, token_zero_idx, 
                    unigram_max_seq_len, bigram_max_seq_len, char_trigram_max_seq_len, 
                    unknown_map_func)
                unigram_indices.append(ui[0])
                bigram_indices.append(bi[0])
                char_trigram_indices.append(ci[0])
        cat_in_product.append(count)
    
    return np.asarray(cat_indices, dtype=np.int32),\
        np.asarray(cat_in_product, dtype=np.int32),\
        np.asarray(unigram_indices, dtype=np.int32),\
        np.asarray(bigram_indices, dtype=np.int32),\
        np.asarray(char_trigram_indices, dtype=np.int32)


def convert_attrs(
    arr_attrs, token_2_idx, attr_2_idx, 
    token_zero_idx, attr_zero_idx, unknown_map_func,
    unigram_max_seq_len, bigram_max_seq_len, char_trigram_max_seq_len):

    attr_indices = []
    attr_in_product = []

    unigram_indices = []
    bigram_indices = []
    char_trigram_indices = []

    for attr_str in arr_attrs:
        zz = attr_str.split("|")

        if len(attr_str.strip()) == 0:
            attr_in_product.append(1)
            attr_indices.append(attr_zero_idx)
            unigram_indices.append([token_zero_idx,] * unigram_max_seq_len)
            bigram_indices.append([token_zero_idx,] * bigram_max_seq_len)
            char_trigram_indices.append([token_zero_idx,] * char_trigram_max_seq_len)
            continue

        count = 0
        for t in zz:
            attr_token = "#".join(t.split("#")[:2])
            if attr_token in attr_2_idx:
                count += 1
                attr_indices.append(attr_2_idx[attr_token])
                attr_name = query_preprocessing(t.split("#")[-1])
                ui, bi, ci = convert_strings(
                    [attr_name], token_2_idx, token_zero_idx, 
                    unigram_max_seq_len, bigram_max_seq_len, char_trigram_max_seq_len, 
                    unknown_map_func)
                unigram_indices.append(ui[0])
                bigram_indices.append(bi[0])
                char_trigram_indices.append(ci[0])
        attr_in_product.append(count)
    
    return np.asarray(attr_indices, dtype=np.int32),\
        np.asarray(attr_in_product, dtype=np.int32),\
        np.asarray(unigram_indices, dtype=np.int32),\
        np.asarray(bigram_indices, dtype=np.int32),\
        np.asarray(char_trigram_indices, dtype=np.int32)


def convert_features(features, precomputed_min, precomputed_max):
    """
    Convert raw features to normalized features
    :param features: batch_size x num_of_features
    :param precomputed_min: (num_of_features,)
    :param precompute_max: (num_of_features,)
    :return: normalize features
    """

    precomputed_min = np.expand_dims(
        np.asarray(precomputed_min, dtype=np.float32), axis=0)
    precomputed_max = np.expand_dims(
        np.asarray(precomputed_max, dtype=np.float32), axis=0)
    features = np.asarray(features, dtype=np.float32)
    
    features = (features - precomputed_min) / precomputed_max

    return features