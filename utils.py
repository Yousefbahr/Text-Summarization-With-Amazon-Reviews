import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import random

def load_glove_embeddings(glove_file, max_embeddings=20000, important_words=None):
    """
    Load GloVe embeddings prioritizing important words and most frequent words.
    
    Args:
        glove_file: Path to the GloVe embeddings file
        max_embeddings: Maximum number of embeddings to load
        important_words: Set of words that must be included
    
    Returns:
        words_to_index: Dictionary mapping words to indices
        index_to_words: Dictionary mapping indices to words
        word_to_vec_map: Dictionary mapping words to embeddings
    """
    # Initialize important words set if not provided
    if important_words is None:
        important_words = set()
    else:
        important_words = set(important_words)
    
    # First collect all embeddings
    all_embeddings = []
    important_embeddings = []
    
    print("Reading embeddings file...")
    with open(glove_file, 'r') as f:
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            embedding = np.array(line[1:], dtype=np.float64)
            
            if curr_word in important_words:
                important_embeddings.append((curr_word, embedding))
            else:
                all_embeddings.append((curr_word, embedding))
    
    # Calculate how many regular embeddings we can take
    remaining_capacity = max_embeddings - len(important_embeddings)
    
    # Take only as many as we have capacity for
    if remaining_capacity > 0:
        selected_embeddings = important_embeddings + all_embeddings[:remaining_capacity]
    else:
        selected_embeddings = important_embeddings[:max_embeddings]
    
    # Create the word sets and mappings
    words = set()
    word_to_vec_map = {}
    for word, embedding in selected_embeddings:
        words.add(word)
        word_to_vec_map[word] = embedding
    
    # Build index mappings
    i = 1
    words_to_index = {}
    index_to_words = {}
    for w in sorted(words):
        words_to_index[w] = i
        index_to_words[i] = w
        i = i + 1

    return words_to_index, index_to_words, word_to_vec_map
    

def pre_process(x, Tx, word_to_index, add_eos= False):
    """
    pre-process data and return X of shape (m, Tx) padded and contain indices
    x: shape (m,) -> m samples with a string of text for each sample
    Tx: longest sequence in all samples
    word_to_index: dict mapping from word to index from the pretrained embeddings
    add_eos: bool indicating adding EOS token at the end of target sequence and before padding , used with target sequence
    """
    # from sentences to words
    all_words = np.empty((x.shape[0], Tx), dtype='<U15')
    for i, sentence in enumerate(x):
        words = sentence.split()
        for j, word in enumerate(words):
            # sequences less than or equal to Tx
            if j >= Tx:
                break

            all_words[i, j] = word

    # remove punctuation and 'br' tags
    for i, sentence in enumerate(all_words):
        for j, word in enumerate(sentence):
            w = re.sub(r"(?:[^\w]|<br/?\s*|<br)+", "", word)
            all_words[i, j] = re.sub(r"^br|br$", "", w)


    # words to indices
    X = np.zeros((x.shape[0], Tx))
    for i, sentence in enumerate(all_words):
        for j, word in enumerate(sentence):

            if word.lower() in word_to_index:
                X[i, j] = word_to_index[word.lower()]
            
            else:
            	# insert unknown word token
            	X[i, j] = word_to_index["UNK"]

    # shifts all padding to the right only
    for i in range(X.shape[0]):
        non_zero = X[i][X[i] != 0]
        num_zeros = Tx - len(non_zero)
        X[i] = np.concatenate([non_zero, np.zeros(num_zeros, dtype=X.dtype)])
        if add_eos:
            first_zero_index = np.where(X[i] == 0)[0]
            # there is zero -> there is padding
            
            # substitute first zero (padding) with an EOS token
            if len(first_zero_index) != 0:
            
            	X[i, first_zero_index[0]] = word_to_index["EOS"]
    
    
    if add_eos:
        for line in range(X.shape[0]):
            X[line, :] =  np.insert(X[line, :], 0, word_to_index["SOS"])[:-1]
            first_zero_index = np.where(X[line, :] == 0)[0]
            if len(first_zero_index) == 0:                
                X[line, -1] = word_to_index["EOS"]

    return X
    
    
    
    
