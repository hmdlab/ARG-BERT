import numpy as np
import pandas as pd

from proteinbert.shared_utils.util import log

ALL_AAS = 'ACDEFGHIKLMNPQRSTUVWXY'
ADDITIONAL_TOKENS = ['<OTHER>', '<START>', '<END>', '<PAD>']
mechanism_labels = {
            
        'antibiotic target alteration':0,
        'antibiotic target replacement':1,
        'antibiotic target protection':2,
        'antibiotic inactivation':3,
        'antibiotic efflux':4,
        'others':5
            
        }

# Each sequence is added <START> and <END> tokens
ADDED_TOKENS_PER_SEQ = 2

n_aas = len(ALL_AAS)
aa_to_token_index = {aa: i for i, aa in enumerate(ALL_AAS)}
additional_token_to_index = {token: i + n_aas for i, token in enumerate(ADDITIONAL_TOKENS)}
token_to_index = {**aa_to_token_index, **additional_token_to_index}
index_to_token = {index: token for token, index in token_to_index.items()}
n_tokens = len(token_to_index)

def tokenize_seq(seq):
    other_token_index = additional_token_to_index['<OTHER>']
    return [additional_token_to_index['<START>']] + [aa_to_token_index.get(aa, other_token_index) for aa in parse_seq(seq)] + \
            [additional_token_to_index['<END>']]
            
def parse_seq(seq):
    if isinstance(seq, str):
        return seq
    elif isinstance(seq, bytes):
        return seq.decode('utf8')
    else:
        raise TypeError('Unexpected sequence type: %s' % type(seq))

def encode_dataset(dataset, input_encoder, mechanism_labels, seq_len = 512, needs_filtering = True, dataset_name = 'Dataset', verbose = True):#seq_len = 512
    
    seqs = dataset['sequence']
    raw_Y = dataset['mechanism']
    
    if needs_filtering:
        dataset = filter_dataset_by_len(dataset, seq_len = seq_len, dataset_name = dataset_name, verbose = verbose)
        seqs = dataset['sequence']
        raw_Y = dataset['mechanism']
    
    X = input_encoder.encode_X(seqs, seq_len)
    Y, sample_weigths = encode_Y(raw_Y, seq_len = seq_len, mechanism_labels = mechanism_labels)
    return X, Y, sample_weigths

def encode_Y(raw_Y, seq_len = 512, mechanism_labels = mechanism_labels):
    return encode_categorical_Y(raw_Y, mechanism_labels), np.ones(len(raw_Y))
    
def encode_seq_Y(seqs, seq_len, is_binary, mechanism_labels):

    label_to_index = {str(label): i for i, label in enumerate(mechanism_labels)}

    Y = np.zeros((len(seqs), seq_len), dtype = int)
    sample_weigths = np.zeros((len(seqs), seq_len))
    
    for i, seq in enumerate(seqs):
        
        for j, label in enumerate(seq):
            # +1 to account for the <START> token at the beginning.
            Y[i, j + 1] = label_to_index[label]
            
        sample_weigths[i, 1:(len(seq) + 1)] = 1
        
    if is_binary:
        Y = np.expand_dims(Y, axis = -1)
        sample_weigths = np.expand_dims(sample_weigths, axis = -1)
    
    return Y, sample_weigths
    
def encode_categorical_Y(labels, mechanism_labels):
    
    label_to_index = {label: i for i, label in enumerate(mechanism_labels)}
    Y = np.zeros(len(labels), dtype = int)
    
    for i, label in enumerate(labels):
        Y[i] = label_to_index[label]
        
    return Y
    
def filter_dataset_by_len(dataset, seq_len = 512, seq_col_name = 'sequence', dataset_name = 'Dataset', verbose = True):
    
    max_allowed_input_seq_len = seq_len - ADDED_TOKENS_PER_SEQ
    filtered_dataset = dataset[dataset[seq_col_name].str.len() <= max_allowed_input_seq_len]
    n_removed_records = len(dataset) - len(filtered_dataset)
    
    if verbose:
        log('%s: Filtered out %d of %d (%.1f%%) records of lengths exceeding %d.' % (dataset_name, n_removed_records, len(dataset), 100 * n_removed_records / len(dataset), \
                max_allowed_input_seq_len))
    
    return filtered_dataset
    
def split_dataset_by_len(dataset, seq_col_name = 'sequence', start_seq_len = 512, start_batch_size = 32, increase_factor = 2):#start_seq_len = 512

    seq_len = start_seq_len
    batch_size = start_batch_size
    
    while len(dataset) > 0:
        max_allowed_input_seq_len = seq_len - ADDED_TOKENS_PER_SEQ
        len_mask = (dataset[seq_col_name].str.len() <= max_allowed_input_seq_len)
        len_matching_dataset = dataset[len_mask]
        yield len_matching_dataset, seq_len, batch_size, len_matching_dataset.index.tolist()
        dataset = dataset[~len_mask]
        seq_len *= increase_factor
        batch_size = max(batch_size // increase_factor, 1)