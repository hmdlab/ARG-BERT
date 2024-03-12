import numpy as np
import pandas as pd
import argparse

import tensorflow as tf
from tensorflow import keras

from proteinbert.shared_utils.util import log
from _new_tokenization import ADDED_TOKENS_PER_SEQ
from _new_tokenization import encode_dataset,split_dataset_by_len

class Config_Test:

    # fold,use_LHD(bool),threshold,gpuを使う
    # ,n_UNIQUE_LABELS,CUDA関連
    # BENCHMARKS_DIR,BENCHMARK_NAME:__str__
    # n_UNIQUE_LABELS:__len__
    
    def __init__(self, args):
        
        self.mechanism_labels = {
            
        'antibiotic target alteration':0,
        'antibiotic target replacement':1,
        'antibiotic target protection':2,
        'antibiotic inactivation':3,
        'antibiotic efflux':4,
        'others':5
            
        }
        self.fold = args.fold
        self.use_LHD = args.use_LHD
        self.threshold = args.threshold
        self.seed = args.seed
        self.get_all_attention = args.get_all_attention
        #self.create_dataset_path = create_dataset_path
        
    def create_input_path(self, create_dataset_path):

        if self.use_LHD:
            sub_path = 'LHD/c%1f/fold_%d_%1f'%(self.fold, self.threshold)
        else:
            sub_path = 'HMDARG-DB/fold_%d'%self.fold

        if create_dataset_path:
            root_dir = 'dataset'
            return os.path.join(root_dir, '%s.test.csv' % sub_path)
        else:
            root_dir = 'finetuned_model'
            return os.path.join(root_dir, '%s.finetuned_model.h5' % sub_path)
    
    def create_output_path(self, create_dataset_path):
        
        if self.use_LHD:
            sub_path = 'LHD/c%1f/fold_%d_%1f'%(self.fold, self.threshold)
        else:
            sub_path = 'HMDARG-DB/fold_%d'%self.fold

        if create_dataset_path:
            root_dir = 'result'
            return os.path.join(root_dir, '%s.test.csv' % sub_path)
        else:
            root_dir = 'attention'
            return os.path.join(root_dir, '%s.attn.csv' % sub_path)
        
    def n_mechanism_labels(self):
        return len(self.mechanism_labels)

    def set_seed(self):
        if self.seed != None:
            return tf.random.set_seed(self.seed)

def evaluate_by_len(model_generator, input_encoder, config,  df, start_seq_len = 512, start_batch_size = 32, increase_factor = 2):#output_spec,
    
    assert model_generator.optimizer_weights is None
    
    #dataset = pd.DataFrame({'seq': df['sequence'], 'raw_y':df['mechanism']})
        
    results = []
    results_names = []
    y_trues = []
    y_preds = []
    index_list = []
    #inverse_UNIQUE_LABELS = {v:k for k,v in output_spec.unique_labels.items()}
    index_to_label = {}
    #mutation_map = pd.DataFrame(columns = list(range(1392),index = list(range(20)))# 変異調査
    
    for len_matching_dataset, seq_len, batch_size, index in split_dataset_by_len(df, start_seq_len = start_seq_len, start_batch_size = start_batch_size, \
            increase_factor = increase_factor):

        X, y_true, sample_weights = encode_dataset(len_matching_dataset, input_encoder, \
                seq_len = seq_len, needs_filtering = False)#output_spec,
        #print(seq_len)
        #print(y_pred)
        
        assert set(np.unique(sample_weights)) <= {0.0, 1.0}
        y_mask = (sample_weights == 1)
        
        #model = model_generator.create_model(seq_len)
        model = keras.models.load_model(config.create_input_path(create_dataset_path = False))
        y_pred = model.predict(X, batch_size = batch_size)
        
        y_true = y_true[y_mask].flatten()
        #print(y_pred)
        #print(y_mask)
        y_pred = y_pred[y_mask]
        
        #if output_spec.output_type.is_categorical:
        y_pred = y_pred.reshape((-1, y_pred.shape[-1]))
        #else:
            #y_pred = y_pred.flatten()
        
        #result, y_pred_classes, propotion = get_evaluation_results(y_true, y_pred, output_spec)
        #results.append(get_evaluation_results(y_true, y_pred, output_spec)[0])
        #results_names.append(seq_len)
        
        y_trues.append(y_true)
        y_preds.append(y_pred)
        index_list += index
        
    y_true = np.concatenate(y_trues, axis = 0)
    y_pred = np.concatenate(y_preds, axis = 0)
    prediction, confusion_matrix = get_evaluation_results(y_true, y_pred, config, return_confusion_matrix = True)#output_spec,
    
    #results.append(all_results)
    #results_names.append('All')
    
    #results = pd.DataFrame(results, index = results_names)
    #results.index.name = 'Model seq len'
    """
    df['mechanism_preds'] = 0
    for i,i_d in enumerate(index_list):
        df.loc[i_d,'mechanism_preds'] = inverse_UNIQUE_LABELS[mechanism_preds[i]]
    """
    for i in list(range(len(df))):
        index_to_label[index_list[i]] = prediction[i]
    df = pd.concat([df,pd.DataFrame.from_dict(index_to_label, orient='index').sort_index()],axis = 1)
        
    return df, confusion_matrix

def get_evaluation_results(y_true, y_pred, config, return_confusion_matrix = False):#output_spec,

    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
            
    #results = {}
    #results['# records'] = len(y_true)
            
    #if output_spec.output_type.is_numeric:
        #results['Spearman\'s rank correlation'] = spearmanr(y_true, y_pred)[0]
        #confusion_matrix = None
    #else:
    
    str_unique_labels = list(map(str, config.unique_labels))
    """
        #if output_spec.output_type.is_binary:
            
            #y_pred_classes = (y_pred >= 0.5)
            
            if len(np.unique(y_true)) == 2:
                results['AUC'] = roc_auc_score(y_true, y_pred)
            else:
                results['AUC'] = np.nan
          
        #elif output_spec.output_type.is_categorical:
    """ 
    y_pred_classes = y_pred.argmax(axis = -1)
            #results['Accuracy'] = accuracy_score(y_true, y_pred_classes)
        #else:
            #raise ValueError('Unexpected output type: %s' % output_spec.output_type)
                    
    confusion_matrix = pd.DataFrame(confusion_matrix(y_true, y_pred_classes, labels = np.arange(config.n_mechanism_labels)), index = str_unique_labels, \
                    columns = str_unique_labels)
         
    if return_confusion_matrix:
        return y_pred_classes, confusion_matrix
    else:
        return results
        
def calculate_attentions(model, input_encoder, seq, seq_len = None):
    
    from tensorflow.keras import backend as K
    from ._new_tokenization import index_to_token
    
    if seq_len is None:
        seq_len = len(seq) + 2
    
    X = input_encoder.encode_X([seq], seq_len)
    (X_seq,), _ = X
    seq_tokens = list(map(index_to_token.get, X_seq))

    model_inputs = [layer.input for layer in model.layers if 'InputLayer' in str(type(layer))][::-1]
    model_attentions = [layer.calculate_attention(layer.input) for layer in model.layers if 'GlobalAttention' in str(type(layer))]
    invoke_model_attentions = K.function(model_inputs, model_attentions)
    attention_values = invoke_model_attentions(X)
    
    attention_labels = []
    merged_attention_values = []

    for attention_layer_index, attention_layer_values in enumerate(attention_values):
        for head_index, head_values in enumerate(attention_layer_values):
            attention_labels.append('Attention %d - head %d' % (attention_layer_index + 1, head_index + 1))
            merged_attention_values.append(head_values)

    attention_values = np.array(merged_attention_values)
    
    return attention_values, seq_tokens, attention_labels

def main(config):
    
    df, confusion_matrix = evaluate_by_len(model_generator, input_encoder, config, test_set, \
            start_seq_len = 512, start_batch_size = 32)
    df = df.replace({v: k for k, v in config.mechanism_labels.items()})
    df.to_csv(config.create_output_path(create_dataset_path = True)
    
    if config.get_all_attention:
        attention_fold = {}
        for i in test_set.index:
            ID = test_set.iloc[i,0]
            seq = test_set.iloc[i,-1]
            seq_len = len(seq) + 2
            
            pretrained_model_generator, input_encoder = load_pretrained_model()
            pretrained_model = pretrained_model_generator.create_model(seq_len)
            pretrained_attention_values, pretrained_seq_tokens, pretrained_attention_labels = calculate_attentions(pretrained_model, input_encoder, seq, \
                    seq_len = seq_len)
            
            finetuned_model = keras.models.load_model(config.create_input_path(create_dataset_path = False))
            finetuned_attention_values, finetuned_seq_tokens, finetuned_attention_labels = calculate_attentions(finetuned_model, input_encoder, seq,\
                    seq_len = seq_len)
    
            attention = finetuned_attention_values - pretrained_attention_values
            attention_fold[ID] =  attention[-4:,:].mean(axis = 0)
            
        attention_fold = pd.DataFrame.from_dict(attention_fold, orient='index').T
        attention_fold.to_csv(config.create_output_path(create_dataset_path = False))

if __name__ == "__main__": 

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fold', type=int, help='The number of iterations in 5-fold CV.')
    #parser.add_argument('--dir', type=str, help='Path to the dataset.')
    parser.add_argument('-LHD', '--use_LHD', action='store_true', help='Whether you use Low Homology Dataset or not.')
    parser.add_argument('-t', '--threshold', type=float, help='Sequence similarity thresholds set when creating LHD.', default='')
    parser.add_argument('-s', '--seed', type=int, help='Set random seed.', default=None)
    parser.add_argument('-attn', '--get_all_attention', type=int, help='Whether you need attention or not.', default=False)
    config = Config_Test(parser.parse_args())
    
    main(config)
