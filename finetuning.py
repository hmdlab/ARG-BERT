import numpy as np
import pandas as pd
import os
import argparse
import pickle
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

from proteinbert.existing_model_loading import load_pretrained_model
from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs
from proteinbert.shared_utils.util import log
from proteinbert.model_generation import FinetuningModelGenerator
from tokenization import index_to_token, ADDED_TOKENS_PER_SEQ, mechanism_labels, encode_dataset
            
class Config_Finetuning:

    def __init__(self, args):
        
        
        self.mechanism_labels = mechanism_labels
        self.fold = args.fold
        self.use_LHD = args.use_LHD
        self.threshold = args.threshold
        self.gpu = args.gpu
        self.seed = args.seed
        
    def create_dataset_or_model_path(self, create_dataset_path):

        if self.use_LHD:
            sub_path = 'LHD/c%1f/fold_%d_%1f'%(self.fold, self.threshold)
        else:
            sub_path = 'HMDARG-DB/fold_%d'%self.fold

        if create_dataset_path:
            root_dir = 'inputs'
            return os.path.join(root_dir, '%s.train.csv' % sub_path)
        else:
            root_dir = 'outputs/finetuned_model'
            return os.path.join(root_dir,'%s.finetuned_model.h5' % sub_path)

    def set_gpu(self):

        if self.gpu != None:
            config = tf.compat.v1.ConfigProto(
            gpu_options=tf.compat.v1.GPUOptions(
                visible_device_list= str(self.gpu), # specify GPU number
                allow_growth=True,
                per_process_gpu_memory_fraction=0.2
                )
            )
            sess = tf.compat.v1.Session(config=config)

    
    def set_seed(self):
        if self.seed != None:
            return tf.random.set_seed(self.seed)
    
            
def finetune(model_generator, input_encoder, config, train, valid = None, seq_len = 512, batch_size = 32, \
        max_epochs_per_stage = 40, lr = None, begin_with_frozen_pretrained_layers = True, lr_with_frozen_pretrained_layers = None, n_final_epochs = 1, \
        final_seq_len = 1024, final_lr = None, callbacks = []):
    print(seq_len)
        
    encoded_train_set, encoded_valid_set = encode_train_and_valid_sets(train, valid, input_encoder, config, seq_len)
        
    if begin_with_frozen_pretrained_layers:
        log('Training with frozen pretrained layers...')
        model_generator.train(encoded_train_set, encoded_valid_set, seq_len, batch_size, max_epochs_per_stage, lr = lr_with_frozen_pretrained_layers, \
                callbacks = callbacks, freeze_pretrained_layers = True)
     
    log('Training the entire fine-tuned model...')
    model_generator.train(encoded_train_set, encoded_valid_set, seq_len, batch_size, max_epochs_per_stage, lr = lr, callbacks = callbacks, \
            freeze_pretrained_layers = False)
                
    if n_final_epochs > 0:
        log('Training on final epochs of sequence length %d...' % final_seq_len)
        final_batch_size = max(int(batch_size / (final_seq_len / seq_len)), 1)
        encoded_train_set, encoded_valid_set = encode_train_and_valid_sets(train, valid, input_encoder, config, final_seq_len)
        model_generator.train(encoded_train_set, encoded_valid_set, final_seq_len, final_batch_size, n_final_epochs, lr = final_lr, callbacks = callbacks, \
                freeze_pretrained_layers = False)
                
    model_generator.optimizer_weights = None

def make_train_and_valid_sets(path):
    
    train_set = pd.read_csv(path, index_col = 0)#.dropna()#.drop_duplicates()
    train_set, valid_set = train_test_split(train_set, test_size = 0.1, random_state = 1)

    print(f'{len(train_set)} training set records, {len(valid_set)} validation set records')

    return train_set, valid_set
    

def encode_train_and_valid_sets(train, valid, input_encoder, config, seq_len):
    
    encoded_train_set = encode_dataset(train, input_encoder, config.mechanism_labels, seq_len = seq_len, needs_filtering = True, \
            dataset_name = 'Training set')
    
    if valid is None:
        encoded_valid_set = None
    else:
        encoded_valid_set = encode_dataset(valid, input_encoder, mechanism_labels, seq_len = seq_len, needs_filtering = True, \
                dataset_name = 'Validation set')

    return encoded_train_set, encoded_valid_set


def main(config):

    config.set_gpu()
    config.set_seed()
    #output_spec = None
    
    dataset_path = config.create_dataset_or_model_path(create_dataset_path = True)
    train_set, valid_set = make_train_and_valid_sets(dataset_path)
    
    # Loading the pre-trained model and fine-tuning it on the loaded dataset

    pretrained_model_generator, input_encoder = load_pretrained_model(local_model_dump_dir = 'proteinbert/proteinbert_models', local_model_dump_file_name = 'default.pkl')

    # get_model_with_hidden_layers_as_outputs gives the model output access to the hidden layers (on top of the output)
    model_generator = FinetuningModelGenerator(pretrained_model_generator, pretraining_model_manipulation_function = \
                                               get_model_with_hidden_layers_as_outputs, dropout_rate = 0.5)

    training_callbacks = [
        keras.callbacks.ReduceLROnPlateau(patience = 1, factor = 0.25, min_lr = 1e-05, verbose = 1),
        keras.callbacks.EarlyStopping(patience = 2, restore_best_weights = True),
    ]

    finetune(model_generator, input_encoder, config, train_set, valid_set, \
            seq_len = 512, batch_size = 32, max_epochs_per_stage = 40, lr = 1e-04, begin_with_frozen_pretrained_layers = True, \
            lr_with_frozen_pretrained_layers = 1e-02, n_final_epochs = 1, final_seq_len = 1024, final_lr = 1e-05, callbacks = training_callbacks)
    
    finetuned_model = model_generator.create_model(seq_len = 1578)
    
    finetuned_model_path = config.create_dataset_or_model_path(create_dataset_path = False)
    with open(finetuned_model_path, 'wb') as f:
        pickle.dump((finetuned_model.get_weights(), finetuned_model.optimizer.get_weights()), f)

if __name__ == "__main__": 

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fold', type=int, help='The number of iterations in 5-fold CV.')
    parser.add_argument('-LHD', '--use_LHD', action='store_true', help='Whether you use Low Homology Dataset or not.', default=False)
    parser.add_argument('-t', '--threshold', type=float, help='Sequence similarity thresholds set when creating LHD.', default=0)
    parser.add_argument('-g', '--gpu', type=int, help='Assign the GPU devices you will use.', default=None)
    parser.add_argument('-s', '--seed', type=int, help='Set random seed.', default=None)
    config = Config_Finetuning(parser.parse_args())
    
    main(config)