#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lavanya
Main script for constrained contrastive learning
Requires training data in HDF5 files with
images ( N x H x W x Contrast dim)
constraint maps (N x H x W x 1)
"""
import sys

import pathlib
sys.path.append('./')

import config.pretrain_config as cfg
from utils.utils import setup_TF_environment, get_callbacks

setup_TF_environment(cfg.gpus_available)

#%%  Load model    

from utils.model_utils_gn import modelObj

mm_utils = modelObj(cfg)

if cfg.partial_decoder:
    print('Initializing partial_decoder')
    ae_pretrain = mm_utils.encoder_decoder_network(add_PH=True, num_dec_levels=3, PH_str='ccl') 
else:
    print('Initializing full decoder')
    ae_pretrain = mm_utils.encoder_decoder_network(add_PH=True, PH_str='ccl') 

if cfg.warm_start:
    print('Loading weights for warm start', cfg.warm_start_wts)
    ae_pretrain.load_weights(cfg.warm_start_wts, by_name=True)
    save_str = '_warm_start' 
else:
    print('No warm start')
    save_str = ''

#%% Load the datagenerator.
from Datagen.h5_pretrain_Data_Generator import DataLoaderObj
''' Load train data'''

train_gen = DataLoaderObj(cfg, 
                        train_flag=True) 

val_gen = DataLoaderObj(cfg, 
                        train_flag=False)
#%%

         
#%% Import loss function and compile model
from tensorflow.keras import optimizers
from utils.constrained_contrastive_loss import lossObj

loss = lossObj(cfg)
customLoss = loss.calc_CCL_batchwise  
AdamOpt = optimizers.Adam(learning_rate=cfg.lr_pretrain)
ae_pretrain.compile(optimizer=AdamOpt, loss=customLoss) 


#%% Generate save_dir

wts_save_dir = cfg.save_dir + '/' + cfg.datatype  
wts_save_dir = wts_save_dir + '/patchsize_' + str(cfg.patch_size)+ '_LR' + str(cfg.lr_pretrain) + '_tau' + str(cfg.temperature) 
wts_save_dir = wts_save_dir + '_top' + str(cfg.topk) + save_str  + '/'
pathlib.Path(wts_save_dir).mkdir(parents=True, exist_ok=True)  
print('Creating ', wts_save_dir) 
modelSavePath = wts_save_dir + 'weights_{epoch:02d}'
csvPath = wts_save_dir + 'training.log'

#%% Create callbacks  

callbacks = get_callbacks(csvPath) 


#%% Train the model

history = ae_pretrain.fit(train_gen,
                        epochs=cfg.num_epochs,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=val_gen,
                        initial_epoch=cfg.initial_epoch,
                        use_multiprocessing=False)

 
# Save final weights
ae_pretrain.save_weights(
    wts_save_dir + 'weights_' + str(cfg.num_epochs) +'.hdf5',
    overwrite=True,
    save_format='h5',
    options=None) 
 
#%% Save the configuration for the run
cfg_txt_name = wts_save_dir + 'config_params.txt'
with open(cfg_txt_name, 'w') as f:
    for name, value in cfg.__dict__.items():
        f.write('{} = {!r}\n'.format(name, value))     
    
    
    
    

