#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lavanya
Configuration file to control training parameters: These configuration parameters are with respect to the segmentation tasks on multi-parametric BraTS dataset.
Except loss params all other settings can be changed. Pretraining assumes hdf5 files for MR images/volumes and their respective constraint maps are already available.
See Datagen/generate_constraint_maps.py and Datagen/generate_h5_pretraining
"""

gpus_available = '0'

''' Data params'''
img_size_x   = 160
img_size_y   = 160
dataset      = 'brats'
contrast_idx = [0,3]        # Choose the contrasts that would be used in the downstream tasks. Here, it is T1Gd + T2-FLAIR along channels 0 and 3
datatype     = 't1_t2'      # datatype string for contrast type
num_channels = len(contrast_idx)
zmean        = True         # zmean normalization is set when multi-parametric data is present in input and per channel normalization is performed.

''' Training params'''
batch_size     = 10
lr_pretrain    = 1e-3
latent_dim     = 64
initial_epoch  = 0
num_epochs     = 150

# settings for full/partial decoder
'''
Note: Training a partial decoder allows more free weights for the finetuning task
Set partial_decoder to 0 for downstream multi-organ segmentation tasks
Set partial_decoder to 1 for general downstream segmentation tasks e.g., tumor
'''

partial_decoder  = 1
warm_start       = 0     # Set to 1 when initializing with global or global/local contrastive pretraining
 
''' Loss params: These are recommended settings'''  
temperature           = 0.1
patch_size            = 4  
topk                  = 100
num_samples_loss_eval = 20    
contrastive_loss_type = 2    # pairwise,   options are 1: setwise, 2: pairwise (recommended)
use_mask_sampling     = 1    # always set it to 1, datagenerator randomly identifies local regions for loss calculation
 
 
base_save_dir = '/pretrained_wts/save/path/here'
 
hdf5_train_cluster_filename = '/path/train_constraint_map.hdf5'
hdf5_val_cluster_filename   = '/path/val_constraint_map.hdf5'

if zmean:
    ''' Select appropriate training HDF5 files'''
    hdf5_train_img_filename = '/path/train_here.hdf5'
    hdf5_val_img_filename   = '/path/val_here.hdf5'
     
    if partial_decoder:
        save_dir = base_save_dir + 'pretrain_partial_dec_zmean/' 
    else:
        save_dir = base_save_dir + 'pretrain_full_dec_zmean/'
else:
    hdf5_train_img_filename = '/file/here.hdf5'
    hdf5_val_img_filename   = '/file/here.hdf5'
    if partial_decoder:
        save_dir = base_save_dir + 'pretrain_partial_dec/' 
    else:
        save_dir = base_save_dir + 'pretrain_full_dec/'
 
 


