#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lavanya
Configuration file to control training parameters
"""

gpus_available = '1'

''' Data params'''
img_size_x   = 160
img_size_y   = 160
dataset      = 'brats'
contrast_idx = [0,3]    # Choose the contrasts that would be used in the downstream tasks. Here, it is T1Gd + T2-FLAIR
datatype     = 't1_t2'      # datatype string for contrast type
num_channels = len(contrast_idx)
zmean        = True 

''' Training params'''
batch_size     = 10
lr_pretrain    = 1e-3
latent_dim     = 64
initial_epoch  = 0
num_epochs     = 150
# settings for partial decoder
partial_decoder    = 1
partial_warm_start = 0

# settings for full/partial decoder
warm_start_enc     = 0
warm_start_dec     = 0

''' Loss params'''  
temperature           = 0.1
patch_size            = 4
topk                  = 100
num_samples_loss_eval = 20   # options are 10, 20, and 30
contrastive_loss_type = 2   
use_mask_sampling     = 1
 
 
base_save_dir = '/pretrained_wts/save/path/here'
 
hdf5_train_cluster_filename = '/path/train_constraint_map.hdf5'
hdf5_val_cluster_filename   = '/path/val_constraint_map.hdf5'

if zmean:
    ''' Select appropriate training HDF5 files'''
    hdf5_train_img_filename = '/path/train_here.hdf5'
    hdf5_val_img_filename   = '/path/val_here.hdf5'
     
    if partial_decoder:
        save_dir = base_save_dir + 'pretrain_partial_zmean/' 
    else:
        save_dir = base_save_dir + 'pretrain_full_zmean/'
else:
    hdf5_train_img_filename = '/file/here.hdf5'
    hdf5_val_img_filename   = '/file/here.hdf5'
    if partial_decoder:
        save_dir = base_save_dir + 'pretrain_partial/' 
    else:
        save_dir = base_save_dir + 'pretrain_full/'
 
 


