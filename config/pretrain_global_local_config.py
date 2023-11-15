#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Config for global local contrastive loss
For details on how global contrastive loss works, please refer to
https://github.com/krishnabits001/domain_specific_cl
Example config file provided with respect to brain tumor segmentation dataset

"""
gpus_available = '0'
dataset = ''
datatype = ''

''' Data params'''
img_size_x = 160
img_size_y = 160
contrast_idx = [0, 3]        
num_channels = len(contrast_idx)

''' Training params'''
batch_size = 12
lr_pretrain = 1e-3
n_parts = 3
n_vols = 4
latent_dim = 16
initial_epoch = 0
num_iters = 10000
 
save_dir = '/path/to/save/globalCL/weights/'
data_dir = '/path/to/training/data/folders/'
zmean_norm = 1

''' Loss params''' 
temperature = 0.1
patch_size = 3
num_samples_loss_eval = 10
global_contrastive_loss_type = 'anat_sim'    # options: simCLR (GR) or anat_sim (GD-) 
