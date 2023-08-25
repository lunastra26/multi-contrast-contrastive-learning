#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lavanya
Generate HDF5 files for pretraining  with the MR-contrast guided contrastive learning approach
Script combines nii data with the constraint maps generated using generate_constraint_maps.py
"""

import nibabel as nib
import h5py  
import numpy as np
import os, natsort

import sys
import scipy.io as sio
import csv

sys.path.append('./')


from utils import myCrop3D, contrastStretch
 
datadir            = '/file/location/here'   # location of pre-processed nii files for pretraining
labeldir           = '/file/location/constraint_maps/'  # location of constraint maps for pretraining
save_dir           = '/output/file/location/here/Constraint_maps/'       
datatype           = 'train'    # options are train/val
contrast_type      = 'contrast_name_here'
opShape            = (160,160)
zmean_norm         = True   # perform zero mean unit std normalization, suited for multi-parametric multi-contrast data
num_param_cluster  = '20'


def normalize_img(img):
    img = (img - img.min())/(img.max()-img.min())
    return img

def normalize_img_zmean(img, mask):
    ''' Zero mean unit standard deviation normalization based on a mask'''
    mask_signal = img[mask>0]
    mean_ = mask_signal.mean()
    std_ = mask_signal.std()
    img = (img - mean_ )/ std_
    return img


''' Use an appropriate dataloader to load multi-contrast training data
    Example here is for the brain tumor segmentation dataset'''

def load_unl_brats_img(datadir, subName, opShape): 
    ''' Loads a 4D volume from the brats dataset HxWxDxT where the contrasts are [T1Gd, T2w, T1w, T2-FLAIR]'''
    print('Loading MP-MR images for ', subName)
    data_suffix = ['_t1ce.nii.gz', '_t2.nii.gz', '_t1.nii.gz' , '_flair.nii.gz']
    sub_img = []
    for suffix in data_suffix:
        temp = nib.load(datadir + subName + '/' + subName + suffix)
        temp = np.rot90(temp.get_fdata(),-1)
        temp = myCrop3D(temp, opShape)
        temp = normalize_img(temp)
        # generate a brain mask from the first volume. If mask is available, skip this step
        if suffix == data_suffix[0]:  
            mask = np.zeros(temp.shape)
            mask[temp > 0] = 1
        # histogram based channel-wise contrast stretching
        temp = contrastStretch(temp, mask, 0.01, 99.9)
        if zmean_norm:
            temp = normalize_img_zmean(temp, mask)
        else:
            temp = normalize_img(temp)
        sub_img.append(temp)
    sub_img = np.stack((sub_img), axis=-1)
    return  sub_img 

def load_brats_cluster(subName, opShape): 
    print('Loading constraint maps for ', subName)
    temp = sio.loadmat(labeldir + subName + '/' + 'Param_'+ num_param_cluster +'.mat')
    param_label = temp['param']
    param_label = myCrop3D(param_label, opShape)
    return  param_label 
 
#%%
sub_list = natsort.natsorted(os.listdir(datadir)) 
np.random.seed(seed=25000)
np.random.shuffle(sub_list)

''' Select an appropriate split of training and validation data for pretraining'''
if datatype == 'train':
    wd_trunc = sub_list[:-20]
elif datatype == 'val':
    wd_trunc = sub_list[-20:]

if zmean_norm:
    imgs_h5py_filename = save_dir + datatype + '_' + contrast_type + '_pretrain_img_zmean.hdf5' 
else:
    imgs_h5py_filename = save_dir + datatype + '_' + contrast_type + '_pretrain_img.hdf5' 
     
constraintmap_h5py_filename = save_dir + datatype + '_' + contrast_type + '_pretrain_constraint_map.hdf5'  

#%% Generate training and validation files with NxHxWxT for image and HxWxDx1 for constraint maps

ctr = 0 
init_Flag = True   
imgs = []
param_labels = []
num_vols = 30    # writing 30 volumes at a time

for subName in wd_trunc:
    print('SubName', subName, ctr)
    sub_img = load_unl_brats_img(datadir, subName, opShape)
    param_label = load_brats_cluster(subName, opShape)
    sub_img = np.transpose(sub_img, (2,0,1,3))
    param_label = np.transpose(param_label, (2,0,1))
    param_label = param_label[...,np.newaxis] 
    imgs.append(sub_img)
    param_labels.append(param_label)
    img_shape = sub_img.shape
    cluster_shape = param_label.shape
    img_z, img_x, img_y, num_channels = img_shape  
    chunk_size = img_z * num_vols
    ctr+=1
    if ctr // num_vols:
        print('Writing to hdf5')
        imgs = np.stack(imgs)
        param_labels = np.stack(param_labels)   
        imgs = np.reshape(imgs, (chunk_size, img_x, img_y, num_channels))
        param_labels = np.reshape(param_labels, (chunk_size, img_x, img_y, 1)) 
        if init_Flag:
            print('Writing imgs to hdf5')
            with h5py.File(imgs_h5py_filename, 'w') as f:
                dset = f.create_dataset("img", (chunk_size, img_x, img_y, num_channels), 
                                        maxshape=(None, img_x, img_y, num_channels), 
                                        chunks=True, dtype='float64')
                dset[:chunk_size] = imgs
            print('Writing spatial param cluster to hdf5')
            with h5py.File(constraintmap_h5py_filename, 'w') as f:
                dset_p = f.create_dataset("param", (chunk_size, img_x, img_y, 1), 
                                        maxshape=(None, img_x, img_y, 1), 
                                        chunks=(chunk_size, img_x, img_y, 1), dtype='int64')
                dset_p[:chunk_size] = param_labels 
            init_Flag = False
        else:
            print('Appending imgs to hdf5')
            with h5py.File(imgs_h5py_filename, 'a') as f:
                dset = f['img']
                dset_shape = dset.shape
                print('Axis 0 shape', dset_shape[0])
                dset.resize(dset_shape[0] + chunk_size, axis=0)                
                dset[-chunk_size:] = imgs
                print('Changing img shape to ', dset.shape)
                
            with h5py.File(constraintmap_h5py_filename, 'a') as f:
                dset_p = f['param']
                dset_shape = dset_p.shape
                dset_p.resize(dset_shape[0] + chunk_size, axis=0)
                print('Changing cluster shape to ', dset_p.shape)
                dset_p[-chunk_size:] = param_labels 
        print('Resetting counter')
        ctr = 0  # rest counter
        imgs = []
        param_labels = []
 
if len(param_labels) > 0:
    print('Writing last set to hdf5')  
    imgs = np.stack(imgs)
    param_labels = np.stack(param_labels)
    chunk_size = img_z * param_labels.shape[0]
    imgs = np.reshape(imgs, (chunk_size, img_x, img_y, num_channels))
    param_labels = np.reshape(param_labels, (chunk_size, img_x, img_y, 1))
    if init_Flag:
        print('Writing imgs to hdf5')
        with h5py.File(imgs_h5py_filename, 'w') as f:
            dset = f.create_dataset("img", (chunk_size, img_x, img_y, num_channels), 
                                    maxshape=(None, img_x, img_y, num_channels), 
                                    chunks=True, dtype='float64')
            dset[:chunk_size] = imgs  
        print('Writing spatial param cluster to hdf5')
        with h5py.File(constraintmap_h5py_filename, 'w') as f:
            dset_p = f.create_dataset("param", (chunk_size, img_x, img_y, 1), 
                                    maxshape=(None, img_x, img_y, 1)
, 
                                    chunks=(chunk_size, img_x, img_y, 1), dtype='int64')
            dset_p[:chunk_size] = param_labels 
    else:
        print('Appending last set')        
        with h5py.File(imgs_h5py_filename, 'a') as f:
            dset = f['img']
            dset_shape = dset.shape
            dset.resize(dset_shape[0] + imgs.shape[0], axis=0)
            print('Changing shape to ', dset.shape)
            dset[-imgs.shape[0]:] = imgs  
        with h5py.File(constraintmap_h5py_filename, 'a') as f:
                dset_p = f['param']
                dset_shape = dset_p.shape
                dset_p.resize(dset_shape[0] + imgs.shape[0], axis=0)
                print('Changing cluster shape to ', dset_p.shape)
                dset_p[-imgs.shape[0]:] = param_labels 

