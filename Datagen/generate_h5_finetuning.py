"""
Generate HDF5 files for finetuning
for the brats
"""

import nibabel as nib
import h5py  
import numpy as np
import os, natsort
import sys

sys.path.append('./')
from utils import myCrop3D, contrastStretch

datadir = '/path/to/brats/nii'
save_dir = '/path/to/save/'
datatype = 'train'
opShape = (192,192)
zmean_norm = True   # perform zero mean unit std normalization

def normalize_img(img):
    img = (img - img.min())/(img.max()-img.min())
    return img

def normalize_img_zmean(img, mask):
    mask_signal = img[mask>0]
    mean_ = mask_signal.mean()
    std_ = mask_signal.std()
    img = (img - mean_ )/ std_
    return img
    
def load_unl_brats_img(datadir, subName, opShape): 
    print('Loading MP-MR images for ', subName)
    data_suffix = ['_t1ce.nii.gz', '_t2.nii.gz', '_t1.nii.gz' , '_flair.nii.gz']
    sub_img = []
    for suffix in data_suffix:
        temp = nib.load(datadir + subName + '/' + subName + suffix)
        temp = np.rot90(temp.get_fdata(),-1)
        temp = myCrop3D(temp, opShape)
        if suffix == data_suffix[0]:
            mask = np.zeros(temp.shape)
            mask[temp > 0] = 1
        temp = contrastStretch(temp, mask, 0.01, 99.9)
        if zmean_norm:
            temp = normalize_img_zmean(temp, mask)
        else:
            temp = normalize_img(temp)
        sub_img.append(temp)
    sub_img = np.stack((sub_img), axis=-1)
    return  sub_img  

def load_brats_label(labeldir, subName, opShape): 
    print('Loading labels for ', subName)
    label_suffix = '_seg.nii.gz'
    temp = nib.load(labeldir + subName + '/' + subName + label_suffix)
    temp = np.rot90(temp.get_fdata(),-1)
    label = myCrop3D(temp, opShape)    
    return  label 
 
wd = natsort.natsorted(os.listdir(datadir))
np.random.seed(seed=25000)
np.random.shuffle(wd)
 
if datatype == 'train':
    wd_trunc = wd[:100]
elif datatype == 'val':
    wd_trunc = wd[100:120]
elif datatype == 'test':
    wd_trunc = wd[-50:]
 
    
if not os.path.exists(save_dir):
    os.makedirs(save_dir) 

if zmean_norm:
    labels_h5py_filename = save_dir + datatype + '_ft_img_label_zmean.hdf5' 
else:
    labels_h5py_filename =  save_dir + datatype + '_ft_img_label.hdf5' 

ctr = 0 
init_Flag = True   
imgs = []
labels = []

num_vols = 20   # chunk size control
#%%
for subName in wd_trunc:
    print('SubName', subName, ctr)
    sub_img = load_unl_brats_img(datadir, subName, opShape)
    sub_label = load_brats_label(datadir, subName, opShape)
    sub_img = np.transpose(sub_img, (2,0,1,3))
    sub_label = np.transpose(sub_label, (2,0,1))
    sub_label = sub_label[...,np.newaxis]
    imgs.append(sub_img)
    img_shape = sub_img.shape
    labels.append(sub_label)
    img_z, img_x, img_y, num_channels = sub_img.shape   
    chunk_size = img_z * num_vols
    ctr+=1
    if ctr // num_vols:
        print('Writing to hdf5')
        imgs = np.stack(imgs)
        labels = np.stack(labels)       
        imgs = np.reshape(imgs, (chunk_size, img_x, img_y, num_channels))
        labels = np.reshape(labels, (chunk_size, img_x, img_y, 1))
    
        if init_Flag:
            print('Writing imgs to hdf5')
            with h5py.File(labels_h5py_filename, 'w') as f:
                dset = f.create_dataset("img", (chunk_size, img_x, img_y, num_channels), 
                                        maxshape=(None, img_x, img_y, num_channels), 
                                        chunks=True, dtype='float64')
                dset[:chunk_size] = imgs
                dsetp = f.create_dataset("label", (chunk_size, img_x, img_y, 1), 
                                        maxshape=(None, img_x, img_y, 1), 
                                        chunks=True, dtype='int32')
                dsetp[:chunk_size] = labels               
            init_Flag = False
        else:
            print('Appending imgs to hdf5')
            with h5py.File(labels_h5py_filename, 'a') as f:
                dset = f['img']
                dsetp = f['label']
                dset_shape = dset.shape
                print('Axis 0 shape', dset_shape[0])
                dset.resize(dset_shape[0] + chunk_size, axis=0) 
                dsetp.resize(dset_shape[0] + chunk_size, axis=0)
                dset[-chunk_size:] = imgs
                dsetp[-chunk_size:] = labels
                print('Changing img shape to ', dset.shape)
                
        print('Resetting counter')
        ctr = 0 
        imgs = []
        labels = []
 
if len(imgs) > 0:
    print('Writing last set to hdf5')  
    imgs = np.stack(imgs)
    labels = np.stack(labels) 
    chunk_size = img_z * imgs.shape[0]
    imgs = np.reshape(imgs, (chunk_size, img_x, img_y, num_channels))
    labels = np.reshape(labels, (chunk_size, img_x, img_y, 1))


    if init_Flag:
        print('Writing imgs to hdf5')
        with h5py.File(labels_h5py_filename, 'w') as f:
            dset = f.create_dataset("img", (chunk_size, img_x, img_y, num_channels), 
                                    maxshape=(None, img_x, img_y, num_channels), 
                                    chunks=True, dtype='float64')
            dset[:chunk_size] = imgs
            dsetp = f.create_dataset("label", (chunk_size, img_x, img_y, 1), 
                                    maxshape=(None, img_x, img_y, 1), 
                                    chunks=True, dtype='int32')
            dsetp[:chunk_size] = labels
        
    else:
        print('Appending last set')        
        with h5py.File(labels_h5py_filename, 'a') as f:
            dset = f['img']
            dsetp = f['label']
            dset_shape = dset.shape
            dset.resize(dset_shape[0] + imgs.shape[0], axis=0)
            dsetp.resize(dset_shape[0] + chunk_size, axis=0)
            print('Changing shape to ', dset.shape)
            dset[-imgs.shape[0]:] = imgs 
            dsetp[-imgs.shape[0]:] = labels
 
  
 
 


                
 