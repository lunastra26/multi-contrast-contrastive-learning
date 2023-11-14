"""
@author: lavanya
Data generator for pretraining with MR-contrast guided contrastive learning approach
Given location of HDF5 files for images and the corresponding constraint maps, the datagenerator randomly selects images for training
"""

import tensorflow as tf
import numpy as np
from skimage.util import view_as_blocks
import h5py
import threading
import tensorflow as tf
 
class DataLoaderObj(tf.keras.utils.Sequence):
    ''' Config cfg contains the parameters that control training'''
    def __init__(self, cfg, train_flag=True):        
        self.patch_size = cfg.patch_size
        self.batch_size = cfg.batch_size
        self.contrast_idx = cfg.contrast_idx
        self.num_channels = len(self.contrast_idx)
        self.cfg = cfg
        self.num_constraints = 1
        self.num_clusters = cfg.num_samples_loss_eval
        self.train_flag = train_flag
        if train_flag:
            print('Initializing training dataloader')
            self.input_hdf5_img     = h5py.File(cfg.hdf5_train_img_filename, 'r')
            self.input_hdf5_cluster = h5py.File(cfg.hdf5_train_cluster_filename, 'r')
        else:
            print('Initializing validation dataloader')
            self.input_hdf5_img     = h5py.File(cfg.hdf5_val_img_filename, 'r')
            self.input_hdf5_cluster = h5py.File(cfg.hdf5_val_cluster_filename, 'r')
           
        self.img = self.input_hdf5_img['img']
        self.param_cluster = self.input_hdf5_cluster['param']
       
        if cfg.use_mask_sampling:
            self.num_constraints = self.num_constraints+1
           
        self.len_of_data = self.img.shape[0]
        self.num_samples = self.len_of_data // 1  # use it to control size of sampels per epoch
        print('Total samples :', self.num_samples)
        self.img_size_x = self.img.shape[1]
        self.img_size_y = self.img.shape[2]
        self.arr_indexes = np.random.choice(self.len_of_data, self.num_samples, replace=True)
       
    
    
    def __del__(self):
        self.input_hdf5_img.close() 
        self.input_hdf5_cluster.close()
 
        
    def __len__(self):
        return self.num_samples // self.batch_size

    def get_len(self):
        return self.num_samples // self.batch_size
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.arr_indexes = np.random.choice(self.len_of_data, self.num_samples, replace=True)
        
        
    def __shape__(self):
        data = self.__getitem__(0)
        return data.shape
    
    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        with threading.Lock():
            # Generate indices of the batch
            i = self.arr_indexes[idx * self.batch_size : (idx + 1) * self.batch_size]    
            x_train = self.generate_X(i)
            y_train = self.generate_clusters(i)
            x_train = tf.identity(x_train)
            y_train = tf.identity(y_train)              
            return x_train, y_train
        
    def generate_X(self, list_idx):    
        X = np.zeros((self.batch_size, self.img_size_x, self.img_size_y, self.num_channels),dtype="float64")
        for jj in range(0, self.batch_size):                
            X[jj] = self.img[list_idx[jj],...,self.contrast_idx] 
        return X
       
    
    def generate_mask(self, X):
        ''' Generates a mask from foreground regions along 1st channel of image'''
        mask = np.zeros(X.shape) 
        mask[ X > X.min()] = 1
        return mask 
    
    def generate_clusters(self, list_idx):
        patch_size = self.patch_size
        batch_size = self.batch_size
        num_constraints = self.num_constraints

        X    = np.zeros((self.batch_size, self.img_size_x, self.img_size_y, self.num_channels),dtype="float64")
        Y    = np.zeros((self.img_size_x, self.img_size_y, num_constraints, batch_size),dtype="int64")
        mask = np.zeros((self.img_size_x, self.img_size_y, batch_size),dtype="int64")
        ''' Note that batch dim is transposed in Y for view as blocks function '''

        for jj in range(0, self.batch_size): 
            temp         = self.img[list_idx[jj],...,self.contrast_idx]                
            X[jj]        = temp  
            mask[...,jj] = self.generate_mask(temp[...,0])
            Y[...,0,jj]  = np.squeeze(self.param_cluster[list_idx[jj]]) 
            Y[...,-1,jj] = mask[...,jj]
 
        # This section identifies the majority class in non-overlapping patch_size x patch_size regions of the constraint map
        if patch_size > 1:
            y_train_blk = view_as_blocks(Y,(patch_size, patch_size, num_constraints, batch_size)).squeeze()               
            xDim, yDim = y_train_blk.shape[0], y_train_blk.shape[1]
            y_train_blk = np.reshape(y_train_blk,(xDim, yDim, patch_size*patch_size, num_constraints, batch_size))
            y_train = np.apply_along_axis(self.get_freq_labels,-3, y_train_blk)
        else:
            y_train = Y
        
        for jj in range(0, self.batch_size):
            temp = y_train[...,-1,jj].squeeze()  # get the mask
            y_train[...,-1,jj] = self.generate_indices_from_mask(temp)   # generate indices from mask

        ''' Note that batch dim was transposed in Y for view as blocks function '''  
        Y = np.transpose(y_train,(3,0,1,2))
        return Y
    
    def get_freq_labels(self, arr):
        return np.bincount(arr).argmax()
    
    def generate_indices_from_mask(self, mask):
        ''' Returns a mask with random indices from the brain/image for patchwise CCL loss calculation
        ''' 
        xDim, yDim = mask.shape
        mask_f = np.ndarray.flatten(mask.copy())
        all_idx = np.where(mask_f == 1)[0]
        np.random.shuffle(all_idx)
        temp_idx = all_idx[:self.num_clusters]
        idx_mask = np.zeros(mask_f.shape, dtype='int64')                                            
        idx_mask[temp_idx] = 1
        mask = np.reshape(idx_mask,(xDim,yDim))
        return mask
 
 
    
 