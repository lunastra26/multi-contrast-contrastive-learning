"""
@author: lavanya
Data generator for training a DL model encoder with global contrastive loss using anatomical similarity
Parts of the script have been reused/adapted from Global Local Contrastive loss by Chaitanya et al.
For details see: https://github.com/krishnabits001/domain_specific_cl
"""

import tensorflow as tf
import numpy as np
from utils.utils import crop_batch
import random
 
class Data_Generator_Anatomical(tf.keras.utils.Sequence):
    ''' Data generator to randomly sample and augment 2D images with anatomical contrasting strategy'''
    def __init__(self, ip_img_list, cfg):
        print('Initialize data generator for global contrastive learning')
        self.cfg = cfg
        self.batch_size = cfg.batch_size        
        self.n_vols = cfg.n_vols
        self.n_parts = cfg.n_parts
        self.img_x = cfg.img_size_x
        self.img_y = cfg.img_size_y
        self.img_list = ip_img_list
        self.img_arr = np.concatenate(self.img_list, axis=0)  # axis 0 is the slice dimension
        
    def generate_global_0(self):
        '''
        Similar to simple contrastive learning (SimCLR). Generates augmented versions of randomly sampled images
        '''
        while True:
            cfg = self.cfg
            batch_size = cfg.batch_size 
            img_batch = self.shuffle_minibatch()
            # Aug Set 1 - crop followed by intensity aug
            crop_batch1 = crop_batch(img_batch, cfg, batch_size)
            color_batch1 = self.brit_cont_net(crop_batch1)
            # Aug Set 2 - crop followed by intensity aug
            crop_batch2 = crop_batch(img_batch, cfg, batch_size)
            color_batch2 = self.brit_cont_net(crop_batch2)
            # Stitch 3 sets: original images batch, 2 different augmented version of original images batch into 1 batch for pre-training
            cat_batch=self.stitch_two_crop_batches(color_batch1, color_batch2)
            yield(cat_batch,cat_batch)
         
    def generate_global_1(self):
        '''
        Similar to anatomical similarity in Chaitanya et al. https://github.com/krishnabits001/domain_specific_cl
        Images are sampled based on anatomical similarity. see sample_minibatch_for_global_loss_opti for details
        '''
        while True:
            cfg = self.cfg
            n_vols = len(self.img_list)
            n_parts = cfg.n_parts
            batch_size_ft = cfg.batch_size
            img_batch = self.sample_minibatch_for_global_loss_opti(2*batch_size_ft, n_vols=n_vols, n_parts=n_parts) 
            img_batch=img_batch[0:batch_size_ft] 
            # Aug Set 1 - crop followed by intensity aug 
            crop_batch1 = crop_batch(img_batch, cfg, batch_size_ft)
            color_batch1 = self.brit_cont_net(crop_batch1) 
            # Aug Set 2 - crop followed by intensity aug   
            crop_batch2 = crop_batch(img_batch, cfg, batch_size_ft)
            color_batch2 = self.brit_cont_net(crop_batch2)         
            # Stitch 3 sets: original images batch, 2 different augmented version of original images batch into 1 batch for pre-training
            cat_batch = np.concatenate([img_batch,color_batch1,color_batch2],axis=0)
            yield(cat_batch,cat_batch)
        
    def brit_cont_net(self, x_tmp):
        # Pre-training
        rd_brit = tf.image.random_brightness(x_tmp,max_delta=0.5,seed=1)
        rd_cont = tf.image.random_contrast(rd_brit,lower=0.7,upper=1.3,seed=1)
        min_val = x_tmp.min()
        max_val = x_tmp.max()
        rd_fin=tf.clip_by_value(rd_cont,min_val,max_val)
        return  rd_fin
                   
    def sample_minibatch_for_global_loss_opti(self, batch_sz, n_vols=None, n_parts=None):
        '''
        Script reused from https://github.com/krishnabits001/domain_specific_cl
        Create a batch with 'n_parts * n_vols' no. of 2D images where n_vols is no. of 3D volumes and n_parts is no. of partitions per volume.
        input param:
             img_list: input batch of 3D volumes
             cfg: config parameters
             batch_sz: final batch size
             n_vols: number of 3D volumes
             n_parts: number of partitions per 3D volume
        return:
             fin_batch: swapped batch of 2D images.
        '''
    
        count=0
        img_list = self.img_list
        cfg = self.cfg
        
        if n_vols == None:
            n_vols = self.n_vols
        if n_parts == None:
            n_parts = self.n_parts
        
        #select indexes of 'm' volumes out of total M.
        im_ns=random.sample(range(0, len(img_list)), n_vols)
        fin_batch=np.zeros((batch_sz,cfg.img_size_x,cfg.img_size_x,cfg.num_channels))
        #print(im_ns)
        for vol_index in im_ns:
            #print('j',j)
            #if n_parts=4, then for each volume: create 4 partitions, pick 4 samples overall (1 from each partition randomly)
            im_v=img_list[vol_index]
            ind_l=[]
            #starting index of first partition of any chosen volume
            ind_l.append(0)
    
            #find the starting and last index of each partition in a volume based on input image size. shape[0] indicates total no. of slices in axial direction of the input image.
            for k in range(1,n_parts+1):
                ind_l.append(k*int(im_v.shape[0]/n_parts))
            #print('ind_l',ind_l)
    
            #Now sample 1 image from each partition randomly. Overall, n_parts images for each chosen volume id.
            for k in range(0,len(ind_l)-1):
                #print('k',k,ind_l[k],ind_l[k+1])
                if(k+count>=batch_sz):
                    break
                #sample image from each partition randomly
                i_sel=random.sample(range(ind_l[k],ind_l[k+1]), 1)
                #print('k,i_sel',k+count, i_sel)
                fin_batch[k+count]=im_v[i_sel]
            count=count+n_parts
            if(count>=batch_sz):
                break
    
        return fin_batch
    

    def shuffle_minibatch(self):
        tmp_img = []
        img_arr = self.img_arr
        len_of_data = img_arr.shape[0]
        randomize=np.random.choice(len_of_data,size=self.batch_size,replace=True)
        for index_no in randomize:
            tmp_img.append(np.expand_dims(img_arr[index_no,...], axis=0))
        tmp_img = np.concatenate(tmp_img)
        return tmp_img 
    
    
    def stitch_two_crop_batches(self, img_batch1, img_batch2, batch_size=None):
        '''
        # stitch 2 batches of (image,label) pairs with different augmentations applied on the same set of original (image,label) pair
        input params:
            ip_list: list of 2 set of (image,label) pairs with different augmentations applied
            cfg : contains config settings of the image
            batch_size: batch size of final stitched set
        returns:
            cat_img_batch: stitched set of 2 batches of images under different augmentations
            cat_lbl_batch: stitched set of 2 batches of labels under different augmentations
        '''
        
        cfg = self.cfg
        if batch_size == None:
            batch_size = self.batch_size
    
        # cat_img_batch=np.zeros((2*batch_size,cfg.img_size_x,cfg.img_size_y,cfg.num_channels))
        nDim, xDim, yDim, cDim = img_batch1.shape
        cat_img_batch=np.zeros((2*batch_size,xDim,yDim, cDim))
        for index in range(0,2*batch_size,2):
            cat_img_batch[index]  =img_batch1[int(index/2)]
            cat_img_batch[index+1]=img_batch2[int(index/2)]

        return cat_img_batch
    