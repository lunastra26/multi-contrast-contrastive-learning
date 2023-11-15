'''
Loss function to train a DL model encoder using global contrastive loss with anatomical similarity
Scripts for Global contrastive learning adapted/reused from Global Local Contrastive loss by Chaitanya et al.
For details see: https://github.com/krishnabits001/domain_specific_cl
'''

import tensorflow as tf
from keras import backend as K
import numpy as np


class lossObj:
    def __init__(self, cfg):
        self.cfg = cfg
        self.temperature = cfg.temperature
        self.batch_size = cfg.batch_size
        self.n_parts = cfg.n_parts
        self.img_size_x = cfg.img_size_x
        self.img_size_y = cfg.img_size_y
        self.num_channels = cfg.num_channels
        self.global_scale_factor = 10

    def cosine_similarity(self,vector_a,vector_b, temperature):
        '''
        Calculating cosine similarity between two vectors
        '''
        norm_vector_a = tf.nn.l2_normalize(vector_a,axis=-1)
        norm_vector_b = tf.nn.l2_normalize(vector_b,axis=-1)
        cos_sim_val=tf.linalg.matmul(norm_vector_a,norm_vector_b,transpose_b=True)/temperature
        return cos_sim_val
    
    def simCLR(self, y_true, y_pred):
        '''
        Simple Contrastive Learning. (SimCLR)
        For batch size N, list contains of 2N elements. Positive pairs of images 
        exist at following indices: (0,1),(2,3),(4,5)....(2N-1,2N)
        ignore y_true, only use y_pred for unlabeled  2N images
        Input
        -------
        List of 2N images 
        
        Returns
        -------
        Contrastive loss 

        '''
        bs=2*self.batch_size
        temperature = self.temperature
        total_loss=0
        for pos_index in range(0,bs,2):
            # Extract positive sets of examples
            pos_i1=np.arange(pos_index,pos_index+1,dtype=np.int32)
            pos_i2=np.arange(pos_index+1,pos_index+2,dtype=np.int32)
    
            # Extract negative sets of examples for each positive pair
            neg_i1=np.arange(0,bs,dtype=np.int32)
            neg_i1 = np.delete(neg_i1, pos_index)
            neg_i2=np.arange(0,bs,dtype=np.int32)
            neg_i2 = np.delete(neg_i2, pos_index+1)
    
            # extract representations corresponding to positive and negative indices
            z_pos_i1=tf.gather(y_pred,pos_i1)
            z_pos_i2=tf.gather(y_pred,pos_i2)

            z_neg_i1=tf.gather(y_pred,neg_i1)
            z_neg_i2=tf.gather(y_pred,neg_i1)

            # calculate probabilities for z_pos_i1
            pos_sim_i1=self.cosine_similarity(z_pos_i1,z_pos_i2,temperature)
            neg_sim_i1=self.cosine_similarity(z_pos_i1,z_neg_i1,temperature)
            loss_1=-tf.math.log(tf.exp(pos_sim_i1)/tf.reduce_sum(tf.exp(neg_sim_i1)))            
            total_loss = total_loss + loss_1
    
            # calculate probabilities for z_pos_i2
            pos_sim_i2=self.cosine_similarity(z_pos_i2,z_pos_i1,temperature)
            neg_sim_i2=self.cosine_similarity(z_pos_i2,z_neg_i2,temperature)
            loss_2=-tf.math.log(tf.exp(pos_sim_i2)/tf.reduce_sum(tf.exp(neg_sim_i2)))
            total_loss = total_loss + loss_2
        global_loss=total_loss/bs
        return global_loss
    
    def anatomical_similarity_global(self,y_true, y_pred):
        ######################
        # From: https://github.com/krishnabits001/domain_specific_cl
        # G^{D-} - Proposed variant
        # We split each volume into n_parts and select 1 image from each n_part of the volume
        # We select the negative samples that we want to contrast against for a given positive image.
        # Example: if positive image is from partition 1 of volume 1, then NO negative sample are taken from partition 1 of any other volume (including volume 1).
        ######################
        bs=3*self.batch_size
        net_global_loss = 0
       
        n_parts = self.n_parts
        # loop over each pair of positive images in the batch to calculate the Net global contrastive loss over the whole batch.
        for pos_index in range(0,self.batch_size,1):
            #indexes of positive pair of samples (x_1,x_2,x_3) - we can make 3 pairs: (x_1,x_2), (x_1,x_3), (x_2,x_3)
            num_i1=np.arange(pos_index,pos_index+1,dtype=np.int32)
            j=self.batch_size+pos_index
            num_i2=np.arange(j,j+1,dtype=np.int32)
            j=2*self.batch_size+pos_index
            num_i3=np.arange(j,j+1,dtype=np.int32)
            #print('n1,n2,n3',num_i1,num_i2,num_i3)

            # indexes of corresponding negative samples as per positive pair of samples: (x_1,x_2), (x_1,x_3), (x_2,x_3)
            den_index_net=np.arange(0,bs,dtype=np.int32)

            # Pruning the negative samples
            # Deleting the indexes of the samples in the batch used as negative samples for a given positive image. These indexes belong to identical partitions in other volumes in the batch.
            # Example: if positive image is from partition 1 of volume 1, then NO negative sample are taken from partition 1 of any other volume (including volume 1) in the batch
            ind_l=[]
            rem = int(num_i1) % n_parts
            for not_neg_index in range(rem, bs, 4):
                ind_l.append(not_neg_index)

            #print('ind_l',ind_l)
            den_indexes = np.delete(den_index_net, ind_l)
            #print('d1',den_i1,len(den_i1))

            # gather required positive samples x_1,x_2,x_3 for the numerator term
            x_num_i1=tf.gather(y_pred,num_i1)
            x_num_i2=tf.gather(y_pred,num_i2)
            x_num_i3=tf.gather(y_pred,num_i3)

            # gather required negative samples x_1,x_2,x_3 for the denominator term
            x_den=tf.gather(y_pred,den_indexes)

            # calculate cosine similarity score + global contrastive loss for each pair of positive images

            #for positive pair (x_1,x_2);
            # numerator of loss term (num_i1_i2_ss) & denominator of loss term (den_i1_i2_ss) & loss (num_i1_i2_loss)
            num_i1_i2_ss=self.cosine_similarity(x_num_i1,x_num_i2,self.temp_fac)
            den_i1_i2_ss=self.cosine_similarity(x_num_i1,x_den,self.temp_fac)
            num_i1_i2_loss=-tf.math.log(tf.exp(num_i1_i2_ss)/(tf.exp(num_i1_i2_ss)+tf.reduce_sum(tf.exp(den_i1_i2_ss))))
            
            net_global_loss = net_global_loss + num_i1_i2_loss
            # for positive pair (x_2,x_1);
            # numerator same & denominator of loss term (den_i1_i2_ss) & loss (num_i1_i2_loss)
            den_i2_i1_ss=self.cosine_similarity(x_num_i2,x_den,self.temp_fac)
            num_i2_i1_loss=-tf.math.log(tf.exp(num_i1_i2_ss)/(tf.exp(num_i1_i2_ss)+tf.reduce_sum(tf.exp(den_i2_i1_ss))))
           
            net_global_loss = net_global_loss + num_i2_i1_loss

            # for positive pair (x_1,x_3);
            # numerator of loss term (num_i1_i3_ss) & denominator of loss term (den_i1_i3_ss) & loss (num_i1_i3_loss)
            num_i1_i3_ss=self.cosine_similarity(x_num_i1,x_num_i3,self.temp_fac)
            den_i1_i3_ss=self.cosine_similarity(x_num_i1,x_den,self.temp_fac)
            num_i1_i3_loss=-tf.math.log(tf.exp(num_i1_i3_ss)/(tf.exp(num_i1_i3_ss)+tf.reduce_sum(tf.exp(den_i1_i3_ss))))
            net_global_loss = net_global_loss + num_i1_i3_loss
            # for positive pair (x_3,x_1);
            # numerator same & denominator of loss term (den_i3_i1_ss) & loss (num_i3_i1_loss)
            den_i3_i1_ss=self.cosine_similarity(x_num_i3,x_den,self.temp_fac)
            num_i3_i1_loss=-tf.math.log(tf.exp(num_i1_i3_ss)/(tf.exp(num_i1_i3_ss)+tf.reduce_sum(tf.exp(den_i3_i1_ss))))
            net_global_loss = net_global_loss + num_i3_i1_loss

            # for positive pair (x_2,x_3);
            # numerator of loss term (num_i2_i3_ss) & denominator of loss term (den_i2_i3_ss) & loss (num_i2_i3_loss)
            num_i2_i3_ss=self.cosine_similarity(x_num_i2,x_num_i3,self.temp_fac)
            den_i2_i3_ss=self.cosine_similarity(x_num_i2,x_den,self.temp_fac)
            num_i2_i3_loss=-tf.math.log(tf.exp(num_i2_i3_ss)/(tf.exp(num_i2_i3_ss)+tf.reduce_sum(tf.exp(den_i2_i3_ss))))
            net_global_loss = net_global_loss + num_i2_i3_loss
            # for positive pair (x_3,x_2):
            # numerator same & denominator of loss term (den_i3_i2_ss) & loss (num_i3_i2_loss)
            den_i3_i2_ss=self.cosine_similarity(x_num_i3,x_den,self.temp_fac)
            num_i3_i2_loss=-tf.math.log(tf.exp(num_i2_i3_ss)/(tf.exp(num_i2_i3_ss)+tf.reduce_sum(tf.exp(den_i3_i2_ss))))
            net_global_loss = net_global_loss + num_i3_i2_loss
        reg_cost=net_global_loss/bs
        reg_cost = tf.reduce_mean(reg_cost) * self.global_scale_factor
        return reg_cost
    
 