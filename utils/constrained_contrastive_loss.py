"""
@author: lavanya
Constrained contrastive loss for MR contrast guided contrastive learning
For randomly sampled patch locations in an input image, the loss function first identifies
1) top 100 similar patches in the representational space using cosine similarity (background neighbors)
2) patches that share the same class in the constraint map as the patch of interest (close neighbors)
The intersection of the two sets form the positive examples for constrained contrastive learning.
"""

import numpy as np
import tensorflow as tf


class lossObj:
    def __init__(self,cfg):
        self.patch_size            = cfg.patch_size    
        self.topk                  = cfg.topk           
        self.num_samples_loss_eval = cfg.num_samples_loss_eval
        self.temperature           = cfg.temperature   
        self.batch_size            = cfg.batch_size
        self.partial_decoder       = cfg.partial_decoder
        self.contrastive_loss_type = cfg.contrastive_loss_type
        self.use_mask_sampling     = cfg.use_mask_sampling
        self.epsilon               =  1e-7
    

    def cosine_similarity(self, vector_a, vector_b):
        '''
        Calculating cosine similarity between two vectors
        '''     
        norm_vector_a = tf.nn.l2_normalize(vector_a,axis=-1)
        norm_vector_b = tf.nn.l2_normalize(vector_b,axis=-1)
        cosine_similarity_val=tf.linalg.matmul(norm_vector_a,norm_vector_b,transpose_b=True)    
        return cosine_similarity_val
    
    def get_softmax(self, sim_metric):
        '''
        Compute probability associated with a similarity metric
        ''' 
        prob = tf.math.exp(sim_metric/self.temperature )    
        return prob

    
    def calc_CCL_batchwise(self, y_true, y_pred):
        '''
        y_true: cluster_arr and mask
                y_true[...,0] cluster: cluster map or constraint map for the slice/volume of interest
                y_true[...,1]  mask: random patches for loss evaluation
        y_pred: Feature maps in the representational space
        Returns
        -------
        Local constrained contrastive loss per batch
        '''
        batch_size = self.batch_size
        batch_la_loss=0.0
        for batch_idx in range(batch_size):
            curr_batch_idx = np.arange(batch_idx, batch_idx+1)
            ft_img =  tf.gather(y_pred, curr_batch_idx)  
            cluster_mask_arr = tf.gather(y_true, curr_batch_idx)
            curr_img_loss = self.calc_CCL_imagewise(ft_img, cluster_mask_arr)
            batch_la_loss = batch_la_loss + curr_img_loss
        batch_la_loss = batch_la_loss / batch_size
        return batch_la_loss
    
    def calc_CCL_imagewise(self, ft_img, cluster_arr):
        ''' 
        Function calculates constrained contrastive loss from random patch locations in cluster_arr within an image
        ft_img     : Feature image from DL model e.g., \Psi(x) for x
        cluster_arr : consists of [cluster_arr, mask_arr] concatenated along axis=-1
        cluster_arr: class map for constraints. 
        mask_arr   : contains random patch locations for loss calculations for each image. calculated a priori by setting use_mask_sampling in config
        patch_size : local neighborhood size over which features are averaged
        top_k      : number of background neighbors for relative similarity calculation   
        contrastive_loss_type: 2 for pairwise (recommended), 1 for setwise
        '''
    
        patch_size            = self.patch_size     
        partial_decoder       = self.partial_decoder  
        num_samples_loss_eval = self.num_samples_loss_eval

            
        if partial_decoder:
            ''' Note that constraint maps are already downsampled patch size x patch size (datagenerator) as they retain max class value per patch
                On using partial decoder, a 1x1 patch in the output feature map corresponds to 4x4 in the full decoder'''
            patch_size=1   
        
        if patch_size == 1:
            pred_blk = tf.squeeze(ft_img)
        else:
            ''' generates (num_patches, num_patches, patch_size x patch_size x latent dim)'''
            pred_blk = tf.image.extract_patches(ft_img,  \
                                                sizes=[1,patch_size,patch_size,1], \
                                                strides=[1,patch_size,patch_size,1], \
                                                rates=[1,1,1,1], \
                                                padding='VALID')
     
            
            pred_blk = tf.squeeze(pred_blk)  

        num_patches = pred_blk.shape[0]
        
        ''' calculate  ft representation in each patch 
        In CCL, local representations are learnt w.r.t the image in question. As such, all local patches within an image are contained in the
        ''memory bank'' 
        '''
        memory_bank = tf.reshape(pred_blk,(num_patches*num_patches, -1))
        cluster_arr = tf.reshape(tf.squeeze(cluster_arr),(num_patches*num_patches, -1))
        
        self.len_memory_bank = len(memory_bank) 

        if (self.use_mask_sampling):
            cluster_arr, mask_arr = tf.split(cluster_arr,
                                                num_or_size_splits=2,
                                                axis=-1)    
        if self.use_mask_sampling:                                                
            ''' identify the non-zero indices from mask to select random inputs'''
            random_ip_idx = tf.random.shuffle(self.get_mask_indices(tf.squeeze(mask_arr)))                    
        else:
            ''' identify random locations from the memory bank for loss calculations'''
            random_ip_idx = self.sample_input_patches(num_patches)                    
            
        ''' calculate loss only if atleast topk patches exist in an image'''    
        local_loss = 0.0 
        if (len(memory_bank) > self.topk+1):
            random_ip =  tf.gather(memory_bank, random_ip_idx)  
            for patch_idx in range(0,num_samples_loss_eval):                 
                ip_idx =  tf.gather(random_ip_idx, np.arange(patch_idx, patch_idx+1))
                curr_ip = tf.gather(random_ip, np.arange(patch_idx, patch_idx+1)) 
                pos_prob, neg_prob = self.get_nei_probability(curr_ip,
                                                              ip_idx, 
                                                              memory_bank, 
                                                              cluster_arr)
                curr_patch_loss = self.calculate_contrastive_loss(pos_prob, neg_prob)  
                local_loss = local_loss +  curr_patch_loss 
             
                    
            local_loss_image = local_loss/num_samples_loss_eval
        else:
            local_loss_image = local_loss

        return local_loss_image
    
    
    def get_nei_probability(self, curr_ip, ip_idx, memory_bank, cluster_arr):
        '''
        For an input location, calculate its positive and negative set of neighbors and their probabilities
        Parameters
        ----------
        curr_ip : current input patch  
        ip_idx : index of the current input patch  
        memory_bank : tensor containing all non-overlapping patches in an image  
        cluster_arr : cluster constraints - contains constraint map for the current image
        Returns
        -------
        probabilities for the positive and negative neighbors identitifed through constrained contrastive learning
        '''
        topk        = self.topk
                         
        len_memory_bank = len(memory_bank)
        pcluster_arr_flat = cluster_arr 
        ''' calculate similarity for all data points'''
        all_dp_sim = self.cosine_similarity(memory_bank, curr_ip)
        all_dp_probs = self.get_softmax(all_dp_sim)  
        
        ''' top_k can only be applied along last dim '''
        bg_nei_sim, bg_nei_idx  = tf.nn.top_k(tf.transpose(all_dp_sim), k=topk+1, sorted = False)
        bg_nei_sim = tf.transpose(bg_nei_sim)
        bg_nei_idx = tf.transpose(bg_nei_idx)
   
        ''' Get background similar neighbors''' 
        tf_tensor = tf.zeros_like(all_dp_probs, dtype=tf.bool) 
        tf_updates = tf.ones_like(bg_nei_idx, dtype=tf.bool)
        bg_neighbors = tf.tensor_scatter_nd_update(tf_tensor, bg_nei_idx, tf_updates)
        
        ''' Get parametric neighbors'''
        curr_class_label = tf.gather(pcluster_arr_flat, ip_idx) 
        rep = tf.tile(curr_class_label,  [len_memory_bank,1])
        close_neighbors = tf.equal(pcluster_arr_flat,rep)
           
        relevant_pos_neighbors = tf.logical_and(bg_neighbors, close_neighbors)
        relevant_neg_neighbors = tf.logical_and(bg_neighbors, tf.logical_not(close_neighbors))
        
        pos_prob = tf.gather(tf.squeeze(all_dp_probs),tf.where(tf.squeeze(relevant_pos_neighbors)))
        neg_prob = tf.gather(tf.squeeze(all_dp_probs),tf.where(tf.squeeze(relevant_neg_neighbors)))
        
        return pos_prob, neg_prob
        
    def calculate_contrastive_loss(self, pos_prob, neg_prob):
        loss_type = self.contrastive_loss_type
        if loss_type == 1:
            ''' set-wise contrastive loss'''
            pos_prob = tf.reduce_sum(pos_prob)
            neg_prob = tf.reduce_sum(neg_prob)
            relative_probability =  pos_prob/(pos_prob + neg_prob)
            curr_loss = -tf.reduce_mean(tf.math.log(relative_probability + self.epsilon))
        elif loss_type == 2:
            '''pairwise contrastive loss : RECOMMENDED'''
            # print('pairwise')           
            neg_prob = tf.reduce_sum(neg_prob)
            den_term = pos_prob + neg_prob
            relative_probability = tf.divide(pos_prob, den_term)
            curr_loss = -tf.reduce_mean(tf.math.log(relative_probability + self.epsilon)) 
        return curr_loss
   
    def sample_input_patches(self, num_patches, mask=None, exclude_num=3):
        num_samples_loss_eval = self.num_samples_loss_eval
        if mask is not None:

            non_zero_indexes = tf.cast(tf.where(tf.not_equal(tf.squeeze(mask),0)), tf.int32)
            non_zero_indexes = tf.random.shuffle(non_zero_indexes)
            random_ip_idx = non_zero_indexes[:num_samples_loss_eval] 
        else:            
            img_indices = np.reshape(np.arange(0,num_patches*num_patches),(num_patches,num_patches))
            ''' exclude the first 2 and last 2 rows and columns'''
            img_indices_cropped = img_indices[exclude_num:-exclude_num,exclude_num:-exclude_num]
            img_indices_flat = np.reshape(img_indices_cropped,(-1,))
            random_ip_idx = np.random.choice(img_indices_flat, num_samples_loss_eval, replace=False)        
        return random_ip_idx 

    
    def get_mask_indices(self, mask):
        ''' Returns the indices where mask is set to 1 '''             
        nnz_indexes = tf.cast(tf.where(tf.greater(tf.squeeze(mask),0)), tf.int32)
        return tf.squeeze(nnz_indexes) 
 

    
 
    
 
