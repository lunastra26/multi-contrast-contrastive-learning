"""
Loss functions for segmentation tasks
@author: lavanya
"""

 
import tensorflow as tf
import numpy as np

class lossObj:
    def __init__(self):
        print('Loss init.')

    def tversky_loss(self, y_true, y_pred):
        '''
        # Ref: salehi17, "Twersky loss function for image segmentation using 3D FCDN"
        # -> the score is computed for each class separately and then summed
        # alpha=beta=0.5 : dice coefficient
        # alpha=beta=1   : tanimoto coefficient (also known as jaccard)
        # alpha+beta=1   : produces set of F*-scores
        # using alpha = 0.3, beta=0.7 from recommendation in https://arxiv.org/pdf/1706.05721.pdf
        '''
        alpha = 0.3
        beta = 0.7
        
        ones = tf.cast(tf.ones_like(y_true), tf.float32)
        p0 = y_pred      # prob. that voxels are class i
        p1 = ones-y_pred # prob. that voxels are not class i
        g0 = y_true
        g1 = ones-y_true
        
        num = tf.cast(tf.reduce_sum(tf.multiply(p0,g0), axis=[0,1,2]), tf.float32)
        term1 = tf.cast(alpha * tf.reduce_sum(tf.multiply(p0,g1), axis=[0,1,2]), tf.float32)
        term2 = tf.cast(beta * tf.reduce_sum(tf.multiply(p1,g0), axis=[0,1,2]), tf.float32)        
        den = num + term1 + term2
        
        loss_term = tf.reduce_sum(num/den) # when summing over classes, T has dynamic range [0 Ncl]
        
        num_cl = tf.cast(tf.shape(y_true)[-1], tf.float32)
        return num_cl - loss_term
    
 
 
        

    
 