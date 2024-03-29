 
"""
Created on Mon Sep 20 18:10:58 2021

@author: lavan
"""

 
import nibabel as nib
import numpy as np
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import CSVLogger
from skimage import transform
import random
import scipy.ndimage.interpolation
from skimage.restoration import denoise_tv_bregman as denoise_tv
import os, natsort
from sklearn.metrics import f1_score
import os, logging
 

def setup_TF_environment(gpus_available):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus_available
    # Suppressing TF message printouts
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
    logging.getLogger('tensorflow').setLevel(logging.FATAL)
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    return

def contrastStretch(ipImg,ipMask,lwr_prctile = 10,upr_prctile = 100):
    ''' Histogram based contrast stretching '''
    from skimage import exposure
    mm = ipImg[ipMask > 0]
    p10 = np.percentile(mm,lwr_prctile)
    p100 = np.percentile(mm,upr_prctile)
    opImg = exposure.rescale_intensity(ipImg,in_range=(p10,p100))
    return opImg 

def myCrop3D(ipImg,opShape):
    ''' Crop a 3D volume (H x W x D) to the following size (opShape x opShape x D)'''
    xDim,yDim = opShape
    zDim = ipImg.shape[2]
    opImg = np.zeros((xDim,yDim,zDim))
    
    xPad = xDim - ipImg.shape[0]
    yPad = yDim - ipImg.shape[1]
    
    x_lwr = int(np.floor(np.abs(xPad)/2))
    x_upr = int(np.ceil(np.abs(xPad)/2))
    y_lwr = int(np.floor(np.abs(yPad)/2))
    y_upr = int(np.ceil(np.abs(yPad)/2))
    if xPad >= 0 and yPad >= 0:
        opImg[x_lwr:xDim - x_upr ,y_lwr:yDim - y_upr,:] = ipImg
    elif xPad < 0 and yPad < 0:
        xPad = np.abs(xPad)
        yPad = np.abs(yPad)
        opImg = ipImg[x_lwr: -x_upr ,y_lwr:- y_upr,:]
    elif xPad < 0 and yPad >= 0:
        xPad = np.abs(xPad)
        temp_opImg = ipImg[x_lwr: -x_upr,:,:]
        opImg[:,y_lwr:yDim - y_upr,:] = temp_opImg
    else:
        yPad = np.abs(yPad)
        temp_opImg = ipImg[:,y_lwr: -y_upr,:]
        opImg[x_lwr:xDim - x_upr,:,:] = temp_opImg
    return opImg

def load_unl_brats_img(datadir, subName, opShape, zmean_norm=1): 
    ''' Loads a 4D volume from the brats dataset HxWxDxT where the contrasts are [T1Gd, T2w, T1w, T2-FLAIR]'''
    print('Loading MP-MR images for ', subName)
    data_suffix = ['_t1ce.nii.gz', '_t2.nii.gz', '_t1.nii.gz' , '_flair.nii.gz']
    sub_img = []
    for suffix in data_suffix:
        temp = nib.load(datadir + subName + '/' + subName + suffix)
        temp = np.rot90(temp.get_fdata(),-1)
        temp = myCrop3D(temp, opShape)
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

def load_img_labels_brats(opShape=(256,256), datadir=None, normalization='zmean'):
    ''' Load image and label pairs from the BraTS dataset'''
    wd = natsort.natsorted(os.listdir(datadir))       
    data_suffix = ['_t1ce.nii.gz', '_t2.nii.gz', '_t1.nii.gz' , '_flair.nii.gz']
    label_suffix = '_seg.nii.gz'
    imgs = []
    labels = []
    for subName in wd:
        print('Loading subject ', subName)
        sub_img = []
        for suffix in data_suffix:
            temp = nib.load(datadir + subName + '/' + subName + suffix)
            temp = np.rot90(temp.get_fdata(),-1)
            temp = myCrop3D(temp, opShape)
            temp = normalize_img(temp) 
            if suffix == data_suffix[0]:
                mask = np.zeros(temp.shape)
                mask[temp > 0] = 1
            temp = contrastStretch(temp, mask, 0.01, 99.9)
            if normalization =='zmean':
                mask_signal = temp[mask > 0]
                temp = (temp - mask_signal.mean())/ mask_signal.std() 
            else:
                temp = normalize_img(temp)
            sub_img.append(temp)
        sub_img = np.stack((sub_img), axis=-1)
        imgs.append(sub_img)
        temp = nib.load(datadir + subName + '/' + subName + label_suffix)
        temp = np.rot90(temp.get_fdata(),-1)
        label = myCrop3D(temp, opShape)
        labels.append(label)
    return imgs,labels


def normalize_img_zmean(img, mask):
    ''' Zero mean unit standard deviation normalization based on a mask'''
    mask_signal = img[mask>0]
    mean_ = mask_signal.mean()
    std_ = mask_signal.std()
    img = (img - mean_ )/ std_
    return img


def normalize_img(img):
    img = (img - img.min())/(img.max()-img.min())
    return img
    
def performDenoising(ipImg, wts):
    max_val = np.max(ipImg)
    ipImg = ipImg / max_val  # Rescale to 0 to 1
    opImg = denoise_tv(ipImg,wts)
    opImg = opImg *  max_val
    return opImg

def get_callbacks(csvPath):
    callbacks = [ReduceLROnPlateau(monitor='val_loss',
                                factor=0.5,
                                patience=10,
                                verbose=1),
              CSVLogger(csvPath)]
    return callbacks

    
def augmentation_function(images, labels_present=1, en_1hot=0):
    '''
    Script adapted from Chaitanya et al. Global and local contrastive learning
    To generate affine augmented image,label pairs.
    ip params:
        ip_list: list of 2D slices of images and its labels if labels are present
        dt: dataloader object
        labels_present: to indicate if labels are present or not
        en_1hot: to indicate labels are used in 1-hot encoding format
    returns:
        sampled_image_batch : augmented images generated
        sampled_label_batch : corresponding augmented labels
        Script adapted from Chaitanya et al. Global and local contrastive learning
    '''
    if labels_present == 1:
        images,labels = images
    new_images = []
    new_labels = []
    num_images = images.shape[0]

    for index in range(num_images):

        img = np.squeeze(images[index,...])
        if(labels_present==1):
            lbl = np.squeeze(labels[index,...])

        do_rotations,do_scaleaug,do_fliplr,do_simple_rot=0,0,0,0
        aug_select = np.random.randint(5)

        if(np.max(img)>0.001):
            if(aug_select==0):
                do_rotations=1
            elif(aug_select==1):
                do_scaleaug=1
            # elif(aug_select==2):
            #     do_fliplr=1
            # elif(aug_select==3):
            #     do_simple_rot=1

        # ROTATE between angle -15 to 15
        if do_rotations:
            angles = [-15,15]
            # angles = [-5,5]
            random_angle = np.random.uniform(angles[0], angles[1])
            img = scipy.ndimage.interpolation.rotate(img, reshape=False, angle=random_angle, axes=(1, 0),order=1)
            if(labels_present==1):
                if(en_1hot==1):
                    lbl = scipy.ndimage.interpolation.rotate(lbl, reshape=False, angle=random_angle, axes=(1, 0),order=1)
                else:
                    lbl = scipy.ndimage.interpolation.rotate(lbl, reshape=False, angle=random_angle, axes=(1, 0),order=0)

        # RANDOM SCALE
        if do_scaleaug:
            n_x, n_y = img.shape
            #scale factor between 0.95 and 1.05
            scale_fact_min=0.95
            scale_fact_max=1.05
            scale_val = round(random.uniform(scale_fact_min,scale_fact_max), 2)
            slice_rescaled = transform.rescale(img, scale_val, order=1, preserve_range=True, mode = 'constant')
            img = np.squeeze(myCrop3D(slice_rescaled[...,np.newaxis], (n_x, n_y)))
            if(labels_present==1):
                if(en_1hot==1):
                    slice_rescaled = transform.rescale(lbl, scale_val, order=1, preserve_range=True, mode = 'constant')
                    lbl = np.squeeze(myCrop3D(slice_rescaled[...,np.newaxis], (n_x, n_y)))
                else:
                    slice_rescaled = transform.rescale(lbl, scale_val, order=0, preserve_range=True, mode = 'constant')
                    lbl = np.squeeze(myCrop3D(slice_rescaled[...,np.newaxis],( n_x, n_y)))

        # RANDOM FLIP
        if do_fliplr:
            coin_flip = np.random.randint(2)
            if coin_flip == 0:
                img = np.fliplr(img)
                if(labels_present==1):
                    lbl = np.fliplr(lbl)

        # Simple rotations at angles of 45 degrees
        if do_simple_rot:
            fixed_angle = 45
            random_angle = np.random.randint(8)*fixed_angle

            img = scipy.ndimage.interpolation.rotate(img, reshape=False, angle=random_angle, axes=(1, 0),order=1)
            if(labels_present==1):
                if(en_1hot==1):
                    lbl = scipy.ndimage.interpolation.rotate(lbl, reshape=False, angle=random_angle, axes=(1, 0),order=1)
                else:
                    lbl = scipy.ndimage.interpolation.rotate(lbl, reshape=False, angle=random_angle, axes=(1, 0),order=0)

        new_images.append(img[..., np.newaxis])
        if(labels_present==1):
            new_labels.append(lbl[...])

    sampled_image_batch = np.asarray(new_images)
    if(labels_present==1):
        sampled_label_batch = np.asarray(new_labels)

    if labels_present==1:
        return sampled_image_batch, sampled_label_batch
    else:
        return sampled_image_batch
 

    return f1_val
 
def augmentation_function_mc(images, labels, en_1hot=0):
    '''
    Script adapted from Chaitanya et al. Global and local contrastive learning
    To generate affine augmented image,label pairs for multi channel data
    returns:
        sampled_image_batch : augmented images generated
        sampled_label_batch : corresponding augmented labels
    '''

    new_images = []
    new_labels = []
    num_images = images.shape[0]
    num_channels = images.shape[-1]

    for index in range(num_images):    
        img = np.squeeze(images[index,...])           
        lbl = np.squeeze(labels[index,...])

        
        do_rotations,do_scaleaug,do_fliplr,do_simple_rot=0,0,0,0
        aug_select = np.random.randint(5)    
        if(np.max(img)>0.001):
            if(aug_select==0):
                do_rotations=1
            elif(aug_select==1):
                do_scaleaug=1
            elif(aug_select==2):
                do_fliplr=1
            elif(aug_select==3):
                do_simple_rot=1

        # ROTATE between angle -15 to 15
        if do_rotations:
            angles = [-15,15]
            random_angle = np.random.uniform(angles[0], angles[1])
            
            if num_channels > 1:
                temp_img = np.zeros(img.shape)
                for channel_idx in range(num_channels):
                    curr_channel = img[...,channel_idx]
                    temp_img[...,channel_idx] = scipy.ndimage.interpolation.rotate(curr_channel, reshape=False, angle=random_angle, axes=(1, 0),order=1, mode='constant', cval=curr_channel.min())
                img = temp_img
            else:                    
                img = scipy.ndimage.interpolation.rotate(img, reshape=False, angle=random_angle, axes=(1, 0),order=1, mode='constant', cval=img.min())
           
            if(en_1hot==1):
                lbl = scipy.ndimage.interpolation.rotate(lbl, reshape=False, angle=random_angle, axes=(1, 0),order=0)
            else:
                lbl = scipy.ndimage.interpolation.rotate(lbl, reshape=False, angle=random_angle, axes=(1, 0),order=0)

        # RANDOM SCALE
        if do_scaleaug:
            n_x, n_y = img.shape[0], img.shape[1]
            #scale factor between 0.95 and 1.05
            scale_fact_min=0.95
            scale_fact_max=1.05
            scale_val = round(random.uniform(scale_fact_min,scale_fact_max), 2)
            
            if num_channels > 1:
                temp_img = np.zeros(img.shape)
                for channel_idx in range(num_channels):
                    curr_channel = img[...,channel_idx]
                    slice_rescaled = transform.rescale(curr_channel, scale_val, order=1, preserve_range=True, mode = 'constant', cval=curr_channel.min())
                    temp_img[...,channel_idx] =  np.squeeze(myCrop3D(slice_rescaled[...,np.newaxis], (n_x, n_y)))
                img = temp_img
            else:
                slice_rescaled = transform.rescale(img, scale_val, order=1, preserve_range=True, mode = 'constant', cval=img.min())
                img = np.squeeze(myCrop3D(slice_rescaled[...,np.newaxis], (n_x, n_y)))
            
            if(en_1hot==1):
                slice_rescaled = transform.rescale(lbl, scale_val, order=0, preserve_range=True, mode = 'constant')
                lbl =  myCrop3D(slice_rescaled, (n_x, n_y))
            else:
                slice_rescaled = transform.rescale(lbl, scale_val, order=0, preserve_range=True, mode = 'constant')
                lbl = np.squeeze(myCrop3D(slice_rescaled[...,np.newaxis],( n_x, n_y)))
                
                # RANDOM FLIP
        if do_fliplr:
            coin_flip = np.random.randint(2)
            if coin_flip == 0:
                img = np.fliplr(img)                
                lbl = np.fliplr(lbl)

        # Simple rotations at angles of 45 degrees
        if do_simple_rot:
            fixed_angle = 45
            random_angle = np.random.randint(8)*fixed_angle
            if num_channels > 1:
                temp_img = np.zeros(img.shape)
                for channel_idx in range(num_channels):
                    curr_channel = img[...,channel_idx]
                    temp_img[...,channel_idx] = scipy.ndimage.interpolation.rotate(curr_channel, reshape=False, angle=random_angle, axes=(1, 0),order=1,  mode='constant', cval=curr_channel.min())
                img = temp_img
            else:                    
                img = scipy.ndimage.interpolation.rotate(img, reshape=False, angle=random_angle, axes=(1, 0),order=1,  mode='constant', cval=img.min())
           
            if(en_1hot==1):
                lbl = scipy.ndimage.interpolation.rotate(lbl, reshape=False, angle=random_angle, axes=(1, 0),order=0)
            else:
                lbl = scipy.ndimage.interpolation.rotate(lbl, reshape=False, angle=random_angle, axes=(1, 0),order=0)



        if num_channels > 1:
            new_images.append(img)
        else:                    
            new_images.append(img[..., np.newaxis])            
        new_labels.append(lbl[...])    
    sampled_image_batch = np.asarray(new_images)       
    sampled_label_batch = np.asarray(new_labels)
    
    return sampled_image_batch, sampled_label_batch
