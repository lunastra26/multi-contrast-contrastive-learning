 
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


def calc_labelwise_dice_hd(y_true, y_pred, num_labels):
    from medPyUtils import hd95
    lbl_dice = []
    lbl_hd = []
    ''' Exclude background from Dice calc '''
    for idx in range(1,num_labels):
        x1 = np.zeros(y_true.shape).astype(y_true.dtype)
        x2 = np.zeros(y_true.shape).astype(y_true.dtype)
        x1[y_true == idx] = 1
        x2[y_pred == idx] = 1
        if np.any(x1):
            if np.any(x2):
                lbl_hd.append(hd95(x2,x1))
                lbl_dice.append(f1_score(x1.flatten(), x2.flatten(),  average='binary'))
            else:
                lbl_hd.append(200.0)
                lbl_dice.append(f1_score(x1.flatten(), x2.flatten(),  average='binary'))
        else:
            if np.any(x2):
                lbl_hd.append(200.0)
                lbl_dice.append(f1_score(x1.flatten(), x2.flatten(),  average='binary'))
            else:
                lbl_hd.append(0.0)  
                lbl_dice.append(1.0)
   
    return lbl_dice, lbl_hd

    
def calc_stats_v2(y_true, y_pred): 
    if (y_true.any() and y_pred.any()):
        HD_list = getHausdorff(y_true,y_pred)
        Precision_list = calc_precision(y_true,y_pred)
        Recall_list = calc_recall(y_true,y_pred)
    elif (y_true.any() or y_pred.any()):        
        HD_list = 200.0
        Precision_list = 0
        Recall_list = 0
    else:
        HD_list = 0.0
        Precision_list = 1
        Recall_list = 1
        
    return HD_list, Precision_list, Recall_list

def preprocess_eval_mask_labels(mask):
    ''' generate evaluation labels to be consistent with the BraTS challenge
    Label 1 : necrotic core
    Label 2 : edema
    Label 3: enhancing tumor
    
    '''
    mask_TC = mask.copy()
    mask_TC[mask_TC == 1] = 1
    mask_TC[mask_TC == 2] = 1

    mask_ET = mask.copy()
    mask_ET[mask_ET == 1] = 0
    mask_ET[mask_ET == 2] = 1
    
    eval_mask = mask = np.stack([mask_TC, mask_ET])    
    return eval_mask


def normalize_img(img):
    img = (img - img.min())/(img.max()-img.min())
    return img
    
def performDenoising(ipImg, wts):
    max_val = np.max(ipImg)
    ipImg = ipImg / max_val  # Rescale to 0 to 1
    opImg = denoise_tv(ipImg,wts)
    opImg = opImg *  max_val
    return opImg


 
def calc_precision(y_true, y_pred):
    '''What proportion of positive identifications was actually correct?'''
    intersect = y_true * y_pred    
    precision = np.true_divide(intersect.sum(), y_pred.sum())
    return precision

def calc_recall(y_true, y_pred):
    '''What proportion of actual positives was identified correctly?'''
    intersect = y_true * y_pred    
    recall = np.true_divide(intersect.sum(), y_true.sum())
    return recall


def get_callbacks(csvPath):
    callbacks = [ReduceLROnPlateau(monitor='val_loss',
                                factor=0.5,
                                patience=10,
                                verbose=1),
              CSVLogger(csvPath)]
    return callbacks

    
def augmentation_function(images, labels_present=1, en_1hot=0):
    '''
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
 


def calc_f1_score(gt_mask,predictions_mask):
    y_pred= predictions_mask.flatten()
    y_true= gt_mask.flatten()

    f1_val= f1_score(y_true, y_pred, average=None)

    return f1_val

def getHausdorff(y_true, y_pred):
    import SimpleITK as sitk
    import scipy
    testImage = sitk.GetImageFromArray(y_true.astype('uint8'))
    resultImage = sitk.GetImageFromArray(y_pred.astype('uint8'))
    
    """Compute the Hausdorff distance."""
    
    # Hausdorff distance is only defined when something is detected
    resultStatistics = sitk.StatisticsImageFilter()
    resultStatistics.Execute(resultImage)
    if resultStatistics.GetSum() == 0:
        return float('nan')

    eTestImage   = sitk.BinaryErode(testImage, (1,1,0) )
    eResultImage = sitk.BinaryErode(resultImage, (1,1,0) )
    
    hTestImage   = sitk.Subtract(testImage, eTestImage)
    hResultImage = sitk.Subtract(resultImage, eResultImage)    
    
    hTestArray   = sitk.GetArrayFromImage(hTestImage)
    hResultArray = sitk.GetArrayFromImage(hResultImage)   
    testCoordinates   = [testImage.TransformIndexToPhysicalPoint(x.tolist()) for x in np.transpose( np.flipud( np.nonzero(hTestArray) ))]
    resultCoordinates = [testImage.TransformIndexToPhysicalPoint(x.tolist()) for x in np.transpose( np.flipud( np.nonzero(hResultArray) ))]      
    # Use a kd-tree for fast spatial search
    def getDistancesFromAtoB(a, b):    
        kdTree = scipy.spatial.KDTree(a, leafsize=100)
        return kdTree.query(b, k=1, eps=0, p=2)[0]
    
    # Compute distances from test to result; and result to test
    dTestToResult = getDistancesFromAtoB(testCoordinates, resultCoordinates)
    dResultToTest = getDistancesFromAtoB(resultCoordinates, testCoordinates)    
    
    return max(np.percentile(dTestToResult, 95), np.percentile(dResultToTest, 95))
    
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