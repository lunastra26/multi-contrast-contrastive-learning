'''
Pretrain encoder with global contrastive loss
Adapted/Resued parts from: Global Local Contrastive loss from Chaitanya et al.
For details see: https://github.com/krishnabits001/domain_specific_cl
'''
 
gpus_available = '0'
import os
import numpy as np
import pathlib
import sys
import logging
import scipy.io as sio
from keras.callbacks import CSVLogger

os.environ['CUDA_VISIBLE_DEVICES'] = gpus_available
 
# Suppressing TF message printouts
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

import tensorflow as tf
sys.path.append('./')
from utils.utils import *
 
print('load CNN and data configs')
import config.pretrain_global_local_config as cfg 
from utils.model_utils import modelObj
import Datagen.GL_Data_Generator as GL_generator
from utils.GL_loss import lossObj

setup_TF_environment(cfg.gpus_available)

model = modelObj(cfg, kernel_init= tf.keras.initializers.HeNormal(seed=1))
loss = lossObj(cfg)

global_contrastive = cfg.global_contrastive_loss_type  
opShape = (cfg.img_size_x, cfg.img_size_y) 
datatype = cfg.datatype

# Save directory
save_dir=str(cfg.save_dir)+ '/pretr_global_' + global_contrastive
save_dir = save_dir +'_' + datatype + '_LR_' +  str(cfg.lr_pretrain) +  '/'
pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
modelSavePath = save_dir + 'weights_{epoch:02d}.hdf5'
csvPath = save_dir + 'training.log'
callbacks =  [CSVLogger(csvPath)]

'''
Anatomical similarity requires that the input data be provided as an image_list 
where len(image_list) = number of subjects
image_list[i].shape = (Slices x Height x Width x Channel)
Use an appropriate script to load data here. The section below is for illustration only
'''
 
wd = os.listdir(cfg.data_dir)
train_imgs = []
for subName in wd:
    sub_img = load_unl_brats_img(cfg.data_dir, subName, cfg.opShape,zmean_norm=cfg.zmean_norm)
    sub_img = np.transpose(sub_img, (2,0,1,3))
    train_imgs.append(sub_img)


'''
generate_global_0 : simCLR
generate_global_1: anat_sim
'''
if global_contrastive == 'anat_sim':
    train_generator = GL_generator.Data_Generator_Anatomical(train_imgs, cfg).generate_global_1()
else:
    train_generator = GL_generator.Data_Generator_Anatomical(train_imgs, cfg).generate_global_0()


initial_epoch = cfg.initial_epoch
num_iters = cfg.num_iters
train_step =  round(len(train_imgs) / (cfg.batch_size)  )                
num_epochs = int(num_iters/train_step)

if global_contrastive == 'anat_sim':
    customLoss = loss.anatomical_similarity_global
else:
    customLoss = loss.simCLR

ae = model.encoder_network(add_PH=True)
Adamopt = tf.keras.optimizers.Adam(learning_rate=cfg.lr_pretrain)
ae.compile(loss=customLoss, optimizer=Adamopt)

# Train the model
ae.fit(train_generator,
    steps_per_epoch=train_step,
    epochs=num_epochs,
    verbose=1,
    callbacks=callbacks,
    initial_epoch=initial_epoch,
    use_multiprocessing=False) 

 
# Save final weights
ae.save_weights(
    save_dir + 'weights_' + str(num_iters) +'.hdf5',
    overwrite=True,
    save_format='h5',
    options=None) 
