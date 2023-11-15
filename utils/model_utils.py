'''
@author: lavanya
2D U-Net Architectures for pretraining and finetuning
'''

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, Dense
from tensorflow.keras.layers import UpSampling2D, Softmax  
from tensorflow.keras.layers import Concatenate, Activation
from tensorflow.keras.layers import Flatten 
from tensorflow.keras.layers import MaxPooling2D
from tensorflow_addons.layers import  GroupNormalization 
 
 

class modelObj:
    def __init__(self, cfg, kernel_init=None):
        self.img_size_x   = cfg.img_size_x
        self.img_size_y   = cfg.img_size_y
        self.num_channels = cfg.num_channels 
        self.latent_dim   = cfg.latent_dim   # representational space dim
        self.conv_kernel  = (3,3)
        self.no_filters   = [1, 16, 32, 64, 128, 128]
        if kernel_init == None:
            self.kernel_init = tf.keras.initializers.HeNormal(seed=1)
        else:
            self.kernel_init = kernel_init
        
    def encoder_block_contract(self, block_input, num_fts, pool_flag=True, block_name=1):        
        ''' Defining a UNET block in the feature downsampling path '''
        conv_kernel = self.conv_kernel
        num_groups = num_fts // 4
        if pool_flag:
            block_input = MaxPooling2D(pool_size=(2,2), name='enc_mp_' + str(block_name))(block_input)
            
        down = Conv2D(num_fts, conv_kernel, padding='same', name = 'enc_conv1_'+ str(block_name), kernel_initializer=self.kernel_init)(block_input)
        down = GroupNormalization(groups = num_groups, name = 'enc_gn1_'+ str(block_name))(down)
        down = Activation('relu',name = 'enc_act1_'+ str(block_name))(down)
        down = Conv2D(num_fts, conv_kernel, padding='same', name = 'enc_conv2_'+ str(block_name), kernel_initializer=self.kernel_init)(down)
        down = GroupNormalization(groups = num_groups, name = 'enc_gn2_'+ str(block_name))(down)
        down = Activation('relu', name = 'enc_act2_'+ str(block_name ))(down)
        return down

    def decoder_block_expand(self, block_input, numFts, concat_block, upsample_flag=True, block_name=1):
        '''Defining a UNET block in the feature upsampling path '''
        conv_kernel = self.conv_kernel
        num_groups = numFts // 4
        if upsample_flag:
            block_input = UpSampling2D(size=(2,2),name='dec_upsamp_'+ str(block_name))(block_input)
            block_input = Conv2D(numFts, kernel_size=(2,2),padding='same', name = 'dec_upsamp_conv_'+ str(block_name), kernel_initializer=self.kernel_init)(block_input)
            block_input = GroupNormalization(groups = num_groups, name = 'dec_upsamp_gn_'+ str(block_name))(block_input)
            block_input = Activation('relu', name = 'dec_upsamp_act_'+ str(block_name ))(block_input)
            block_input = Concatenate(axis=-1, name = 'dec_concat_'+ str(block_name))([block_input, concat_block])
        up = Conv2D(numFts,conv_kernel,padding='same', name = 'dec_conv1_'+ str(block_name), kernel_initializer=self.kernel_init)(block_input)
        up = GroupNormalization(groups = num_groups, name = 'dec_gn1_'+ str(block_name))(up)
        up = Activation('relu', name = 'dec_act1_'+ str(block_name ))(up)
        up = Conv2D(numFts,conv_kernel,padding='same', name = 'dec_conv2_'+ str(block_name), kernel_initializer=self.kernel_init)(up)
        up = GroupNormalization(groups = num_groups, name = 'dec_gn2_'+ str(block_name))(up)
        up = Activation('relu', name = 'dec_act2_'+ str(block_name))(up)
        return up

    
    def encoder_network(self, inputs=None, list_return=0, add_PH=False, 
                        name='enc_model'):
        ''' Define the encoder network: Encoder architecture similar to Global Local CL by Chaitanya et al.'''
        no_filters = self.no_filters
        #layers list for skip connections
        layers_list=[]
        # Level 1
        if inputs == None:
            inputs = Input((self.img_size_x, self.img_size_y, self.num_channels))
            
        enc_c1 = self.encoder_block_contract(inputs, no_filters[1], pool_flag=False, block_name=1)
        enc_c2 = self.encoder_block_contract(enc_c1, no_filters[2], block_name=2)
        enc_c3 = self.encoder_block_contract(enc_c2, no_filters[3], block_name=3)
        enc_c4 = self.encoder_block_contract(enc_c3, no_filters[4], block_name=4)
        enc_c5 = self.encoder_block_contract(enc_c4, no_filters[5], block_name=5)
        enc_c6 = self.encoder_block_contract(enc_c5, no_filters[5], block_name=6)
        
        layers_list.append(enc_c1)
        layers_list.append(enc_c2)
        layers_list.append(enc_c3)
        layers_list.append(enc_c4)
        layers_list.append(enc_c5)
        
        if add_PH:
            '''Encoder network with a non-linear projection head (MLP)'''
            PH_flat = Flatten()(enc_c6)
            PH_a = Dense(1024, name = 'PH_a', activation='relu', use_bias=False)(PH_flat)
            PH_b = Dense(128, name = 'PH_b', activation=None, use_bias=False)(PH_a) 
            model = Model(inputs=[inputs], outputs=[PH_b], name=name)
        else:       
            model = Model(inputs=[inputs], outputs=[enc_c6], name=name)
       
        if(list_return==1):
            return model, layers_list
        else:
            return model
      
    def encoder_decoder_network(self, num_dec_levels=5, enc_pretr_wts=None, 
                                enc_freeze=False, add_PH=False, name='dec_model', PH_str = ''):
        
        no_filters = self.no_filters  
        latent_dim = self.latent_dim   
        num_groups = latent_dim // 4        
        inputs = Input((self.img_size_x, self.img_size_y, self.num_channels))
        enc_model,enc_layers_list  = self.encoder_network(inputs,list_return=1)
        
        if enc_pretr_wts is not None:
            print('Loading pretrained_weights for encoder, matching by name')
            enc_model.load_weights(enc_pretr_wts, by_name  = True)
        enc_c6 = enc_model.output
       
        layer_idx = len(no_filters)-1
        tmp_dec = self.decoder_block_expand(enc_c6, no_filters[layer_idx], enc_layers_list[layer_idx-1], block_name=layer_idx)        
        for dec_layer in range(1,num_dec_levels):
            print('Generating dec layer ', dec_layer)
            layer_idx = layer_idx-1
            tmp_dec = self.decoder_block_expand(tmp_dec, no_filters[layer_idx], enc_layers_list[layer_idx-1], block_name=layer_idx)
        
        
        if add_PH:  
            '''Decoder network with a non-linear projection head (MLP)'''
            PH_A = Conv2D(latent_dim,(1,1),padding='same', use_bias=False, name='PH_A_conv1'+PH_str, kernel_initializer=self.kernel_init)(tmp_dec)        
            PH_A = GroupNormalization(groups = num_groups, name = 'PH_A_bn1'+PH_str)(PH_A)
            PH_A = Activation('relu', name = 'PH_A_act1'+PH_str)(PH_A)
            PH_B = Conv2D(latent_dim,(1,1),padding='same', use_bias=False, name='PH_B1'+PH_str, kernel_initializer=self.kernel_init)(PH_A)        
            enc_dec = Model(inputs=[inputs], outputs=[PH_B], name=name)
        else:   
            enc_dec = Model(inputs=[inputs], outputs=[tmp_dec], name=name) 
        
        if enc_freeze:
            print('Freezing all encoder weights, except Batch Normalization')            
            for layer in enc_dec.layers:
                layer_name = layer.name
                if 'enc' in layer_name:                  
                    layer.trainable = False 
        return enc_dec
    
    def seg_unet(self, num_classes):
        'Network for downstream segmentation tasks'
        no_filters = self.no_filters  
        inputs = Input((self.img_size_x, self.img_size_y, self.num_channels))

        ###################################
        # Encoder network
        ###################################
        enc_c1 = self.encoder_block_contract(inputs, no_filters[1], pool_flag=False, block_name=1)
        enc_c2 = self.encoder_block_contract(enc_c1, no_filters[2], block_name=2)
        enc_c3 = self.encoder_block_contract(enc_c2, no_filters[3], block_name=3)
        enc_c4 = self.encoder_block_contract(enc_c3, no_filters[4], block_name=4)
        enc_c5 = self.encoder_block_contract(enc_c4, no_filters[5], block_name=5)
        enc_c6 = self.encoder_block_contract(enc_c5, no_filters[5], block_name=6)

        ###################################
        # Decoder network - Upsampling Path
        ###################################
        dec_c5 = self.decoder_block_expand(enc_c6, no_filters[5], enc_c5, block_name=5)        
        dec_c4 = self.decoder_block_expand(dec_c5, no_filters[4], enc_c4, block_name=4)
        dec_c3 = self.decoder_block_expand(dec_c4, no_filters[3], enc_c3, block_name=3)
        dec_c2 = self.decoder_block_expand(dec_c3, no_filters[2], enc_c2, block_name=2)
        dec_c1 = self.decoder_block_expand(dec_c2, no_filters[1], enc_c1, block_name=1)
            
        model_op = Conv2D(16,3,padding='same', kernel_initializer=self.kernel_init)(dec_c1)
        model_op = GroupNormalization(groups = 4)(model_op)
        seg_op2 = Activation('relu')(model_op)
        seg_op = Conv2D(num_classes , 1, name='seg_layer', padding='same', use_bias=False, kernel_initializer=self.kernel_init)(seg_op2)
        seg_fin_layer = Softmax()(seg_op)
        model = Model(inputs=[inputs], outputs=[seg_fin_layer])
        return model
    
