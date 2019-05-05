# -*- coding: utf-8 -*-
"""
Created on Sun May  5 00:20:47 2019

@author: hai huynh
"""

import tensorflow as tf

"""
Settings
"""
feature_depth = 32
i_lcl_out = 1 # index of layer which feature maps is used for local objective
drop = 0.3

"""
Loss
"""
gbl_scale = 1.
lcl_scale = 1.
gan_scale = 1.

"""
convNet
"""
convNet_arg = {1: {'n_hid': 16, 'kernel': (3,3), 'strides': (2,2), 
                     'activation': tf.nn.relu},
               2: {'n_hid': feature_depth, 'kernel': (3,3), 'strides': (2,2), 
                     'activation': None}}

"""
conv_1x1
"""
conv_1x1_arg = [{1: {'n_hid': feature_depth, 'kernel': (1,1),
                     'activation': tf.nn.relu},
                 2: {'n_hid': feature_depth, 'kernel': (1,1)}, 
                     'activation': None},
                {1: {'n_hid': feature_depth, 'kernel': (1,1), 
                     'activation': tf.nn.relu}}]

"""
fc_global
"""
fc_global_arg = [{1: {'n_hid': feature_depth, 'drop': drop,
                      'activation': tf.nn.relu},
                 2: {'n_hid': feature_depth, 'drop': drop,
                      'activation': None}},
                {1: {'n_hid': feature_depth, 'drop': drop,
                      'activation': tf.nn.relu}}]

"""
fc_disc
"""
fc_disc_arg = {1: {'n_hid': 32, 'drop': drop,
                   'activation': tf.nn.relu},
               2: {'n_hid': 16, 'drop': drop,
                   'activation': None},
               3: {'n_hid': 1, 'drop': drop,
                   'activation': tf.nn.relu}}
               
"""
training
"""
batch_size = 30
n_epochs = 100
disp_step = 10