# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 09:13:30 2019

@author: hai huynh
"""

import tensorflow as tf
import numpy as np
import math
import config
import visual

from preprocess import trans_space
from read_data import read_architecture
from conv_simple import conv_simple

def run(x_train, y_train, x_test, y_test):    
    """
    Run model
    """
    # convert 1D, 2D to 3D
    x_train = trans_space(x_train)
    y_train = trans_space(y_train)
    x_test = trans_space(x_test)
    y_test = trans_space(y_test)
    
    n_wavelength = config.hi_wl - config.lo_wl + 1
    # Get architecture
    achit = read_architecture()
    # For error tracking
    err_track = {}
    
    # Set placeholders
    x_train_ph = tf.placeholder(tf.float32, [None, n_wavelength, 1])
    y_train_ph = tf.placeholder(tf.float32, [None, 1, 1])
    x_test_ph = tf.placeholder(tf.float32, [None, n_wavelength, 1])
    y_test_ph = tf.placeholder(tf.float32, [None, 1, 1])
    
    # Create models
    conv_model = conv_simple(x_train_ph, y_train_ph, x_test_ph, y_test_ph, achit)
    
    opt_ = conv_model.optimize()
    # List for error tracking
    n_tracks = int(config.n_epochs / config.disp_step)
    losses_train = np.zeros(n_tracks)
    losses_test = np.zeros(n_tracks)
    i_epoch = np.zeros(n_tracks)
    id_ = 0
    err_track = [i_epoch, losses_train, losses_test]
    # Start session
    with tf.Session() as sess:
        ## Initialize variables
        init = tf.global_variables_initializer()
        sess.run(init)
        ## Training each epoch by running optimizer
        for epoch in range(config.n_epochs):
            n_batches = math.ceil(len(x_train) / config.batch_size)
            # Shuffle training dataset
            ids_ = np.arange(len(x_train))
            ids_ = np.random.permutation(ids_)
            s_x_train = x_train[ids_]
            s_y_train = y_train[ids_]
            
            for i_batch in range(n_batches):
                # Sampling into batch
                s = i_batch * config.batch_size
                e = min(len(s_x_train), s + config.batch_size - 1)
                m_x_train = s_x_train[s:e]
                m_y_train = s_y_train[s:e]
                
                if (epoch + 1) % config.disp_step != 0 or i_batch != n_batches - 1:
                    sess.run(opt_['train_op'], feed_dict={conv_model.x_train: m_x_train,
                                                         conv_model.y_train: m_y_train})
                ### At particular epoch, show loss of train and test, features of 
                ### train and test
                else:
                    _, loss_train, loss_test, train_fm, test_fm, w_att =     \
                        sess.run([opt_['train_op'], opt_['loss_train'], 
                                  opt_['loss_test'], opt_['train_features'],
                                  opt_['test_features'], conv_model.similar],
                                feed_dict={conv_model.x_train: m_x_train,
                                           conv_model.y_train: m_y_train,
                                           conv_model.x_test: x_test,
                                           conv_model.y_test: y_test})
                    visual.heatmap_weight(np.array(w_att)[:,0,0,:])
                        
                    i_epoch[id_] = epoch
                    losses_train[id_] = loss_train
                    losses_test[id_] = loss_test
                    id_ = id_ + 1
                    print('At epoch ' + str(epoch) + 'th,')
                    print('    Training loss: ' + str(loss_train))
                    print('    Testing loss: ' + str(loss_test))
                    
    ## Plot train and test loss
    visual.loss_curves(err_track)
            
    return err_track