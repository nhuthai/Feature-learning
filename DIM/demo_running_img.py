# -*- coding: utf-8 -*-
"""
Created on Sun May  5 22:50:24 2019

@author: hai huynh
"""

import config
import visual
import eval
import math

import tensorflow as tf
import numpy as np

from infomax import infomax
from preprocess import trans_space

def run(x_train, x_test):
    """
    Run model
    """
    # convert 3D to 4D
    x_train = trans_space(x_train)
    x_test = trans_space(x_test)
    size_img = x_train.shape[1]
    n_img = x_train.shape[0]
    
    # For error tracking
    err_track = {}
    
    # Set placeholders
    x_train_ph = tf.placeholder(tf.float32, [config.batch_size, size_img, size_img, 1])
    x_test_ph = tf.placeholder(tf.float32, [config.batch_size, size_img, size_img, 1])
    
    # Create models
    DIM_model = infomax(x_train_ph, x_test_ph)
    
    opt_ = DIM_model.optimize()
    # List for error tracking
    n_tracks = int(config.n_epochs / config.disp_step)
    losses_train = np.zeros(n_tracks)
    losses_test = np.zeros(n_tracks)
    losses_global = np.zeros(n_tracks)
    losses_local = np.zeros(n_tracks)
    losses_gen = np.zeros(n_tracks)
    losses_disc = np.zeros(n_tracks)
    i_epoch = np.zeros(n_tracks)
    id_ = 0
    err_track = [losses_train, losses_test, i_epoch,
                 losses_global, losses_local, losses_gen, losses_disc]
    # Start session
    with tf.Session() as sess:
        ## Initialize variables
        init = tf.global_variables_initializer()
        sess.run(init)
        ## Training each epoch by running optimizer
        for epoch in range(config.n_epochs):
            n_batches = math.ceil(n_img / config.batch_size)
            # Shuffle training dataset
            ids_ = np.arange(len(x_train))
            ids_ = np.random.permutation(ids_)
            s_y_train = x_train[ids_]
            
            for i_batch in range(n_batches):
                # Sampling into batch
                s = i_batch * config.batch_size
                e = min(len(x_train), s + config.batch_size - 1)
                m_x_train = s_y_train[s:e]
                
                if (epoch + 1) % config.disp_step != 0 or i_batch != n_batches - 1:
                    sess.run([opt_['gen_opt'], opt_['disc_opt']], 
                             feed_dict={DIM_model.x_train: m_x_train})
                ### At particular epoch, show loss of train and test, features of 
                ### train and test
                else:
                    _, _, gbl_feature, gbl_JSD, lcl_JSD,                \
                                gen_loss, disc_loss, test_feature =     \
                        sess.run([opt_['gen_opt'], opt_['disc_opt'], 
                                  opt_['global_feature'], 
                                  opt_['gbl_JSD'], opt_['lcl_JSD'],
                                  opt_['gen_loss'], opt_['disc_loss'],
                                  opt_['test_feature']],
                                feed_dict={DIM_model.x_train: m_x_train,
                                           DIM_model.x_test: x_test})
    
                    # Evaluate global feature
                    eval_train = eval.eval(gbl_feature)
                    eval_test = eval.eval(test_feature)
                        
                    i_epoch[id_] = epoch
                    losses_train[id_] = eval_train
                    losses_test[id_] = eval_test
                    losses_global[id_] = gbl_JSD
                    losses_local[id_] = lcl_JSD
                    losses_gen[id_] = gen_loss
                    losses_disc[id_] = disc_loss
                    id_ = id_ + 1
                    print('At epoch ' + str(epoch) + 'th,')
                    print('    Training evaluation: ' + str(eval_train))
                    print('    Testing evaluation: ' + str(eval_test))
                    print('    Global loss: ' + str(gbl_JSD))
                    print('    Local loss: ' + str(lcl_JSD))
                    print('    Generative loss: ' + str(gen_loss))
                    print('    Discriminative loss: ' + str(disc_loss))
                    
    ## Plot losses and evaluation
    print("Loss curves")
    visual.loss_curves(err_track[2:])
    print("Accuracy curves")
    visual.accuracy_curve(err_track[:3])
            
    return err_track
        