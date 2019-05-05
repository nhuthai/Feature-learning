# -*- coding: utf-8 -*-
"""
Created on Sat May  4 23:55:52 2019

@author: hai huynh
"""

import tensorflow as tf
import config

def convNet(in_, name='Generator/convNet/', reuse=False):
    fm_ = in_
    lcl_out = in_
    with tf.variable_scope(name, reuse=reuse):
        for i_layer, args in config.convNet_arg.items():
            fm_ = tf.layers.conv2d(fm_, args['n_hid'], args['kernel'], 
                                   strides=args['strides'])
            if i_layer <= config.i_lcl_out:
                lcl_out = fm_
                
    return lcl_out, fm_


def conv_1x1(in_, name='Generator/conv_1x1/', reuse=False):
    fm_ = None
    with tf.variable_scope(name, reuse=reuse):
        for branch in config.conv_1x1_arg:
            fed = in_
            for i_layer, args in branch.items():
                fed = tf.layers.conv2d(fed, args['n_hid'], args['kernel'], 
                                       activation=args['activation'])
            if fm_ is None:
                fm_ = fed
            else:
                fm_ = tf.add(fm_, fed)
        
        fm_ = tf.layers.batch_normalization(fm_)
    return fm_


def fc_global(in_, name='Generator/fc_global/', is_train=True, reuse=False):
    fm_ = None
    with tf.variable_scope(name, reuse=reuse):
        for branch in config.fc_global_arg:
            fed = in_
            for i_layer, args in branch.items():
                fed = tf.layers.dense(fed, args['n_hid'], activation=args['activation'])
                fed = tf.layers.dropout(fed, rate=args['drop'], training=is_train)
                
            if fm_ is None:
                fm_ = fed
            else:
                fm_ = tf.add(fm_, fed)
    
    return fm_


def fc_disc(in_, name='Discriminator/', is_train=True, reuse=False):
    prob = in_
    with tf.variable_scope(name, reuse=reuse):
        for i_layer, args in config.fc_disc_arg.items():
            prob = tf.layers.dense(prob, args['n_hid'], activation=args['activation'])
            prob = tf.layers.dropout(prob, rate=args['drop'], training=is_train)
        
    return prob


def flatten_fm(fm_):
    return tf.reshape(fm_, shape=(config.batch_size, -1, config.feature_depth))


def permutation_batch(batch):
    new_batch = tf.reshape(batch, (config.batch_size, config.feature_depth))
    shuffle = map(lambda i_depth: tf.random_shuffle(new_batch[:,i_depth]), 
            range(config.feature_depth))
    shuffle = tf.concat(list(shuffle), axis=1)
    
    return shuffle


def score_map(gbl, lcl):
    # reshape: (batch_size*n_vector) x depth
    ## reshape: (batch_size*n_vector) x depth
    gbl_2D = tf.reshape(gbl, shape=(-1, config.feature_depth))
    lcl_2D = tf.reshape(lcl, shape=(-1, config.feature_depth))
    # dot product
    score_mtx = tf.matmul(gbl_2D, lcl_2D, transpose_b=True)
    # reshape: batch_size x batch_size x n_global_vector x n_local_vector
    score_mtx_3D = tf.reshape(score_mtx, shape=(config.batch_size, config.batch_size,
                                                -1))
    
    return score_mtx_3D
    