# -*- coding: utf-8 -*-
"""
Created on Sat May  4 10:34:19 2019

@author: hai huynh
"""

import tensorflow as tf
import math

from functools import reduce

def attention(in_, args, **kwargs):
    fed = {}
    if 'key' in kwargs or 'query' in kwargs:
        fed['value'] = in_
        fed['query'] = kwargs['query']
        fed['key'] = kwargs['key']
    else:
        fed['key'] = in_
        fed['query'] = in_
        fed['value'] = in_
    is_train = kwargs['train'] if 'train' in kwargs else False
    
    """
    Single head
    """
    def one_head(key, query, value, bias=None):
        # linear transformation
        trans_k = tf.layers.dense(key, args['n_hid_channel'])
        trans_q = tf.layers.dense(query, args['n_hid_channel'])
        trans_v = tf.layers.dense(value, args['n_in_channel'])
        
        # find dependencies using multiply of keys and queries
        depend = tf.matmul(trans_q, trans_k, transpose_b=True)
        
        # scaled and bias
        scal_coeff = 1 / math.sqrt(args['n_hid_channel'])
        scal_depend = tf.scalar_mul(scal_coeff, depend)
        scal_depend = scal_depend + bias if not bias is None else scal_depend
        
        # find similarity using softmax of dependencies
        similarity = tf.nn.softmax(scal_depend)
        
        # find head by multiply similarity by values
        head = tf.matmul(similarity, trans_v)
        
        return [head, similarity]
    
    """
    Limited Attention
    """
    def limit_attention(prev, feed):
        def preprocess_patches(keyword):
            # extract as patches [batch, channel, patch, length]
            patches = tf.extract_image_patches(feed[keyword], ksizes=[1,1,3,1], 
                                               strides=[1,1,1,1], rates=[1,1,1,1], 
                                               padding='VALID')
            # Change dimensions [patch, batch, length, channel]
            patches = tf.transpose(patches, perm=[2, 0, 3, 1])
            # Add dummy
            first = patches[0, :, :, :]
            first = tf.reshape(first, (1, -1, 3, 1))
            new_patches = tf.concat([first, patches], 0)
            
            return new_patches
           
        # doing
        att = tf.scan(one_head, new_patches)
        # eliminate dummy
        att = att[1:, :, :, :]
        
        return one_head(feed['key'], feed['query'], feed['value'])
    
    """
    Multi-head
    """
    def concate(x, y):
        return tf.concat([x, y], 1)
    
    with tf.variable_scope('attention'):
        # Layer norm
        fed['key'] = tf.contrib.layers.layer_norm(fed['key'])
        fed['query'] = tf.contrib.layers.layer_norm(fed['query'])
        fed['value'] = tf.contrib.layers.layer_norm(fed['value'])
        # Attention blocks
        meta_heads = map(lambda x: one_head(fed['key'], fed['query'], fed['value']), 
                          range(args['n_heads']))
        #meta_heads = map(lambda x: one_head(fed), range(args['n_heads']))
        meta_heads = list(map(list, zip(*meta_heads)))
        heads = meta_heads[0]
        similarity = meta_heads[-1]
        # Concat heads and linear transformation (in order to use dropout)
        multihead = reduce(concate, heads)
        multihead = tf.layers.dense(multihead, args['n_in_channel'])
        multihead = tf.layers.dropout(multihead, rate=kwargs['drop'])
        
        # Residual
        residual = tf.tile(fed['value'], [1,args['n_heads'],1])
        fed_reg = tf.add(multihead, residual)
            
        # Linear regression block
        # Layer norm
        fed_reg = tf.contrib.layers.layer_norm(fed_reg)
        # ReLU
        hid = tf.layers.dense(fed_reg, args['n_out_channel'], activation=tf.nn.leaky_relu)
        # Linear transformation (in order to use dropout)
        hid = tf.layers.dense(hid, args['n_out_channel'])
        hid = tf.layers.dropout(hid, rate=kwargs['drop'], training=is_train)
        
        # Residual
        new_representation = tf.add(hid, fed_reg)
    
    return new_representation, similarity