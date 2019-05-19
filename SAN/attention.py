# -*- coding: utf-8 -*-
"""
Created on Sat May  4 10:34:19 2019

@author: hai huynh
"""

import tensorflow as tf
import math

from functools import reduce
from config import step_san, width_san, hi_wl, lo_wl

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
        patches = {}
        n_wavelengths = hi_wl - lo_wl + 1
        """
        Attention
        """
        # extract as patches [batch, channel, patch, length]
        patches['key'] = tf.extract_image_patches(feed['key'], 
                                           ksizes=[1,1,width_san,1], 
                                           strides=[1,1,step_san,1], 
                                           rates=[1,1,1,1], 
                                           padding='VALID')
        patches['query'] = tf.extract_image_patches(feed['query'], 
                                           ksizes=[1,1,width_san,1], 
                                           strides=[1,1,step_san,1], 
                                           rates=[1,1,1,1], 
                                           padding='VALID')
        patches['value'] = tf.extract_image_patches(feed['value'], 
                                           ksizes=[1,1,width_san,1], 
                                           strides=[1,1,step_san,1], 
                                           rates=[1,1,1,1], 
                                           padding='VALID')
        # Change dimensions [patch, batch, length, channel]
        patches['key'] = tf.transpose(patches['key'], perm=[2, 0, 3, 1])
        patches['query'] = tf.transpose(patches['query'], perm=[2, 0, 3, 1])
        patches['value'] = tf.transpose(patches['value'], perm=[2, 0, 3, 1])
           
        # doing
        att = tf.map_fn(lambda x: one_head(x['key'], x['query'], x['value']), 
                        patches, name='map_attention')
        
        """
        Intergration
        """
        def shift(in_, i):
            return tf.manip.roll(in_, shift=i[0], axis=1)
        
        def sum_up(prev, in_):
            with tf.variable_scope("sum_up", reuse=tf.AUTO_REUSE):                
                linear = tf.layers.dense(prev, units=1) + in_
                
            return linear
        
        # Padding
        pad = tf.zeros([tf.shape(att)[0], tf.shape(att)[1], 
                        n_wavelengths - step_san, 1], name='pad_zeros')
        pad_att_ = tf.concat([att, pad], 2)
        i_pad = tf.expand_dims(tf.range(tf.shape(pad_att_)[0], dtype=tf.int32), 1)
        pad_att, _ = tf.map_fn(lambda x: (shift(x[0], x[1]), x[1]), (pad_att_, i_pad),
                               name='padding')
        # Sum-up
        re = tf.scan(sum_up, pad_att, name='weighted_sum')
        new_feature = re[-1]
        
        return new_feature
    
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