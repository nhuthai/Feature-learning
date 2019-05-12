# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 15:28:02 2019

@author: hai huynh
"""

import tensorflow as tf

from attention import attention
from config import l_rate


class conv_simple:
    def __init__(self, x_tf, y_tf):
        self.x_train = x_tf
        self.y_train = y_tf
        # if training: 0.3 else: 1
        self.keep_prob = tf.placeholder(tf.float32)
    
        
    def model(self):
        fed = self.x_train
        
        returns_ = {}
        
        """
        Feature extraction
        """
        args = {"n_hid_channel": 1, "n_in_channel": 1, "n_out_channel": 1, 
                "n_heads": 5}
        high_feature, self.similar = attention(fed, args, drop=self.keep_prob)
        
        """
        Convolutional Net
        """
        with tf.variable_scope('conv'):
            conv = tf.layers.conv1d(high_feature, 16, 3, strides=2)
            bn = tf.layers.batch_normalization(conv)
            lrelu = tf.nn.leaky_relu(bn)
            pool = tf.layers.average_pooling1d(lrelu, 2, 2)
            
            returns_['layer_1'] = pool
        
        """
        Regression
        """
        vtor = tf.layers.flatten(pool)
        
        regress = tf.layers.dense(vtor, 1)
        regress = tf.layers.dropout(regress, rate=self.keep_prob)
        
        returns_['regression'] = regress
        
        return returns_
        
        
    def optimize(self):
        """
        Optimization
        """
        pred_ = self.model()
        loss_ = tf.reduce_mean(tf.pow(pred_['regression'] - self.y_train, 2)) / 2
        optimizer = tf.train.AdamOptimizer(learning_rate=l_rate)
        train_op = optimizer.minimize(loss_)
        
        return {'train_op': train_op, 
                'loss': loss_,
                'features': pred_}