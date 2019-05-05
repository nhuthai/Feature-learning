# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 15:28:02 2019

@author: hai huynh
"""

import tensorflow as tf

from attention import attention
from config import l_rate


class conv_simple:
    def __init__(self, x_train_tf, y_train_tf, x_test_tf, y_test_tf):
        self.x_train = x_train_tf
        self.y_train = y_train_tf
                
        self.x_test = x_test_tf
        self.y_test = y_test_tf
    
        
    def model(self, is_train=True):
        if is_train:
            fed = self.x_train
        else:
            fed = self.x_test
            
        reuse = not is_train
        
        returns_ = {}
        
        """
        Feature extraction
        """
        args = {"n_hid_channel": 1, "n_in_channel": 1, "n_out_channel": 1, 
                "n_heads": 5, "drop": 0.3}
        high_feature, self.similar = attention(fed, args, reuse, train=is_train)
        
        """
        Convolutional Net
        """
        with tf.variable_scope('conv', reuse=reuse):
            conv = tf.layers.conv1d(high_feature, 16, 3, strides=2)
            bn = tf.layers.batch_normalization(conv, training=is_train)
            lrelu = tf.nn.leaky_relu(bn)
            pool = tf.layers.average_pooling1d(lrelu, 2, 2)
            
            returns_['layer_1'] = pool
        
        """
        Regression
        """
        vtor = tf.layers.flatten(pool)
        
        regress = tf.layers.dense(vtor, 1)
        regress = tf.layers.dropout(regress, rate=0.3, training=is_train)
        
        returns_['regression'] = regress
        
        return returns_
        
        
    def optimize(self):
        """
        Optimization
        """
        train_pred = self.model()
        loss_op = tf.reduce_mean(tf.pow(train_pred['regression'] - self.y_train, 2)) / 2
        optimizer = tf.train.AdamOptimizer(learning_rate=l_rate)
        train_op = optimizer.minimize(loss_op)
        
        """
        Validation
        """        
        test_pred = self.model(is_train=False)
        loss_test = tf.reduce_mean(tf.pow(test_pred['reg'] - self.y_test, 2)) / 2
        
        return {'train_op': train_op, 
                'loss_train': loss_op, 'loss_test': loss_test,
                'train_features': train_pred, 'test_features': test_pred}