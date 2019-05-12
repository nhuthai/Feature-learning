# -*- coding: utf-8 -*-
"""
Created on Sat May  4 10:54:15 2019

@author: hai huynh
"""

import block
import config
import tensorflow as tf
import loss_func as losses

class infomax:
    def __init__(self, x_tf):
        self.x_tf = x_tf
        # if training 0.3, else 1
        self.drop = tf.placeholder(tf.float32)
    
    
    def model(self):
        fed = self.x_tf
        
        returns_ = {}
        shared_map = block.convNet(fed)
        
        """
        image-oriented(global objective)
        """
        # Create global feature
        gbl_gbl = block.fc_global(shared_map, drop=self.drop,
                                  name='Generator/fc_global/global/')
        # Create local feature
        gbl_lcl = block.flatten_fm(shared_map)
        # Create score map
        gbl_score_map = block.score_map(gbl_gbl, gbl_lcl)
        returns_['global_score'] = gbl_score_map
        
        """
        patch-oriented(local objective)
        """
        # Create global feature
        lcl_gbl = block.fc_global(gbl_gbl, drop=self.drop,
                                  name='Generator/fc_global/local/')
        returns_['global_feature'] = lcl_gbl
        # Create local feature
        lcl_lcl = block.conv_1x1(shared_map)
        lcl_lcl = block.flatten_fm(lcl_lcl)
        # Create score map
        lcl_score_map = block.score_map(lcl_gbl, lcl_lcl)
        returns_['local_score'] = lcl_score_map
        
        """
        GAN
        """
        # Create shuffled (fake) feature
        real_fm = lcl_gbl
        fake_fm = block.permutation_batch(real_fm)
        # Calculate real probability
        real_prob = block.fc_disc(real_fm, drop=self.drop)
        returns_['real_score'] = real_prob
        # Calculate fake probability
        fake_prob = block.fc_disc(fake_fm, drop=self.drop, reuse=True)
        returns_['fake_score'] = fake_prob
        
        return returns_
    
    
    def optimize(self):
        """
        Train
        """
        # Get variables
        gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                     scope="Generator")
        disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                      scope="Discriminator")
        
        # Run model
        out_model = self.model()
        global_feature = out_model['global_feature']
        # Loss functions
        gbl_JSD = losses.JSD_loss(out_model['global_score'])
        lcl_JSD = losses.JSD_loss(out_model['local_score'])
        GAN_losses = losses.F_GAN_loss(out_model['real_score'],
                                       out_model['fake_score'])
        gen_loss = config.gan_scale*GAN_losses['gen_loss'] + \
                            config.gbl_scale*gbl_JSD + config.lcl_scale*lcl_JSD
        disc_loss = GAN_losses['disc_loss']
        
        # Optimize
        gen_opt = tf.train.AdamOptimizer(learning_rate=config.l_rate) \
                                        .minimize(gen_loss, var_list=gen_vars)
        disc_opt = tf.train.AdamOptimizer(learning_rate=config.l_rate) \
                                        .minimize(disc_loss, var_list=disc_vars)
        
        return {'global_feature': global_feature,
                'gbl_JSD': gbl_JSD, 'lcl_JSD': lcl_JSD, 
                'gen_loss': gen_loss, 'disc_loss': disc_loss,
                'gen_opt': gen_opt, 'disc_opt': disc_opt}