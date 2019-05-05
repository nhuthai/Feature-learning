# -*- coding: utf-8 -*-
"""
Created on Sun May  5 11:04:52 2019

@author: hai huynh
"""

import tensorflow as tf
import math
import config


def softplus(z):
    # log(1+e^{z})
    return tf.log(tf.add(1., tf.exp(z)))


def JSD_loss(score_mtx):
    # loss = avg(E[-sp(-pos)]) - avg(E[sp(-neg) + neg])
    # E is expectation over all scores for each pair
    # avg is average over pairs for each batch 
    
    ne_score_mtx = tf.negative(score_mtx)
    extreme = math.log(2)
    
    # for positive
    ## calculate softplus
    pos_mi = tf.negative(softplus(ne_score_mtx))
    ## shift mutual information
    shifted_pos = tf.add(extreme, pos_mi)
    ## Expectation
    expected_pos = tf.reduce_mean(shifted_pos, axis=2)
    ## average
    ### filter positive pair using positive mask
    pos_mask = tf.eye(config.batch_size)
    avg_pos = tf.multiply(expected_pos, pos_mask)
    ### sum and then avg each batch
    avg_pos = tf.divide(tf.reduce_sum(avg_pos, axis=1), tf.reduce_sum(pos_mask))
    
    # for negative
    ## calculate softplus
    neg_mi = softplus(ne_score_mtx) + score_mtx
    ## shift mutual information
    shifted_neg = tf.add(neg_mi, -extreme)
    ## Expectation
    expected_neg = tf.reduce_mean(shifted_neg, axis=2)
    ## average
    neg_mask = tf.add(1.,tf.negative(tf.eye(config.batch_size)))
    avg_neg = tf.multiply(expected_neg, neg_mask)
    ### sum and then avg each batch
    avg_neg = tf.divide(tf.reduce_sum(avg_neg, axis=1), tf.reduce_sum(neg_mask))
    
    # JSD loss: avg_neg - avg_pos (batch_size x 1)
    # get negative since convert maximization to minimization
    loss = tf.add(tf.negative(avg_pos), avg_neg)
    
    return loss


def F_GAN_loss(real_prob_map, fake_prob_map):
    # overall loss: E(g(.)) + E(-g(.)) = E(-sp(.)) + E(-[-sp(.)])
    # disc. opt.: max{ E(-sp(.)) + E(-[-sp(.)]) } ~ min{ -E(-sp(.)) - E(-[-sp(.)]) }
    # gen. opt.: min{ E(-[-sp(.)]) }
    # E is expectation over all scores
    
    ne_real_prob = tf.negative(real_prob_map)
    ne_fake_prob = tf.negative(fake_prob_map)
    
    # for generator loss
    ## calculate softplus
    gen_loss = softplus(ne_fake_prob)
    ## Expectation
    gen_loss = tf.reduce_mean(gen_loss)
    
    # for discriminator loss
    ## calculate softplus
    real_score = softplus(ne_real_prob)
    fake_score = softplus(ne_fake_prob) + fake_prob_map
    ## Expectation
    disc_loss = tf.reduce_mean(real_score) - tf.reduce_mean(fake_score)
    
    return {'gen_loss': gen_loss, 'disc_loss': disc_loss}