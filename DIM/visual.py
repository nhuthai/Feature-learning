# -*- coding: utf-8 -*-
"""
Created on Sun May  5 23:41:27 2019

@author: hai huynh
"""

import matplotlib.pyplot as plt

def loss_curves(losses, type_err=[1,2,3,4]):
    def id_type_err(code):
        if type(type_err) is int:
            return type_err == code
        return code in type_err
    
    is_global = id_type_err(1)
    is_local = id_type_err(2)
    is_gen = id_type_err(3)
    is_disc = id_type_err(4)
    
    iter_, loss_global, loss_local, loss_gen, loss_disc = losses
        
    if is_global:
        plt.plot(iter_, loss_global, color='g', label='global')
    if is_local:
        plt.plot(iter_, loss_local, color='r', label='local')
    if is_gen:
        plt.plot(iter_, loss_gen, color='b', label='gen')
    if is_disc:
        plt.plot(iter_, loss_disc, color='y', label='disc')
        
    plt.title("Loss Curve")
    plt.xlabel("# iterations")
    plt.ylabel("Loss value")
    
    plt.legend()
    
    plt.show()


def accuracy_curve(accr, type_err=[1,2]):
    def id_type_err(code):
        if type(type_err) is int:
            return type_err == code
        return code in type_err
    
    is_train = id_type_err(1)
    is_test = id_type_err(2)
    
    loss_train, loss_test, iter_ = accr
        
    if is_train:
        plt.plot(iter_, loss_train, color='g', label='global')
    if is_test:
        plt.plot(iter_, loss_test, color='r', label='local')
        
    plt.title("Loss Curve")
    plt.xlabel("# iterations")
    plt.ylabel("Loss value")
    
    plt.legend()
    
    plt.show()