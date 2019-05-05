# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 13:18:56 2019

@author: hai huynh
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
exp_dir = os.getcwd()
sys.path.append(exp_dir)
from config import style_ln

def loss_curve(err_track):
    iter_, loss_train, loss_test = err_track
    plt.plot(iter_, loss_train, color='green', label='Training Error')
    plt.plot(iter_, loss_test, color='red', label='Testing Error')
    
    plt.title("Loss Curve")
    plt.xlabel("# iterations")
    plt.ylabel("Loss value")
    
    plt.legend()
    
    plt.show()
    

def loss_curves(err_track, type_err=[1,2]):
    def id_type_err(code):
        if type(type_err) is int:
            return type_err == code
        return code in type_err
    
    is_train = id_type_err(1)
    is_test = id_type_err(2)
    
    i_style = 0
    for model_name, records in err_track.items():
        iter_, loss_train, loss_test, loss_dev = records
        
        if is_train:
            plt.plot(iter_, loss_train, color='g', linestyle=style_ln[i_style], 
                     label='train ' + str(model_name))
        if is_test:
            plt.plot(iter_, loss_test, color='r', linestyle=style_ln[i_style], 
                     label='test ' + str(model_name))
        i_style = i_style + 1
        
    plt.title("Loss Curve")
    plt.xlabel("# iterations")
    plt.ylabel("Loss value")
    
    plt.legend()
    
    plt.show()
    
    
def heatmap_weight(weight):
    #plt.rcParams["figure.figsize"] = 5,5
    if weight.shape[0] == 1:
        weight = np.reshape(weight, (weight.shape[-1]))
        fig, (ax,ax2) = plt.subplots(nrows=2, sharex=True)
        
        x = np.arange(len(weight))
        extent = [x[0]-(x[1]-x[0])/2., x[-1]+(x[1]-x[0])/2.,0,1]
        
        ax.imshow(weight[np.newaxis,:], cmap="plasma", aspect="auto", extent=extent)
        ax.set_yticks([])
        ax.set_xlim(extent[0], extent[1])
        
        ax2.plot(x,weight)
    else:
        plt.imshow(weight, cmap='plasma', aspect='auto')
    
    plt.tight_layout()
    plt.show()

