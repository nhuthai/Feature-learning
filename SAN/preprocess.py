# -*- coding: utf-8 -*-
"""
Created on Sat May  4 09:40:33 2019

@author: hai huynh
"""

import numpy as np


def trans_space(df_: np.array):
    """
    Task: convert 1D or 2D array to 3D array, that feeds to neural network
    Args:
        df_: 1D or 2D array, should be numpy array
    Returns:
        df_: 3D array which is numpy array
    """
    if not df_ is None:
        # 2D to 3D
        if len(df_.shape) > 1:
            df_ = np.array([df_])
            df_ = np.reshape(df_, (df_.shape[1], df_.shape[2], 1))
        # 1D to 3D
        else:
            df_ = np.array([[df_.values]])
            df_ = np.reshape(df_, (df_.shape[2], df_.shape[1], 1))
        
        return df_