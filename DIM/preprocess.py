# -*- coding: utf-8 -*-
"""
Created on Sun May  5 23:34:15 2019

@author: hai huynh
"""

import numpy as np


def trans_space(df_: np.array):
    """
    Task: convert 3D array to 4D array, that feeds to neural network
    Args:
        df_: 3D array, should be numpy array
    Returns:
        df_: 4D array which is numpy array
    """
    if not df_ is None:
        # 3D to 4D
        df_ = np.array([df_])
        df_ = np.reshape(df_, (df_.shape[1], df_.shape[2], df_.shape[3], 1))
        
        return df_