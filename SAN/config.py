# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 09:29:25 2019

@author: hai huynh
"""

"""
Read data
"""
archi_fname = 'architecture.json'

"""
Preprocessing
"""
lo_wl = 350
hi_wl = 800

extrem_norm = 1e-3

"""
conv simple
"""
l_rate = 1e-3
n_epochs = 100
disp_step = 10
drop = 0.3

"""
attention
"""
width_san = 400
step_san = 100

"""
training
"""
batch_size = 200

"""
Visualization
"""
style_ln = ['-', '--', '-.', '.']