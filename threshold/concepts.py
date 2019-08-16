#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Description: 
# @File: concepts.py
# @Project: tf_learn
# @Author: Yiheng
# @Email: GuoYiheng89@gmail.com
# @Time: 8/16/2019 16:00

import os
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True, gpu_options=gpu_options)

x1 = tf.constant([1, 2, 3, 4])
x2 = tf.constant([5, 6, 7, 8])
result = tf.multiply(x1, x2)
print(f'x1 is {x1}')
print(f'result is {result}')

with tf.Session(config=config) as sess:
    print(f'sess x1 is {sess.run(x1)}')
    print(f'sess result is {sess.run(result)}')
