#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Description: 
# @File: traffic_signs.py
# @Project: tf_learn
# @Author: Yiheng
# @Email: GuoYiheng89@gmail.com
# @Time: 8/19/2019 17:42
import os


def is_dir(parent_path, son_path):
    full_path = os.path.join(parent_path, son_path)
    return os.path.isdir(full_path)


def list_dir(parent_path, son_path=None):
    full_path = parent_path if son_path is None else os.path.join(parent_path, son_path)
    return os.listdir(full_path)


def load_data(data_dir):
    datas = [(int(son_file), grand_son) for son_file in list_dir(data_dir) if is_dir(data_dir, son_file)
             for grand_son in list_dir(data_dir, son_file) if grand_son.endswith('ppm')]
    """
    for img, label in datas:
        print(f'{img}, {label}')
    """

    return zip(*datas)


if __name__ == '__main__':
    img_root_path = 'E:/tf_test'
    train_dir = os.path.join(img_root_path, 'Training')
    test_dir = os.path.join(img_root_path, 'Testing')

    train_labels, train_images = load_data(train_dir)
