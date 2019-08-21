#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Description: 
# @File: traffic_signs.py
# @Project: tf_learn
# @Author: Yiheng
# @Email: GuoYiheng89@gmail.com
# @Time: 8/19/2019 17:42
import os
import random
import numpy as np
import tensorflow as tf
from skimage import transform
from skimage import color
from skimage import data
import matplotlib.pyplot as plt


def is_dir(parent_path, son_path):
    full_path = os.path.join(parent_path, son_path)
    return os.path.isdir(full_path)


def list_dir(parent_path, son_path=None):
    full_path = parent_path if son_path is None else os.path.join(parent_path, son_path)
    return os.listdir(full_path)


def read_img(data_dir, img_dir, img_fname):
    img_dir_path = os.path.join(data_dir, img_dir)
    img_path = os.path.join(img_dir_path, img_fname)
    return data.imread(img_path)


def get_data_tuple(data_dir, img_dir, img_fname):
    # print(f'{data_dir}, {img_dir}, {img_fname}')
    return int(img_dir), read_img(data_dir, img_dir, img_fname)


def load_data(data_dir):
    datas = [get_data_tuple(data_dir, son_file, grand_son)
             for son_file in list_dir(data_dir) if is_dir(data_dir, son_file)
             for grand_son in list_dir(data_dir, son_file) if grand_son.endswith('ppm')]
    """
    for img, label in datas:
        print(f'{img}, {label}')
    """

    return zip(*datas)


def show_img(images, img_id):
    # Fill out the subplots with the random images that you defined
    for i in range(len(img_id)):
        plt.subplot(1, 4, i + 1)
        plt.axis('off')
        img2show = images[img_id[i]]
        plt.imshow(img2show)
        plt.subplots_adjust(wspace=0.5)
        plt.show()
        print(f'shape: {img2show.shape}, min: {img2show.min()}, max: {img2show.max()}')


def show_label_info(labels):
    plt.hist(labels, len(set(labels)))
    plt.show()


def show_label_img(labels, images, cmap=None):
    unique_labels = set(labels)
    plt.figure(figsize=(18, 16))
    for i, label in enumerate(unique_labels):
        plt.subplot(8, 8, i + 1)
        plt.axis('off')
        img2show = images[labels.index(label)]
        plt.title(f'label {label}, img count: {labels.count(label)}')
        plt.imshow(img2show, cmap=cmap)
    plt.show()


def pretreate_image(images):
    # resize
    images = [transform.resize(image, (28, 28)) for image in images]
    images_arr = np.array(images)
    print(f'image array ndim: {images_arr.ndim}, shape: {images_arr.shape}, size: {images_arr.size}')
    # grayscale
    return color.rgb2gray(images_arr)


if __name__ == '__main__':
    img_root_path = 'E:/tf_test'
    train_dir = os.path.join(img_root_path, 'Training')
    test_dir = os.path.join(img_root_path, 'Testing')

    train_labels, train_images = load_data(train_dir)
    train_images = pretreate_image(train_images)
    # show_img(train_images, [300, 2250, 3650, 4000])
    # show_label_img(train_labels, train_images, cmap='gray')

    test_labels, test_images = load_data(test_dir)
    test_images = pretreate_image(test_images)

    x = tf.placeholder(tf.float32, shape=[None, 28, 28])
    y = tf.placeholder(tf.int32, shape=[None])
    images_flat = tf.layers.flatten(x)
    logits = tf.layers.dense(images_flat, 62)
    logits = tf.nn.relu(logits)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(cross_entropy)
    train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    correct_pred = tf.argmax(logits, 1)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    print(f'images_flat: {images_flat}')
    print(f'logits: {logits}')
    print(f'loss: {loss}')
    print(f'predicted_labels: {correct_pred}')

    tf.set_random_seed(1234)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(201):
            loss_value, acc_value = sess.run([loss, accuracy], feed_dict={x: train_images, y: train_labels})
            if i % 10 == 0:
                print(f'Loss: {loss_value}, Acc: {acc_value}')

        predicted = sess.run([correct_pred], feed_dict={x: test_images})[0]
        match_count = sum([int(y == y_) for y, y_ in zip(test_labels, predicted)])
        accuracy = match_count / len(test_labels)
        print(f'test acc: {accuracy}')
