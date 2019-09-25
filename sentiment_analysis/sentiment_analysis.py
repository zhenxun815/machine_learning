#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Description: 
# @File: sentiment_analysis.py
# @Project: machine_learning
# @Author: Yiheng
# @Email: GuoYiheng89@gmail.com
# @Time: 9/8/2019 11:56

import pandas as pd
import os
import tensorflow as tf
from common import file_utils
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense


def csv2df(csv_dir, csv_name):
    txt_file = os.path.join(csv_dir, csv_name)
    df = pd.read_csv(txt_file, sep='\t', names=['sentence', 'label'])
    source = file_utils.get_filename(csv_name, lambda name: name.split('_')[0])
    df['source'] = source
    return df


def read_csv(csv_dir):
    df_list = [csv2df(csv_dir, csv) for csv in os.listdir(csv_dir)]
    df = pd.concat(df_list)
    # print(f'{df.head()}')
    return df


def get_train_data_set(df, x_name, y_name):
    sentences = df[x_name].values
    labels = df[y_name].values
    return train_test_split(sentences, labels, test_size=0.25, random_state=9527)


def run_baseline_model():
    classifier = LogisticRegression()
    classifier.fit(x_train, y_train)
    score = classifier.score(x_test, y_test)
    print(f'base line model score is {score}')


def create_model1(input_dim):
    model = Sequential()
    model.add(Dense(10, input_dim=input_dim, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


if __name__ == '__main__':
    all_df = read_csv('./txt')
    yelp_df = all_df[all_df['source'] == 'yelp']
    # print(f'{yelp_df.head()}')
    x_train, x_test, y_train, y_test = get_train_data_set(yelp_df, 'sentence', 'label')
    # print(f'{x_train}')
    vectorizer = CountVectorizer()
    vectorizer.fit(x_train)
    x_train = vectorizer.transform(x_train)
    x_test = vectorizer.transform(x_test)
    print(f'x_train shape {x_train.shape}')

    input_dim = x_train.shape[1]
    print(f'input dim is {input_dim}')
    model = create_model1(input_dim)

    model.fit(x_train, y_train, batch_size=150, epochs=20, validation_data=(x_test, y_test), verbose=False)

    train_loss, train_acc = model.evaluate(x_train, y_train, verbose=False)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=False)
    print(f'train:\tloss: {"{:.3f}".format(train_loss)}, acc: {"{:.3f}".format(train_acc)}')
    print(f'test:\tloss: {"{:.3f}".format(test_loss)}, acc: {"{:.3f}".format(test_acc)}')
