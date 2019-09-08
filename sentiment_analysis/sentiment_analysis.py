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
from common import file_utils
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


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


if __name__ == '__main__':
    all_df = read_csv('./txt')
    yelp_df = all_df[all_df['source'] == 'yelp']
    # print(f'{yelp_df.head()}')
    x_train, x_test, y_train, y_test = get_train_data_set(yelp_df, 'sentence', 'label')
    print(f'{x_train}')
    vectorizer = CountVectorizer()
    vectorizer.fit(x_train)
    x_train = vectorizer.transform(x_train)
    x_test = vectorizer.transform(x_test)
