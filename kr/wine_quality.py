#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Description: 
# @File: wine_quality.py
# @Project: tf_learn
# @Author: Yiheng
# @Email: GuoYiheng89@gmail.com
# @Time: 8/21/2019 15:48

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def print_df(df, info=False, head=True, describe=False, isna=False):
    print(f'ndim: {df.ndim}, shape: {df.shape}, size: {df.size}')
    if info:
        print('info is:')
        print(f'{df.info()}')
    if head:
        print('head is:')
        print(f'{df.head()}')
    if describe:
        print('description:')
        print(f'{df.describe()}')
    if isna:
        print(f'is null:')
        print(f'{pd.isna(df)}')


def show_alcohol(white_df, red_df):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    spec = plt.specgram()

    ax[0].hist(red_df['alcohol'], 10, fc='red', alpha=0.5, label="Red wine")
    ax[1].hist(white_df['alcohol'], 10, fc='white', ec="black", lw=0.5, alpha=0.5, label="White wine")

    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.5, hspace=0.05, wspace=1)
    ax[0].set_ylim([0, 1000])
    ax[0].set_xlabel("Alcohol in % Vol")
    ax[0].set_ylabel("Frequency")
    ax[1].set_xlabel("Alcohol in % Vol")
    ax[1].set_ylabel("Frequency")
    # ax[0].legend(loc='best')
    # ax[1].legend(loc='best')
    fig.suptitle("Distribution of Alcohol in % Vol")

    plt.show()


def show_sulfate_quality(white_df, red_df):
    fig = plt.figure(figsize=(8, 4))
    fig.suptitle("Wine Quality by Amount of Sulphates", y=1.05)
    plt.subplot(1, 2, 1)
    plt.scatter(red_df['quality'], red_df['sulphates'], color='red')
    plt.title("Red Wine")
    plt.xlabel("Quality")
    plt.ylabel("Sulphates")
    plt.xlim([0, 10])
    plt.ylim([0, 2.5])
    plt.subplot(1, 2, 2)
    plt.scatter(white_df['quality'], white_df['sulphates'], color='white', edgecolors='black', linewidths=0.5)
    plt.title("White Wine")
    plt.xlabel("Quality")
    plt.ylabel("Sulphates")
    plt.xlim([0, 10])
    plt.ylim([0, 2.5])
    plt.subplots_adjust(wspace=0.5)
    plt.show()


def show_corr(wines_df):
    corr = wines_df.corr(method='pearson')
    sns.heatmap(corr, robust=True,
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values)
    plt.show()


if __name__ == '__main__':

    white = pd.read_csv('winequality-white.csv', sep=';')
    white['type'] = 0
    red = pd.read_csv('winequality-red.csv', sep=';')
    red['type'] = 1
    wines = red.append(white, ignore_index=True)
    # print_df(white)
    print_df(wines, describe=True)
    # show_alcohol(white, red)
    # show_sulfate_quality(white, red)
    X = wines.iloc[:, 0:11]
    y = np.ravel(wines.type)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
