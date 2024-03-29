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
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Model
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score

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


def print_model(md: Model):
    print(f'model shape:')
    print(f'{md.output_shape}')

    print(f'model summary:')
    print(f'{md.summary()}')

    print(f'model config:\n {md.get_config()}')
    print(f'model weight:\n {md.get_weights()}')


def print_evaluate(y_true, y_pred):
    print('confusion matrix is:')
    print(f'{confusion_matrix(y_true, y_pred)}')
    print('precision score is:')
    print(f'{precision_score(y_true, y_pred)}')
    print('recall score is:')
    print(f'{recall_score(y_true, y_pred)}')
    print('f1 score is:')
    print(f'{f1_score(y_true, y_pred)}')
    print('cohen kappa score is:')
    print(f'{cohen_kappa_score(y_true, y_pred)}')


def classify_wine_type(wines_df):
    x = wines_df.iloc[:, 0:11]
    y = np.ravel(wines_df.type)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    model = Sequential()
    model.add(Dense(12, activation='relu', input_shape=(11,)))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # print_model(model)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5, batch_size=30, verbose=2)
    # Return the prediction score
    # y_pred = model.predict(X_test, verbose=1)
    # Return the prediction class
    y_pred = model.predict_classes(x_test, verbose=1)

    # The score is a list that holds the combination of the loss and the accuracy
    score = model.evaluate(x_test, y_test, verbose=1)
    print(f'score is: {score}')


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
    # classify_wine_type(wines)
    y = wines['quality']
    x = wines.drop('quality', axis=1)
    x = StandardScaler().fit_transform(x)


    seed = 7
    np.random.seed(seed)

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for train, test in kfold.split(x, y):
        model = Sequential()
        model.add(Dense(64, activation='relu', input_shape=(12,)))
        model.add(Dense(1))
        model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
        model.fit(x[train], y[train], epochs=10, verbose=1)

        mse_value, mae_value = model.evaluate(x[test], y[test], verbose=0)
        print(f'mse_value is: {mse_value}')
        print(f'mae_value is: {mae_value}')
        #r2_score(y[test], y_pred)