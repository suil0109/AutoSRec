# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys
sys.path.append('../')

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import tensorflow as tf
import autokeras as ak
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pandas import concat
from pandas import DataFrame
import numpy as np
from sklearn.preprocessing import LabelEncoder

import logging
from autorecsys.pipeline import SparseFeatureMapper, RatingPredictionOptimizer, DenseFeatureMapper
from autorecsys.pipeline.interactor_ts import Bi_LSTMInteractor, LSTMInteractor, GRUInteractor
from autorecsys.pipeline.preprocessor_ts import MovielensPreprocessor

# logging setting
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# rnames = ['userID','gender','age','occupation','movieID','genres','releaseDate','rating','daily']
dataset = pd.read_csv("./example_datasets/movielens/combined.csv", sep=',', nrows=10000, index_col=0, engine='python')

df_cat_tmp = dataset['gender'].astype('category')
df_cat_tmp = df_cat_tmp.unique()
cat_to_int = {word: ii for ii, word in enumerate(df_cat_tmp, 1)}
dataset['gender'] = dataset['gender'].map(cat_to_int)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
print(dataset.head(15))
print('hhhhhhhhh')

# values = dataset.values
# integer encode direction
# encoder = LabelEncoder()
# values[:, 4] = encoder.fit_transform(values[:, 4])
# ensure all data is float
# values = values.astype('float32')
# normalize features
# scaler = MinMaxScaler(feature_range=(0, 1))
# dataset[['userID','gender','age','occupation','movieID','genres','releaseDate','rating','daily']] = scaler.fit_transform(dataset[['userID','gender','age','occupation','movieID','genres','releaseDate','rating','daily']])
# frame as supervised learning
# reframed = series_to_supervised(scaled, 1, 1)
# drop columns we don't want to predict

reframed = dataset.reindex(
    columns=['daily', 'gender', 'age', 'occupation', 'genres', 'releaseDate', 'movieID', 'rating'])
reframed = reframed.sort_values(by=['daily'], ascending=True)

#####
# split into train and test sets
values = reframed.values
#batch_size = 24
n_train_hours = 8000
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]

# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

from keras.preprocessing.sequence import TimeseriesGenerator
train = TimeseriesGenerator(train_X, train_y,
                              length=256, batch_size=1774)
test = TimeseriesGenerator(test_X, test_y,
                              length=256, batch_size=1774)

train_X, train_y = train[0]
test_X, test_y = test[0]
print(train_X.shape)
#####
print('preprocess data end')

# train_X_categorical = movielens.get_x_categorical(train_X)
# val_X_categorical = movielens.get_x_categorical(val_X)
# test_X_categorical = movielens.get_x_categorical(test_X)

# Step 2: Build the recommender, which provides search space
# Step 2.1: Setup mappers to handle inputs
input_node = ak.Input(shape=(256, train_X.shape[2]))

# Step 2.2: Setup interactors to handle models
output1 = GRUInteractor()([input_node])

# Step 2.3: Setup optimizer to handle the target task
output = ak.RegressionHead()(output1)

# Step 3: Build the searcher, which provides search algorithm
auto_model = ak.AutoModel(inputs=input_node,
                          outputs=output,
                          objective='val_mean_squared_error',
                          max_trials=2,
                          overwrite=True)

# Step 4: Use the searcher to search the recommender
auto_model.fit(x=[train_X],
               y=train_y,
               batch_size=256,
               epochs=2,
               callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1)])

# logger.info('Validation Accuracy (mse): {}'.format(auto_model.evaluate(x=[val_X],
#                                                                      y=val_y)))
# Step 5: Evaluate the searched model
logger.info('Test Accuracy (mse): {}'.format(auto_model.evaluate(x=[test_X],
                                                                 y=test_y)))
