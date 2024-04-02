# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import autokeras as ak
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pandas import concat
from pandas import DataFrame
import numpy as np
from sklearn.preprocessing import LabelEncoder

import sys
sys.path.append('../')



physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")
print("Num GPUs Available: ", len(physical_devices))


import logging
from autorecsys.pipeline import SparseFeatureMapper, RatingPredictionOptimizer, DenseFeatureMapper
from autorecsys.pipeline.interactor_ts import Bi_LSTMInteractor, LSTMInteractor, GRUInteractor
from autorecsys.pipeline.preprocessor_ts import MovielensPreprocessor

# logging setting
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# load dataset
##Netflix Dataset
# dataset_paths = ["./examples/datasets/net flix-prize-data/combined_data_" + str(i) + ".txt" for i in range(1, 5)]
# data = NetflixPrizePreprocessor(dataset_paths)

# Step 1: Preprocess data
# movielens = MovielensPreprocessor()
# train_X, train_y, val_X, val_y, test_X, test_y = movielens.preprocess()

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# rnames = ['userID','gender','age','occupation','movieID','genres','releaseDate','rating','daily']
dataset = pd.read_csv("./example_datasets/movielens/combined.csv", sep=',', nrows=10000, index_col=0, engine='python')

df_cat_tmp = dataset['gender'].astype('category')
df_cat_tmp = df_cat_tmp.unique()
cat_to_int = {word: ii for ii, word in enumerate(df_cat_tmp, 1)}
dataset['gender'] = dataset['gender'].map(cat_to_int)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# values = dataset.values
# integer encode direction
# encoder = LabelEncoder()
# values[:, 4] = encoder.fit_transform(values[:, 4])
# ensure all data is float
# values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
dataset[['userID','gender','age','occupation','movieID','genres','releaseDate','daily']] = scaler.fit_transform(dataset[['userID','gender','age','occupation','movieID','genres','releaseDate','daily']])
# frame as supervised learning
# reframed = series_to_supervised(scaled, 1, 1)
# drop columns we don't want to predict

reframed = dataset.reindex(
    columns=['userID', 'daily', 'gender', 'age', 'occupation', 'genres', 'releaseDate', 'movieID', 'rating'])

reframed = reframed.sort_values(by=['userID', 'daily'], ascending=(True,True))
reframed = reframed.reset_index(drop=True)
#print(reframed.shape,'2')
#print(reframed['userID'].value_counts(dropna=False))
print(reframed.head(10))

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

# # reshape input to be 3D [samples, timesteps, features]
# train_X = create_dataset(train_X, 24)
# test_X = create_dataset(test_X, 24)
# #train_X = train_X.reshape((train_X.shape[0], 24, train_X.shape[1]))
# #test_X = test_X.reshape((test_X.shape[0], 24, test_X.shape[1]))


print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

from keras.preprocessing.sequence import TimeseriesGenerator
train = TimeseriesGenerator(train_X, train_y,
                              length=64, batch_size=1774)
test = TimeseriesGenerator(test_X, test_y,
                              length=64, batch_size=1774)

train_X, train_y = train[0]
test_X, test_y = test[0]
print(train_X.shape)

#####
print('preprocess data end1')

# train_X_categorical = movielens.get_x_categorical(train_X)
# val_X_categorical = movielens.get_x_categorical(val_X)
# test_X_categorical = movielens.get_x_categorical(test_X)

# Step 2: Build the recommender, which provides search space
# Step 2.1: Setup mappers to handle inputs
input_node = ak.Input(shape=(64, train_X.shape[2]))

# Step 2.2: Setup interactors to handle models
from autorecsys.pipeline.interactor import MLPInteraction
output1 = LSTMInteractor()([input_node])
# output2 = MLPInteraction()([output1])

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
               batch_size=64,
               epochs=2,
               callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1)])

# logger.info('Validation Accuracy (mse): {}'.format(auto_model.evaluate(x=[val_X],
#                                                                      y=val_y)))
# Step 5: Evaluate the searched model
logger.info('Test Accuracy (mse): {}'.format(auto_model.evaluate(x=[test_X],
                                                                 y=test_y)))