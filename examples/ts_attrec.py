# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import sys

sys.path.append('../')

import tensorflow as tf
import autokeras as ak
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pandas import concat
from pandas import DataFrame
import numpy as np
from sklearn.preprocessing import LabelEncoder
from autorecsys.pipeline.interactor import MLPInteraction

import logging
from autorecsys.pipeline import SparseFeatureMapper, RatingPredictionOptimizer, DenseFeatureMapper
from autorecsys.pipeline.interactor_ts import Bi_LSTMInteractor, LSTMInteractor, \
    GRUInteractor, TokenAndPositionEmbedding, TransformerBlock, Transformers2, BertBlock, TransformerBlock3sas
from autorecsys.pipeline.preprocessor_ts import MovielensPreprocessor

print(tf.config.list_physical_devices())
print(sys.version)
print(tf.__version__)
print('v1.4')


seed_value=0
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
tf.random.set_seed(seed_value)




# logging setting
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

movielens = MovielensPreprocessor(csv_path="./example_datasets/movielens/", nrows=50000)
train_X, train_y, val_X, val_y, test_X, test_y = movielens.preprocess()

del train_y
del test_y
train_X_numerical = movielens.train[['userID', 'age', 'releaseDate']]
train_X_categorical = movielens.train[['gender', 'occupation', 'Action', 'Adventure',
                                       'Animation', "Children's", 'Comedy', "Crime", 'Documentary', 'Drama',
                                       'Fantasy', "Film-Noir", 'Horror', 'Musical', 'Mystery', 'Romance',
                                       'Sci-Fi', 'Thriller', 'War', 'Western']]
train_yr = movielens.train[['rating']]
train_ym = movielens.train[['movieID']]


test_X_numerical = movielens.test[['userID', 'age', 'releaseDate']]
test_X_categorical = movielens.test[['gender', 'occupation', 'Action', 'Adventure',
                                     'Animation', "Children's", 'Comedy', "Crime", 'Documentary', 'Drama',
                                     'Fantasy', "Film-Noir", 'Horror', 'Musical', 'Mystery', 'Romance',
                                     'Sci-Fi', 'Thriller', 'War', 'Western']]
test_yr = movielens.test[['rating']]
test_ym = movielens.test[['movieID']]

train_X = movielens.train[['userID', 'movieID']]

# start here

print('start here')
all_items = movielens.full['movieID'].unique()
users_item = movielens.users_item



full = movielens.full[['userID', 'movieID', 'rating']]
print(full.shape)

# full = full[['userID', 'movieID', 'rating']]
user_sessions = full.groupby('userID')['movieID'].apply(lambda x: x.tolist()) \
    .reset_index().rename(columns={'movieID': 'movieID_list'})

print(user_sessions.head(10))