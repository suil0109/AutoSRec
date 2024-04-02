# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import sys

sys.path.append('../')

import tensorflow as tf
import scipy.sparse as sp

import itertools

import autokeras as ak
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pandas import concat
from pandas import DataFrame
import numpy as np
from sklearn.preprocessing import LabelEncoder
from autorecsys.pipeline.interactor import MLPInteraction

import logging
from autorecsys.pipeline.ts_baselines import create_matrix, create_train_test, BPR, auc_score
from autorecsys.pipeline import SparseFeatureMapper, RatingPredictionOptimizer, DenseFeatureMapper, ElementwiseInteraction
from autorecsys.pipeline.interactor_ts import Bi_LSTMInteractor, LSTMInteractor, \
    GRUInteractor, TokenAndPositionEmbedding, TransformerBlock, Transformers2, BPRMultiply,BPRInteractor
from autorecsys.pipeline.interactor import InnerProductInteraction
from autorecsys.pipeline.preprocessor_ts import MovielensPreprocessor
from autorecsys.pipeline.mapper_ts import BPREmbed


from os import path
from collections import OrderedDict
from tqdm import tqdm
from typing import Dict

from sklearn.metrics import roc_auc_score


print(tf.config.list_physical_devices())
print(sys.version)
print(tf.__version__)
print('v1.4')

# logging setting
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

movielens = MovielensPreprocessor(csv_path="./example_datasets/movielens/")
train_X, train_y, val_X, val_y, test_X, test_y = movielens.preprocess()

del train_X, train_y
del test_X, test_y
train_X = movielens.train[['userID', 'movieID', 'rating']]
test_X = movielens.test[['userID', 'movieID', 'rating']]

unique_users = train_X.userID.unique()
user_ids = dict(zip(unique_users, np.arange(unique_users.shape[0], dtype=np.int32)))
unique_movies = train_X.movieID.unique()
movie_ids = dict(zip(unique_movies, np.arange(unique_movies.shape[0], dtype=np.int32)))
train_X['user_id'] = train_X.userID.apply(lambda u: user_ids[u])
train_X['movie_id'] = train_X.movieID.apply(lambda m: movie_ids[m])

print(train_X.head(50))

ground_truth_train = train_X[train_X.rating > 3].groupby('user_id').movie_id.agg(list).reset_index()

print(ground_truth_train.head(10))

df_triplets = pd.DataFrame(columns=['user_id', 'positive_m_id', 'negative_m_id'])

data = []
users_without_data = []

for user_id in tqdm(train_X.user_id.unique()):
    positive_movies = train_X[(train_X.user_id == user_id) & (train_X.rating > 3)].movie_id.values
    negative_movies = train_X[(train_X.user_id == user_id) & (train_X.rating <= 3)].movie_id.values

    if negative_movies.shape[0] == 0 or positive_movies.shape[0] == 0:
        users_without_data.append(user_id)
        continue


    for positive_movie in positive_movies:
        for negative_movie in negative_movies:
            data.append({'user_id': user_id, 'positive_m_id': positive_movie, 'negative_m_id': negative_movie})

df_triplets = df_triplets.append(data, ignore_index=True)

print(df_triplets.shape)
df_triplets.user_id = df_triplets.user_id.astype(int)
df_triplets.positive_m_id = df_triplets.positive_m_id.astype(int)
df_triplets.negative_m_id = df_triplets.negative_m_id.astype(int)

print(df_triplets.user_id.shape, df_triplets.positive_m_id.shape, df_triplets.negative_m_id.shape)
print(df_triplets.head(10))