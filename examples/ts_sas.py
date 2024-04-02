# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import tensorflow as tf
import sys
import autokeras as ak
import pandas as pd
import numpy as np
import logging

sys.path.append('../')
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

from keras.preprocessing.sequence import TimeseriesGenerator
from autorecsys.pipeline.preprocessor_ts import MovielensPreprocessor
from autorecsys.pipeline.interactor_ts import LSTMInteractor, GRUInteractor, TokenAndPositionEmbedding, TransformerBlock
from autorecsys.pipeline.mapper import DenseFeatureMapper, SparseFeatureMapper
from autorecsys.pipeline.interactor import MLPInteraction

# print(tf.config.list_physical_devices('GPU'))
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
import numpy as np
np.random.seed(seed_value)

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.random.set_seed(seed_value)


# logging setting
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

'''
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
'''

movielens = MovielensPreprocessor(csv_path="./example_datasets/movielens/", nrows=100000)
train_X, train_y, val_X, val_y, test_X, test_y = movielens.preprocess()

del train_y
# del test_y
# train_X_numerical = movielens.train[['age', 'releaseDate']]
# train_X_categorical = movielens.train[['gender', 'occupation', 'Action', 'Adventure',
#                                        'Animation', "Children's", 'Comedy', "Crime", 'Documentary', 'Drama',
#                                        'Fantasy', "Film-Noir", 'Horror', 'Musical', 'Mystery', 'Romance',
#                                        'Sci-Fi', 'Thriller', 'War', 'Western']]
# train_yr = movielens.train[['rating']]
# train_ym = movielens.train[['movieID']]
#
# test_X_numerical = movielens.test[['age', 'releaseDate']]
# test_X_categorical = movielens.test[['gender', 'occupation', 'Action', 'Adventure',
#                                      'Animation', "Children's", 'Comedy', "Crime", 'Documentary', 'Drama',
#                                      'Fantasy', "Film-Noir", 'Horror', 'Musical', 'Mystery', 'Romance',
#                                      'Sci-Fi', 'Thriller', 'War', 'Western']]
# test_yr = movielens.test[['rating']]
# test_ym = movielens.test[['movieID']]
#
# train_X_numerical, train_X_categorical, train_yr, train_ym = \
#     train_X_numerical.values, train_X_categorical.values, train_yr.values, train_ym.values
# test_X_numerical, test_X_categorical, test_yr, test_ym = \
#     test_X_numerical.values, test_X_categorical.values, test_yr.values, test_ym.values

# print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

train_X_numerical = movielens.train[['userID', 'age', 'releaseDate']]
train_X_categorical = movielens.train[['gender', 'occupation', 'movieID', 'Action', 'Adventure',
                                       'Animation', "Children's", 'Comedy', "Crime", 'Documentary', 'Drama',
                                       'Fantasy', "Film-Noir", 'Horror', 'Musical', 'Mystery', 'Romance',
                                       'Sci-Fi', 'Thriller', 'War', 'Western']]
train_y = movielens.train[['rating']]

test_X_numerical = movielens.test[['userID', 'age', 'releaseDate']]
test_X_categorical = movielens.test[['gender', 'occupation', 'movieID', 'Action', 'Adventure',
                                     'Animation', "Children's", 'Comedy', "Crime", 'Documentary', 'Drama',
                                     'Fantasy', "Film-Noir", 'Horror', 'Musical', 'Mystery', 'Romance',
                                     'Sci-Fi', 'Thriller', 'War', 'Western']]
test_y = movielens.test[['rating']]


#####
print('preprocess data end1')

# Step 2: Build the recommender, which provides search space
# Step 2.1: Setup mappers to handle inputs
dense_input_node = ak.Input(shape=[train_X_numerical.shape[1]])
sparse_input_node = ak.Input(shape=[train_X_categorical.shape[1]])
# dense_feat_emb = DenseFeatureMapper(
#     num_of_fields=train_X_numerical.shape[1],
#     embedding_dim=8)(dense_input_node)
# sparse_feat_emb = SparseFeatureMapper(
#     num_of_fields=train_X_categorical.shape[1],
#     hash_size=None,
#     embedding_dim=8)(sparse_input_node)

# Step 2.2: Setup interactors to handle models
# fm_output = FMInteraction()([sparse_feat_emb])
# bottom_mlp_output = MLPInteraction()([dense_feat_emb])
# top_mlp_output = MLPInteraction()([fm_output, bottom_mlp_output])

sparse_input = TokenAndPositionEmbedding(maxlen=train_X.shape[1], vocab_size=10000)([sparse_input_node])
dense_input = TokenAndPositionEmbedding(maxlen=train_X.shape[1], vocab_size=10000)([dense_input_node])
transformer_block1 = TransformerBlock()(sparse_input)
transformer_block2 = TransformerBlock()(dense_input)
top_mlp_output = MLPInteraction()([transformer_block1, transformer_block2])


# Step 2.3: Setup optimizer to handle the target task
output = ak.RegressionHead()(top_mlp_output)
output = ak.RegressionHead()(top_mlp_output)


# Step 3: Build the searcher, which provides search algorithm
auto_model = ak.AutoModel(inputs=[dense_input_node, sparse_input_node],
                          outputs=[output1, output2],
                          max_trials=2,
                          objective='val_loss',
                          tuner='random',
                          overwrite=True)

# Step 4: Use the searcher to search the recommender
auto_model.fit(x=[train_X_numerical.values, train_X_categorical.values],
               y=[train_yr, train_ym],
               batch_size=64,
               epochs=3,
               callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1)]
               )

a = auto_model.predict(x=[test_X_numerical.values, test_X_categorical.values], batch_size=64)
print('a')
print(a.shape)
print('test_y')
print(test_y.shape)

from autorecsys.pipeline.ts_eval import Eval_Topk
timestep = 0
eval = Eval_Topk(pred=a, data=val_X, topk=5, timestep=timestep)
acc = eval.HR_MRR_nDCG()


# logger.info('Validation Accuracy (logloss): {}'.format(auto_model.evaluate(x=[val_X_numerical, val_X_categorical],
#                                                                            y=val_y)))
#
# # Step 5: Evaluate the searched model
# logger.info('Test Accuracy (logloss): {}'.format(auto_model.evaluate(x=[test_X_numerical, test_X_categorical],
#                                                                      y=test_y)))
