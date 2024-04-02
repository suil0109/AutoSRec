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
from autorecsys.pipeline import SparseFeatureMapper, RatingPredictionOptimizer, DenseFeatureMapper, ConcatenateInteraction
from autorecsys.pipeline.interactor_ts import LSTMInteractor, \
    GRUInteractor, TokenAndPositionEmbedding, TransformerBlock, Transformers2, BertBlock, TransformerBlock3sas, SelfAttentionInteraction
from autorecsys.pipeline.preprocessor_ts2 import MovielensPreprocessor

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

movielens = MovielensPreprocessor(csv_path="./example_datasets/movielens/", nrows=20000)
train_X, train_y, val_X, val_y, test_X, test_y = movielens.preprocess()

del train_y
del test_y
train_X_numerical = movielens.train[['age', 'releaseDate']]
train_X_categorical = movielens.train[['gender', 'occupation', 'Action', 'Adventure',
                                       'Animation', "Children's", 'Comedy', "Crime", 'Documentary', 'Drama',
                                       'Fantasy', "Film-Noir", 'Horror', 'Musical', 'Mystery', 'Romance',
                                       'Sci-Fi', 'Thriller', 'War', 'Western']]
train_yr = movielens.train[['rating']]
train_ym = movielens.train[['movieID']]


test_X_numerical = movielens.test[['age', 'releaseDate']]
test_X_categorical = movielens.test[['gender', 'occupation', 'Action', 'Adventure',
                                     'Animation', "Children's", 'Comedy', "Crime", 'Documentary', 'Drama',
                                     'Fantasy', "Film-Noir", 'Horror', 'Musical', 'Mystery', 'Romance',
                                     'Sci-Fi', 'Thriller', 'War', 'Western']]
test_yr = movielens.test[['rating']]
test_ym = movielens.test[['movieID']]


train_X_numerical, train_X_categorical, train_yr, train_ym = train_X_numerical.values, train_X_categorical.values,\
                                                  train_yr.values, train_ym.values
test_X_numerical, test_X_categorical, test_yr, test_ym = test_X_numerical.values, test_X_categorical.values,\
                                               test_yr.values, test_ym.values

print(train_X_numerical.shape, train_X_categorical.shape, train_yr.shape)
print(test_X_numerical.shape, test_X_categorical.shape, test_ym.shape)
print('preprocess data end')
hash_size = None
print(train_ym.shape, train_yr.shape)
# Step 2: Build the recommender, which provides search space
# Step 2.1: Setup mappers to handle inputs
# input_node = ak.Input(shape=(train_X.shape[1],))
dense_input_node = ak.Input(shape=[train_X_numerical.shape[1]])
sparse_input_node = ak.Input(shape=[train_ym.shape[1]])
# categorical_input_node = ak.Input(shape=[train_X_categorical.shape[1]])

comb = MLPInteraction()([dense_input_node, sparse_input_node])

# Step 2.2: Setup interactors to handle models
sparse_input = TokenAndPositionEmbedding(maxlen=1000, vocab_size=10000)([comb])
transformer_block = SelfAttentionInteraction()(sparse_input)

# transformer_block1 = Transformers2(maxlen=train_X.shape[1], vocab_size=100000)(dense_input_node)
# transformer_block2 = Transformers2(maxlen=train_X.shape[1], vocab_size=100000)(sparse_input_node)
# transformer_block = MLPInteraction()([transformer_block1])


# transformer_block1 = BertBlock(max_sequence_length=train_X.shape[1])(dense_input_node)
# transformer_block2 = BertBlock(max_sequence_length=train_X.shape[1])(sparse_input_node)
# transformer_block = MLPInteraction()([transformer_block1, transformer_block2])

# Step 2.3: Setup optimizer to handle the target task
output1 = ak.RegressionHead()(transformer_block)
output2 = ak.ClassificationHead()(transformer_block)


# from keras.callbacks import LambdaCallback
# print_weights = LambdaCallback(on_epoch_end=lambda batch, logs: print(self.model.get_weights()))

# Step 3: Build the searcher, which provides search algorithm
auto_model = ak.AutoModel(inputs=[dense_input_node, sparse_input_node],
                          outputs=[output1, output2],
                          max_trials=3,
                          overwrite=True)

# Step 4: Use the searcher to search the recommender
auto_model.fit(x=[train_X_numerical, train_X_categorical],
               y=[train_yr, train_ym],
               batch_size=256,
               epochs=5)
a = auto_model.predict(x=[test_X_numerical, test_X_categorical], batch_size=256)
from autorecsys.pipeline.ts_eval import Eval_Topk

print(a)
eval = Eval_Topk(pred=a[0], data=val_X, topk=5)
acc = eval.HR_MRR_nDCG()

# logger.info('Validation Accuracy (mse): {}'.format(auto_model.evaluate(x=[val_X],
#                                                                      y=val_y)))
# Step 5: Evaluate the searched model
