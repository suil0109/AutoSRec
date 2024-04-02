# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import os

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
    GRUInteractor, TokenAndPositionEmbedding, TransformerBlock, Transformers2, BertBlock
from autorecsys.pipeline.preprocessor_ts import MovielensPreprocessor

print(tf.config.list_physical_devices())
print(sys.version)
print(tf.__version__)
print('v1.4')
os.environ["CUDA_VISIBLE_DEVICES"] = "5"



# logging setting
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

movielens = MovielensPreprocessor(csv_path="./example_datasets/movielens/")
train_X, train_y, val_X, val_y, test_X, test_y = movielens.preprocess()

del train_y
del test_y
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

train_X_numerical, train_X_categorical, train_y = train_X_numerical.values, train_X_categorical.values, train_y.values
test_X_numerical, test_X_categorical, test_y = test_X_numerical.values, test_X_categorical.values, test_y.values

print(train_X_numerical.shape, train_X_categorical.shape, train_y.shape)
print(test_X_numerical.shape, test_X_categorical.shape, test_y.shape)
print('preprocess data end')
hash_size = None
# Step 2: Build the recommender, which provides search space
# Step 2.1: Setup mappers to handle inputs
#i nput_node = ak.Input(shape=(train_X.shape[1],))
dense_input_node = ak.Input(shape=[3])
sparse_input_node = ak.Input(shape=[22])


# Step 2.2: Setup interactors to handle models
# embedding_layer = TokenAndPositionEmbedding(maxlen=train_X.shape[1], vocab_size=100000)([input_node])
transformer_block1 = BertBlock()([dense_input_node])
transformer_block2 = BertBlock()([sparse_input_node])
transformer_block = MLPInteraction()([transformer_block1, transformer_block2])
# transformer_block1 = BertBlock(max_sequence_length=train_X.shape[1])(dense_input_node)
# transformer_block2 = BertBlock(max_sequence_length=train_X.shape[1])(sparse_input_node)
# transformer_block = MLPInteraction()([transformer_block1, transformer_block2])


# Step 2.3: Setup optimizer to handle the target task
output = ak.RegressionHead()(transformer_block)

# Step 3: Build the searcher, which provides search algorithm
auto_model = ak.AutoModel(inputs=[dense_input_node, sparse_input_node],
                          outputs=output,
                          objective='val_mean_squared_error',
                          max_trials=2,
                          overwrite=True)

# Step 4: Use the searcher to search the recommender
auto_model.fit(x=[train_X_numerical, train_X_categorical],
               y=train_y,
               batch_size=256,
               epochs=5,
               callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1)])

a = auto_model.predict(x=[test_X_numerical, test_X_categorical], batch_size=256)
from autorecsys.pipeline.ts_eval import Eval_Topk

eval = Eval_Topk(pred=a, data=val_X, topk=5)
acc = eval.HR_MRR_nDCG()

# logger.info('Validation Accuracy (mse): {}'.format(auto_model.evaluate(x=[val_X],
#                                                                      y=val_y)))
# Step 5: Evaluate the searched model
