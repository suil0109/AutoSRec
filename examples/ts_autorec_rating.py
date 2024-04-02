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
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

from keras.preprocessing.sequence import TimeseriesGenerator
from autorecsys.pipeline.preprocessor_ts import MovielensPreprocessor
from autorecsys.pipeline.interactor_ts import LSTMInteractor, GRUInteractor, HyperInteraction3d

# print(tf.config.list_physical_devices('GPU'))
print(tf.config.list_physical_devices())
print(sys.version)
print(tf.__version__)
print('v1.4')
# logging setting
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

movielens = MovielensPreprocessor(csv_path="./example_datasets/movielens/", nrows=300000, lcount=300, hcount=1300)
train_X, train_y, val_X, val_y, test_X, test_y = movielens.preprocess()

del train_y
del test_y
train_X_numerical = movielens.train[['age', 'releaseDate']]
train_X_categorical = movielens.train[['gender', 'occupation', 'movieID', 'Action', 'Adventure',
                                       'Animation', "Children's", 'Comedy', "Crime", 'Documentary', 'Drama',
                                       'Fantasy', "Film-Noir", 'Horror', 'Musical', 'Mystery', 'Romance',
                                       'Sci-Fi', 'Thriller', 'War', 'Western']]
train_y = movielens.train[['rating']]

test_X_numerical = movielens.test[['age', 'releaseDate']]
test_X_categorical = movielens.test[[ 'gender', 'occupation', 'movieID', 'Action', 'Adventure',
                                     'Animation', "Children's", 'Comedy', "Crime", 'Documentary', 'Drama',
                                     'Fantasy', "Film-Noir", 'Horror', 'Musical', 'Mystery', 'Romance',
                                     'Sci-Fi', 'Thriller', 'War', 'Western']]
test_y = movielens.test[['rating']]

print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
test_yy = test_y
timestep = 10
# train = TimeseriesGenerator(train_X, train_y,
#                               length=timestep, batch_size=5000)
#
#
# test = TimeseriesGenerator(test_X, test_y,
#                               length=timestep, batch_size=5000)
# train_X, train_y = train[0]
# test_X, test_y = test[0]
# print(train_X.shape)
# print(test_X.shape, test_y.shape)


train_numerical = TimeseriesGenerator(train_X_numerical.values, train_y.values,
                                      length=timestep, batch_size=1000000)
train_categorical = TimeseriesGenerator(train_X_categorical.values, train_y.values,
                                        length=timestep, batch_size=1000000)
test_numerical = TimeseriesGenerator(test_X_numerical.values, test_y.values,
                                     length=timestep, batch_size=1000000)
test_categorical = TimeseriesGenerator(test_X_categorical.values, test_y.values,
                                       length=timestep, batch_size=1000000)

train_X_numerical, train_y = train_numerical[0]
train_X_categorical, train_y = train_categorical[0]
test_X_numerical, test_y = test_numerical[0]
test_X_categorical, test_y = test_categorical[0]

print(train_X_numerical.shape[2])

#####
print('preprocess data end1')

# Step 2: Build the recommender, which provides search space
# Step 2.1: Setup mappers to handle inputs
# input_node = ak.Input(shape=(64, train_X.shape[2]))
dense_input_node = ak.Input(shape=(256, train_X_numerical.shape[2]))
sparse_input_node = ak.Input(shape=(256, train_X_categorical.shape[2]))

# Step 2.2: Setup interactors to handle models
from autorecsys.pipeline.interactor_ts import MLPInteraction
from autorecsys.pipeline.interactor import HyperInteraction, SelfAttentionInteraction

sparse_feat_bottom_output1 = HyperInteraction3d(meta_interator_num=2)([dense_input_node])
dense_feat_bottom_output1 = HyperInteraction3d(meta_interator_num=2)([sparse_input_node])
hyper_output = MLPInteraction()([sparse_feat_bottom_output1, dense_feat_bottom_output1])


# Step 2.3: Setup optimizer to handle the target task
output = ak.RegressionHead()(hyper_output)
# Step 3: Build the searcher, which provides search algorithm
auto_model = ak.AutoModel(inputs=[dense_input_node, sparse_input_node],
                          outputs=output,
                          objective='val_mean_squared_error',
                          tuner='random',
                          max_trials=3,
                          overwrite=True)

# Step 4: Use the searcher to search the recommender
auto_model.fit(x=[train_X_numerical, train_X_categorical],
               y=train_y,
               batch_size=256,
               epochs=3,
               callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1)])

# logger.info('Validation Accuracy (mse): {}'.format(auto_model.evaluate(x=[val_X],
#                                                                      y=val_y)))
# Step 5: Evaluate the searched model
a = auto_model.predict(x=[test_X_numerical, test_X_categorical], batch_size=256)

from autorecsys.pipeline.ts_eval import Eval_Topk
eval = Eval_Topk(pred=a, data=val_X, topk=5, timestep=timestep)
acc = eval.HR_MRR_nDCG()

print('\n\n\n')
print('--starting rerankingggg--')
rerank_y = auto_model.predict(x=[train_X_numerical, train_X_categorical], batch_size=256)

dense_input_ = ak.Input(shape=(32, test_X_numerical.shape[2]))
sparse_input_ = ak.Input(shape=(32, test_X_categorical.shape[2]))
outputt = MLPInteraction()([dense_input_, sparse_input_])
output = ak.RegressionHead()(outputt)
auto_model = ak.AutoModel(inputs=[dense_input_, sparse_input_],
                          outputs=output,
                          objective='val_mean_squared_error',
                          max_trials=2,
                          overwrite=True)
auto_model.fit(x=[train_X_numerical, train_X_categorical],
               y=rerank_y,
               batch_size=32,
               epochs=5,
               callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1)])
b = auto_model.predict(x=[test_X_numerical, test_X_categorical], batch_size=128)
eval = Eval_Topk(pred=b, data=val_X, topk=5, timestep=timestep)
acc1 = eval.HR_MRR_nDCG()
