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
from autorecsys.pipeline.preprocessor_ts import MovielensPreprocessor, AmazonBeautyPreprocessor
from autorecsys.pipeline.interactor_ts import LSTMInteractor, GRUInteractor, HyperInteraction3d

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

beauty_url = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Beauty.csv"
sports_url = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Sports_and_Outdoors.csv"
amazonbeauty = AmazonBeautyPreprocessor(non_csv_path=sports_url, nrows=1000000)
train_X, train_y, val_X, val_y, test_X, test_y = amazonbeauty.preprocess()

del train_y
del test_y
# train_X_numerical = amazonbeauty.train[['userID']].values
train_X_categorical = amazonbeauty.train[['userID', 'movieID']].values
train_yr = amazonbeauty.train[['rating']].values

# test_X_numerical = amazonbeauty.test[['userID']].values
test_X_categorical = amazonbeauty.test[['userID', 'movieID']].values
test_yr = amazonbeauty.test[['rating']].values

# print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
# test_yy = test_y
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


# train_numerical = TimeseriesGenerator(train_X_numerical.values, train_yr.values,
#                                       length=timestep, batch_size=1000000)
# train_categorical = TimeseriesGenerator(train_X_numerical.values, train_ym.values,
#                                         length=timestep, batch_size=1000000)
# test_numerical = TimeseriesGenerator(test_X_numerical.values, test_yr.values,
#                                      length=timestep, batch_size=1000000)
# test_categorical = TimeseriesGenerator(test_X_numerical.values, test_ym.values,
#                                        length=timestep, batch_size=1000000)
#
# train_X_numerical, train_yr = train_numerical[0]
# train_X_categorical, train_ym = train_categorical[0]
# test_X_numerical, test_yr = test_numerical[0]
# test_X_categorical, test_ym = test_categorical[0]

# print(train_X_numerical.shape[1])


#####
print('preprocess data end1')
from autorecsys.pipeline.interactor_ts import MLPInteraction
from autorecsys.pipeline.interactor import HyperInteraction, FMInteraction

# Step 2: Build the recommender, which provides search space
# Step 2.1: Setup mappers to handle inputs
# input_node = ak.Input(shape=(64, train_X.shape[2]))

# dense_input_node = ak.Input(shape=(timestep, train_X_numerical.shape[1]))
sparse_input_node = ak.Input(shape=(timestep, train_X_categorical.shape[1]))

from autorecsys.pipeline import DenseFeatureMapper, SparseFeatureMapper
# dense_feat_emb = DenseFeatureMapper(
#     num_of_fields=train_X_numerical.shape[1],
#     embedding_dim=8)(dense_input_node)
sparse_feat_emb = SparseFeatureMapper(
    num_of_fields=train_X_categorical.shape[1],
    hash_size=None,
    embedding_dim=2)(sparse_input_node)


# Step 2.2: Setup interactors to handle models
fm_output = FMInteraction()([sparse_feat_emb])
# bottom_mlp_output = MLPInteraction()([dense_feat_emb])
hyper_output = MLPInteraction()([fm_output])
# sparse_feat_mlp_output = MLPInteraction()([sparse_feat_emb])
# # dense_feat_mlp_output = MLPInteraction()([dense_feat_emb])
# hyper_output = MLPInteraction(num_layers=2)([sparse_feat_mlp_output])

# Step 2.2: Setup interactors to handle models
# hyper_output = HyperInteraction3d(meta_interator_num=2)([dense_input_node])
# sparse_feat_bottom_output = HyperInteraction3d(meta_interator_num=2)([sparse_input_node])
# hyper_output = MLPInteraction()([dense_feat_bottom_output])

# Step 2.3: Setup optimizer to handle the target task
output1 = ak.RegressionHead()(hyper_output)
# output2 = ak.ClassificationHead()(hyper_output)

# "Combinator loss" to emphasize on classificationhead()
# Only output1 with classification ex) output3 = combinationhead()(hyper_output)

# Step 3: Build the searcher, which provides search algorithm
auto_model = ak.AutoModel(inputs=[ sparse_input_node],
                          outputs=[output1],
                          max_trials=3,
                          tuner='random',
                          overwrite=True)

# Step 4: Use the searcher to search the recommender
auto_model.fit(x=[train_X_categorical],
               y=[train_yr],
               batch_size=256,
               epochs=10,
               callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1)])

# logger.info('Validation Accuracy (mse): {}'.format(auto_model.evaluate(x=[val_X],
#                                                                      y=val_y)))
# Step 5: Evaluate the searched model
print('noot noot')
a = auto_model.predict(x=[test_X_categorical], batch_size=256)
# print(a[0])
from autorecsys.pipeline.ts_eval import Eval_Topk
eval = Eval_Topk(pred=a, data=val_X, topk=5, timestep=0)
acc = eval.HR_MRR_nDCG()
