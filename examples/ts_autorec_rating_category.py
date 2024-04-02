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
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"

from keras.preprocessing.sequence import TimeseriesGenerator
from autorecsys.pipeline.preprocessor_ts2 import MovielensPreprocessor
from autorecsys.pipeline.interactor_ts import LSTMInteractor, GRUInteractor, HyperInteraction3d, SelfAttentionInteraction
from autorecsys.pipeline.interactor import FMInteraction


# print(tf.config.list_physical_devices('GPU'))
# print(tf.config.list_physical_devices())
# print(sys.version)
# print(tf.__version__)
# print('v1.4')

seed_value=0
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
# import random
# random.seed(seed_value)

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

movielens = MovielensPreprocessor(csv_path="./example_datasets/movielens/", nrows=2000)
train_X, train_y, val_X, val_y, test_X, test_y = movielens.preprocess()

del train_y
del test_y
train_X_numerical = movielens.train[['age', 'releaseDate']]
train_X_categorical = movielens.train[['occupation', 'Action', 'Adventure',
                                       'Animation', "Children's", 'Comedy', "Crime", 'Documentary', 'Drama',
                                       'Fantasy', "Film-Noir", 'Horror', 'Musical', 'Mystery', 'Romance',
                                       'Sci-Fi', 'Thriller', 'War', 'Western']]
train_yr = movielens.train[['rating']]
train_ym = movielens.train[['movieID']]

test_X_numerical = movielens.test[['age', 'releaseDate']]
test_X_categorical = movielens.test[['occupation', 'Action', 'Adventure',
                                     'Animation', "Children's", 'Comedy', "Crime", 'Documentary', 'Drama',
                                     'Fantasy', "Film-Noir", 'Horror', 'Musical', 'Mystery', 'Romance',
                                     'Sci-Fi', 'Thriller', 'War', 'Western']]
test_yr = movielens.test[['rating']]
test_ym = movielens.test[['movieID']]

print(train_X_numerical.columns)
print(train_X_categorical.columns)

timestep = 10

train_numerical = TimeseriesGenerator(train_X_numerical.values, train_yr.values,
                                      length=timestep, batch_size=1000000)
train_categorical = TimeseriesGenerator(train_X_categorical.values, train_ym.values,
                                        length=timestep, batch_size=1000000)
test_numerical = TimeseriesGenerator(test_X_numerical.values, test_yr.values,
                                     length=timestep, batch_size=1000000)
test_categorical = TimeseriesGenerator(test_X_categorical.values, test_ym.values,
                                       length=timestep, batch_size=1000000)

train_X_numerical, train_yr = train_numerical[0]
train_X_categorical, train_ym = train_categorical[0]
test_X_numerical, test_yr = test_numerical[0]
test_X_categorical, test_ym = test_categorical[0]



#####
print('preprocess data end1')
from autorecsys.pipeline.interactor import HyperInteraction, MLPInteraction

# Step 2: Build the recommender, which provides search space
dense_input_node = ak.Input(shape=(timestep, train_X_numerical.shape[2]))
sparse_input_node = ak.Input(shape=(timestep, train_X_categorical.shape[2]))

# Step 2.2: Setup interactors to handle models
dense_feat_bottom_output = HyperInteraction3d(meta_interator_num=2)([dense_input_node])
sparse_feat_bottom_output = HyperInteraction3d(meta_interator_num=2)([sparse_input_node])
hyper_output = MLPInteraction()([dense_feat_bottom_output, sparse_feat_bottom_output])

# Step 2.3: Setup optimizer to handle the target task
output1 = ak.RegressionHead()(hyper_output)
output2 = ak.ClassificationHead()(hyper_output)

# Step 3: Build the searcher, which provides search algorithm
auto_model = ak.AutoModel(inputs=[dense_input_node, sparse_input_node], outputs=[output1, output2],
                          max_trials=3,
                          tuner='random',
                          overwrite=True)

# Step 4: Use the searcher to search the recommender
auto_model.fit(x=[train_X_numerical, train_X_categorical],
               y=[train_yr, train_ym],
               batch_size=256,
               epochs=8,
               callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1)])

a = auto_model.predict(x=[test_X_numerical, test_X_categorical], batch_size=256)
print(a[0])

from autorecsys.pipeline.ts_eval import Eval_Topk
eval = Eval_Topk(pred=a[0], data=val_X, topk=5, timestep=timestep)
acc = eval.HR_MRR_nDCG()
