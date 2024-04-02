# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import sys
sys.path.append('../')
import logging
import tensorflow as tf
import autokeras as ak
from autorecsys.pipeline import DenseFeatureMapper, SparseFeatureMapper, FMInteraction, MLPInteraction,\
    InnerProductInteraction, CrossNetInteraction
from autorecsys.pipeline.preprocessor_ts import MovielensPreprocessor

seed_value= 0

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
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

movielens = MovielensPreprocessor(csv_path="./example_datasets/movielens/")
train_X, train_y, val_X, val_y, test_X, test_y = movielens.preprocess()

del train_y
del test_y
train_X_numerical = movielens.train[['age', 'releaseDate']]
train_X_categorical = movielens.train[['userID', 'gender', 'occupation', 'Action', 'Adventure',
                                       'Animation', "Children's", 'Comedy', "Crime", 'Documentary', 'Drama',
                                       'Fantasy', "Film-Noir", 'Horror', 'Musical', 'Mystery', 'Romance',
                                       'Sci-Fi', 'Thriller', 'War', 'Western']]
train_yr = movielens.train[['rating']]
train_ym = movielens.train[['movieID']]


test_X_numerical = movielens.test[['age', 'releaseDate']]
test_X_categorical = movielens.test[['userID', 'gender', 'occupation', 'Action', 'Adventure',
                                     'Animation', "Children's", 'Comedy', "Crime", 'Documentary', 'Drama',
                                     'Fantasy', "Film-Noir", 'Horror', 'Musical', 'Mystery', 'Romance',
                                     'Sci-Fi', 'Thriller', 'War', 'Western']]
test_yr = movielens.test[['rating']]
test_ym = movielens.test[['movieID']]

# print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
# test_yy = test_y

# Step 2: Build the recommender, which provides search space
# Step 2.1: Setup mappers to handle inputs
dense_input_node = ak.Input(shape=[train_X_numerical.shape[1]])
sparse_input_node = ak.Input(shape=[train_X_categorical.shape[1]])
dense_feat_emb = DenseFeatureMapper(
    num_of_fields=train_X_numerical.shape[1],
    embedding_dim=8)(dense_input_node)
sparse_feat_emb = SparseFeatureMapper(
    num_of_fields=train_X_categorical.shape[1],
    hash_size=None,
    embedding_dim=8)(sparse_input_node)

# Step 2.2: Setup interactors to handle models
# fm_output = FMInteraction()([sparse_feat_emb])
# bottom_mlp_output = MLPInteraction()([dense_feat_emb])
# top_mlp_output = MLPInteraction()([fm_output, bottom_mlp_output])
sparse_feat_mlp_output = MLPInteraction()([sparse_feat_emb])
dense_feat_mlp_output = MLPInteraction()([dense_feat_emb])
top_mlp_output = MLPInteraction(num_layers=2)([sparse_feat_mlp_output, dense_feat_mlp_output])


# Step 2.3: Setup optimizer to handle the target task
output1 = ak.RegressionHead()(top_mlp_output)
output2 = ak.ClassificationHead()(top_mlp_output)


# Step 3: Build the searcher, which provides search algorithm
auto_model = ak.AutoModel(inputs=[dense_input_node, sparse_input_node],
                          outputs=[output1, output2],
                          max_trials=2,
                          objective='val_loss',
                          tuner='random',
                          overwrite=True)

# Step 4: Use the searcher to search the recommender
auto_model.fit(x=[train_X_numerical.values, train_X_categorical.values],
               y=train_y,
               batch_size=64,
               epochs=3,
               callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1)]
               )

a = auto_model.predict(x=[test_X_numerical.values, test_X_categorical.values], batch_size=64)
print('a')
print(a.shape)
print('test_y')
# print(test_y.shape)

from autorecsys.pipeline.ts_eval import Eval_Topk
timestep = 0
eval = Eval_Topk(pred=a[0], data=val_X, topk=5, timestep=timestep)
acc = eval.HR_MRR_nDCG()


# logger.info('Validation Accuracy (logloss): {}'.format(auto_model.evaluate(x=[val_X_numerical, val_X_categorical],
#                                                                            y=val_y)))
#
# # Step 5: Evaluate the searched model
# logger.info('Test Accuracy (logloss): {}'.format(auto_model.evaluate(x=[test_X_numerical, test_X_categorical],
#                                                                      y=test_y)))
