# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import tensorflow as tf
import sys
import logging

sys.path.append('../')
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

from autorecsys.pipeline.preprocessor_ts import MovielensPreprocessor
from autorecsys.pipeline.ts_baseline import NoDNN_topk

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

del train_y
del test_y
train_X_numerical = movielens.train[['age', 'releaseDate']]
train_X_categorical = movielens.train[['userID','gender', 'occupation', 'movieID', 'Action', 'Adventure',
                                       'Animation', "Children's", 'Comedy', "Crime", 'Documentary', 'Drama',
                                       'Fantasy', "Film-Noir", 'Horror', 'Musical', 'Mystery', 'Romance',
                                       'Sci-Fi', 'Thriller', 'War', 'Western']]
train_y = movielens.train[['rating']]

test_X_numerical = movielens.test[['age', 'releaseDate']]
test_X_categorical = movielens.test[['userID', 'gender', 'occupation', 'movieID', 'Action', 'Adventure',
                                     'Animation', "Children's", 'Comedy', "Crime", 'Documentary', 'Drama',
                                     'Fantasy', "Film-Noir", 'Horror', 'Musical', 'Mystery', 'Romance',
                                     'Sci-Fi', 'Thriller', 'War', 'Western']]
test_y = movielens.test[['rating']]
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

timestep=10
eval = NoDNN_topk(baseline='randompop', data=val_X, topk=5, timestep=timestep)
acc = eval.HR_MRR_nDCG()
