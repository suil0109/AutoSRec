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
from autorecsys.pipeline.preprocessor_ts import LastFMPreprocessor
from autorecsys.pipeline.preprocessor_ts import NetflixPrizePreprocessor

from autorecsys.pipeline.interactor_ts import LSTMInteractor, GRUInteractor

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

netflix = NetflixPrizePreprocessor()
train_X, train_y, val_X, val_y, test_X, test_y = netflix.preprocess()

del train_y
del test_y
train_X_numerical = netflix.train[['movieID', 'userID']]
train_yr = netflix.train[['rating']]

test_X_numerical = netflix.test[['movieID', 'userID']]
test_yr = netflix.test[['rating']]

# print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
timestep = 10

train_numerical = TimeseriesGenerator(train_X_numerical.values, train_yr.values,
                                      length=timestep, batch_size=1000000)
test_numerical = TimeseriesGenerator(test_X_numerical.values, test_yr.values,
                                     length=timestep, batch_size=1000000)

train_X_numerical, train_yr = train_numerical[0]
# train_X_categorical, train_ym = train_categorical[0]
test_X_numerical, test_yr = test_numerical[0]
# test_X_categorical, test_ym = test_categorical[0]

print(train_X_numerical.shape[2])

#####
print('preprocess data end1')

# Step 2: Build the recommender, which provides search space
# Step 2.1: Setup mappers to handle inputs
dense_input_node = ak.Input(shape=(256, train_X_numerical.shape[2]))

# Step 2.2: Setup interactors to handle models
from autorecsys.pipeline.interactor_ts import MLPInteraction
from autorecsys.pipeline.interactor import FMInteraction, SelfAttentionInteraction, InnerProductInteraction

# ncf1 = MLPInteraction()([dense_input_node])
# ncf2 = MLPInteraction()([sparse_input_node])

output = GRUInteractor()([dense_input_node])
output1 = MLPInteraction()([output])



# Step 2.3: Setup optimizer to handle the target task
output4 = ak.RegressionHead()(output1)

# Step 3: Build the searcher, which provides search algorithm
auto_model = ak.AutoModel(inputs=[dense_input_node],
                          outputs=[output4],
                          max_trials=3,
                          overwrite=True)

# Step 4: Use the searcher to search the recommender
auto_model.fit(x=[train_X_numerical],
               y=[train_yr],
               batch_size=256,
               epochs=5,
               callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1)])

# logger.info('Validation Accuracy (mse): {}'.format(auto_model.evaluate(x=[val_X],
#                                                                      y=val_y)))
# Step 5: Evaluate the searched model
a = auto_model.predict(x=[test_X_numerical], batch_size=128)
print(a)
# a-> predicted y
# x-> numerical categorical
# ReRanking

from autorecsys.pipeline.ts_eval_netflix import Eval_Topk
eval = Eval_Topk(pred=a, data=val_X, topk=5, timestep=timestep)
acc = eval.HR_MRR_nDCG()

# print(a[0])
# print('\na1')
# print(a[1])
# print('\n\n')
# print('starting reranking')
#
# dense_input_ = ak.Input(shape=(256, train_X_numerical.shape[2]))
# sparse_input_ = ak.Input(shape=(256, train_X_categorical.shape[2]))
# output1 = MLPInteraction()([dense_input_, sparse_input_])
# output2 = ak.RegressionHead()(output1)
#
#
# auto_model = ak.AutoModel(inputs=[dense_input_, sparse_input_],
#                           outputs=output2,
#                           max_trials=3,
#                           overwrite=True)
# auto_model.fit(x=[train_X_numerical, train_X_categorical],
#                y=r[0],
#                batch_size=256,
#                epochs=10,
#                callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1)])
# b = auto_model.predict(x=[test_X_numerical, test_X_categorical], batch_size=128)
# eval = Eval_Topk(pred=b, data=val_X, topk=5, timestep=timestep)
# acc1 = eval.HR_MRR_nDCG()