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
from autorecsys.pipeline import SparseFeatureMapper, RatingPredictionOptimizer, DenseFeatureMapper
from autorecsys.pipeline.interactor_ts import Bi_LSTMInteractor, LSTMInteractor, \
    GRUInteractor, TokenAndPositionEmbedding, TransformerBlock, Transformers2, BertBlock, TransformerBlock3sas, SASREC
from autorecsys.pipeline.preprocessor_ts import MovielensPreprocessor

print(tf.config.list_physical_devices())
print(sys.version)
print(tf.__version__)
print('v1.4')


seed_value = 0
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
# random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
tf.random.set_seed(seed_value)




# logging setting
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# movielens = MovielensPreprocessor(csv_path="./example_datasets/movielens/", nrows=100000)
# train_X, train_y, val_X, val_y, test_X, test_y = movielens.preprocess()


csv_path = "./example_datasets/movielens/"
file_name = 'ratings.dat'
rnames = ['userID', 'movieID', 'rating', 'timestamp']
from pathlib import Path

ratings = pd.read_csv(Path(os.path.join(csv_path, file_name)), nrows=10000, sep='::', header=None, names=rnames,
                     engine='python')

# print(ratings.head(10))

reframed = ratings.sort_values(by=['userID', 'timestamp'], ascending=(True, False))
reframed = reframed.reset_index(drop=True)
counts = reframed['userID'].value_counts(dropna=False)
reframed = reframed[~reframed['userID'].isin(counts[counts < 10].index)]
ratings = reframed.reset_index(drop=True)



name = ratings['userID'].unique()
all_movies = ratings['movieID'].unique()

dfs = dict(tuple(ratings.groupby('userID')))
train_x = pd.DataFrame()
test_x = pd.DataFrame()
pad = 50
length = []
rnd_test = []
prob = pd.DataFrame()
for x in name:
    # adding dummy entries

    dfs[x] = dfs[x].tail(pad)

    if dfs[x].shape[0] < pad:
        padd = pad-dfs[x].shape[0]
        df1 = pd.DataFrame([dfs[x].mean()], index=list(range(padd)), columns=dfs[x].columns)
        df1['movieID'] = df1['movieID'].astype(int)
        df1.iloc[:,df1.columns.get_loc("movieID")] = 0
        dfs[x] = df1.append(dfs[x], ignore_index=True)

    # test = set(all_movies)-set(dfs[x]['movieID'].unique())
    # random_test = random.sample(test, (5))
    # random_test.append(dfs[x].iloc[-1,1])
    # prob = prob.append([random_test])

    test = set(all_movies)-set(dfs[x]['movieID'].unique())
    random_test = random.sample(test, (dfs[x].shape[0]))
    dfs[x]['random'] = random_test
    dfs[x].iloc[0, dfs[x].columns.get_loc("random")] = dfs[x].iloc[-1, dfs[x].columns.get_loc("movieID")]

    train_x = train_x.append(dfs[x].iloc[:-1, :])
    test_x = test_x.append(dfs[x].iloc[:49, :])

# pd.set_option('display.max_columns', None)
train_x = train_x.reset_index(drop=True)
test_x = test_x.reset_index(drop=True)
full = train_x
ttest = test_x



# full = full[['userID', 'movieID', 'rating']]


# create train
movie_sessions = full.groupby('userID')['movieID'].apply(lambda x: x.tolist()) \
    .reset_index().rename(columns={'movieID': 'movielist'})
rating_sessions = full.groupby('userID')['rating'].apply(lambda x: x.tolist()) \
    .reset_index().rename(columns={'rating': 'ratinglist'})

# def create_sequences(values, window_size, step_size):
#     sequences = []
#     start_index = 0
#     while True:
#         end_index = start_index + window_size
#         seq = values[start_index:end_index]
#         if len(seq) < window_size:
#             seq = values[-window_size:]
#             # if len(seq) == window_size:
#                 # sequences.append(seq)
#             break
#         sequences.append(seq)
#
#         start_index += step_size
#     return sequences
#
# sequence_length = 4
# step_size = 1
#
#
# movie_sessions['movielist'] = movie_sessions.movielist.apply(
#     lambda ids: create_sequences(ids, sequence_length, step_size))
#
# rating_sessions['ratinglist'] = rating_sessions.ratinglist.apply(
#     lambda ids: create_sequences(ids, sequence_length, step_size))
# movie_sessions = movie_sessions[["userID", "movielist"]].explode(
#     "movielist", ignore_index=True)
# rating_sessions = rating_sessions[["ratinglist"]].explode("ratinglist", ignore_index=True)

movie_sessions.movielist = movie_sessions.movielist.apply(
    lambda x: ",".join([str(v) for v in x])
)
rating_sessions.ratinglist = rating_sessions.ratinglist.apply(
    lambda x: ",".join([str(v) for v in x])
)


# m = movie_sessions.movielist.str.split(',', expand=True)
m = pd.concat([movie_sessions['userID'], movie_sessions.movielist.str.split(',', expand=True)], axis=1)
r = rating_sessions.ratinglist.str.split(',', expand=True)
# for column in m.columns:
#     m.rename(columns={column: str('movie')+str(column)}, inplace=True)
#     r.rename(columns={column: str('rating')+str(column)}, inplace=True)



# create test
test_movie_sessions = ttest.groupby('userID')['random'].apply(lambda x: x.tolist()) \
    .reset_index().rename(columns={'random': 'movielist'})
test_rating_sessions = ttest.groupby('userID')['rating'].apply(lambda x: x.tolist()) \
    .reset_index().rename(columns={'rating': 'ratinglist'})


# test_movie_sessions = test_movie_sessions[["userID", "movielist"]].explode(
#     "movielist", ignore_index=True
# )
# test_rating_sessions = test_rating_sessions[["userID", "ratinglist"]].explode("ratinglist", ignore_index=True)
#

test_movie_sessions.movielist = test_movie_sessions.movielist.apply(
    lambda x: ",".join([str(v) for v in x])
)
test_rating_sessions.ratinglist = test_rating_sessions.ratinglist.apply(
    lambda x: ",".join([str(v) for v in x])
)
mtest = pd.concat([test_rating_sessions['userID'], test_movie_sessions.movielist.str.split(',', expand=True)], axis=1)
rtest = pd.concat([test_rating_sessions['userID'], test_rating_sessions.ratinglist.str.split(',', expand=True)], axis=1)

# print(test_rating_sessions)
# print(rtest.tail(50))
# for column in mtest.columns:
#     mtest.rename(columns={column: str('movie')+str(column)}, inplace=True)
#     rtest.rename(columns={column: str('rating')+str(column)}, inplace=True)
# we need train x: seq_m, target_m
# we ned train y: target_r
train_xm = m.iloc[:, 1:].values
train_ym = r.iloc[:, -1].values


test_xm = mtest.iloc[:, 1:].values
test_ym = mtest.iloc[:, -1].values
print(r.iloc[:, 1:])
print(mtest.iloc[:, 1:])
# divide into sequence okay?

# print(test_yr.iloc[:,2].head())
# bbb = test_yr.iloc[:,2].head(10)
# print(bbb.values.argsort())
# ccc = bbb.values.argsort()
# print(ccc[:5])
#
# if 2 in ccc[:5]:
#     print('hit')
#
# # print(test_yr['2'])
#
# fff = pd.concat([rtest['userID'], pd.DataFrame(test_yr)], axis=1, ignore_index=True)
# fff.columns = ['userID', 'b', '1', '2', '3', '4']
# print(fff.head(10))
#
# hit=0
# dfss = dict(tuple(fff.groupby('userID')))
# names = fff['userID'].unique()
# for name in names:
#     print(dfss[name]['1'])
#     ccc = dfss[name]['1'].values.argsort()
#     if 3 in ccc[:5]:
#         print(ccc)
#         hit+=1
#     print(hit)




train_xm = train_xm.astype(float)
train_ym = train_ym.astype(float)


test_xm = test_xm.astype(float)
test_ym =test_ym.astype(float)

x_train = tf.keras.preprocessing.sequence.pad_sequences(train_xm, maxlen=50)


# positive is target value
# negative is all item - watched + positive
hash_size = None
# Step 2: Build the recommender, which provides search space
# Step 2.1: Setup mappers to handle inputs
# input_node = ak.Input(shape=(train_X.shape[1],))
# user_input_node = ak.Input(shape=[train_xu.shape[1]])
# dense_input_node = ak.Input(shape=[train_X_rating.shape[1]])
sparse_input_node = ak.Input(shape=[train_xm.shape[1]])
# positive_input_node = ak.Input(shape=[test_xm.shape[1]])
# negative_input_node = ak.Input(shape=[train_X_negative.shape[1]])


from autorecsys.pipeline.interactor import ElementwiseInteraction
from autorecsys.pipeline.interactor_ts import DistanceInteractor, FeedForward, SelfAttentionInteraction
# Step 2.2: Setup interactors to handle models
sparse_input = TokenAndPositionEmbedding(maxlen=train_xm.shape[1],
                                         vocab_size=2*len(ratings['movieID'].unique()))([sparse_input_node])
# dense_input = DenseFeatureMapper(
#     num_of_fields=train_X_rating.shape[1],
#     embedding_dim=2)(dense_input_node)
# positive_emb = SparseFeatureMapper(
#     num_of_fields=test_xm.shape[1],
#     hash_size=len(ratings['movieID'].unique()),
#     embedding_dim=2)(positive_input_node)


# sparse_input = TokenAndPositionEmbedding(maxlen=200, vocab_size=10000)([sparse_input_node])

# positive_emb = MLPInteraction(units=32)(positive_input_node)
# negative_emb = MLPInteraction()(negative_input_node)



transformer_block1 = TransformerBlock()([sparse_input])
# transformer_block2 = TransformerBlock()([sparse_input])
# transformer_block1 = TransformerBlock()([dense_input])

# out1 = MLPInteraction()([dense_input, sparse_input])
# transformer_block2 = FeedForward()([transformer_block1])
# transformer_block4 = MLPInteraction(units=32)([transformer_block1])


# transformer_block4 = MLPInteraction()([transformer_block1])
# transformer_block5 = MLPInteraction()([user_input, transformer_block4])


# transformer_block5 = MLPInteraction()([transformer_block3, negative_emb])

# transformer_block5 = SASREC()([transformer_block4, positive_emb])
# transformer_block6 = MLPInteraction()([transformer_block5])

# transformer_block1 = BertBlock(max_sequence_length=train_X.shape[1])(dense_input_node)
# transformer_block2 = BertBlock(max_sequence_length=train_X.shape[1])(sparse_input_node)
# transformer_block = MLPInteraction()([transformer_block3, transformer_block4])


# Step 2.3: Setup optimizer to handle the target task
# output1 = ak.RegressionHead()(transformer_block5)
output1 = ak.RegressionHead()(transformer_block1)

# Step 3: Build the searcher, which provides search algorithm
auto_model = ak.AutoModel(inputs=[sparse_input_node],
                          outputs=[output1],
                          max_trials=3,
                          overwrite=True)

# Step 4: Use the searcher to search the recommender
auto_model.fit(x=[train_xm],
               y=[train_ym],
               batch_size=32,
               epochs=100,
               callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=1)])



a = auto_model.predict(x=[test_xm], batch_size=32)

print(a[:50])

# for name in names:

# from autorecsys.pipeline.ts_eval import Eval_Topk
#
# print(test_X_positive[:20])
# print(a[:20])
# sys.exit()
#
# eval = Eval_Topk(pred=a[0], data=val_X, topk=5)
# acc = eval.HR_MRR_nDCG()
#
# # logger.info('Validation Accuracy (mse): {}'.format(auto_model.evaluate(x=[val_X],
# #                                                                      y=val_y)))
# # Step 5: Evaluate the searched model
