# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

from abc import ABCMeta, abstractmethod
from typing import List

from sklearn.model_selection import train_test_split
from collections import defaultdict

import pandas as pd
import numpy as np
import sys
import time
import gc
import os
import math
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import pprint
import random

#from autorecsys.utils.common import load_pickle, save_pickle


class BasePreprocessor(metaclass=ABCMeta):
    """ Preprocess data into Pandas DataFrame format.

    # Arguments
        non_csv_path (str): Path to convert the dataset into CSV format.
        csv_path (str): Path to save and load the CSV dataset.
        header (int): Row number to use as column names.
        columns (list): String names associated with the columns of the dataset.
        delimiter (str): Separator used to parse lines.
        filler (float): Filler value used to fill missing data.
        dtype_dict (dict): Map string column names to column data type.
        ignored_columns (list): String names associated with the columns to ignore.
        target_column (str): String name associated with the columns containing target data for prediction, e.g.,
            rating column for rating prediction and label column for click-through rate (CTR) prediction.
        numerical_columns (list): String names associated with the columns containing numerical data.
        categorical_columns (list): String names associated with the columns containing categorical data.
        categorical_filter (int): Filter used to group infrequent categories in one column as the same category.
        fit_dictionary_path (str): Path to the fit dictionary for categorical data.
        transform_path (str): Path to the transformed dataset.
        test_percentage (float): Percentage for the test set.
        validate_percentage (float): Percentage for the validation set.
        train_path (str): Path to the training set.
        validate_path (str): Path to the validation set.
        test_path (str): Path to the test set.

    # Attributes
        non_csv_path (str): Path to convert the dataset into CSV format.
        csv_path (str): Path to save and load the CSV dataset.
        header (int): Row number to use as column names.
        columns (list): String names associated with the columns of the dataset.
        delimiter (str): Separator used to parse lines.
        filler (float): Filler value used to fill missing data.
        dtype_dict (dict): Map string column names to column data type.
        ignored_columns (list): String names associated with the columns to ignore.
        data_df (DataFrame): Data loaded in Pandas DataFrame format and contains only relevant columns.
        target_column (str): String name associated with the columns containing target data for prediction, e.g.,
            rating column for rating prediction and label column for click-through rate (CTR) prediction.
        numerical_columns (list): String names associated with the columns containing numerical data.
        categorical_columns (list): String names associated with the columns containing categorical data.
        categorical_filter (int): Filter used to group infrequent categories in one column as the same category.
        fit_dict (dict): Map string categorical column names to dictionary which maps categories to indices.
        fit_dictionary_path (str): Path to the fit dictionary for categorical data.
        transform_path (str): Path to the transformed dataset.
        test_percentage (float): Percentage for the test set.
        validate_percentage (float): Percentage for the validation set.
        train_path (str): Path to the training set.
        validate_path (str): Path to the validation set.
        test_path (str): Path to the test set.
    """

    @abstractmethod
    def __init__(self,
                 non_csv_path=None,
                 csv_path=None,
                 header=None,
                 columns=None,
                 delimiter=None,
                 splitby=None,
                 nrows=None,
                 lcount=None,
                 hcount=None,
                 filler=None,
                 dtype_dict=None,
                 ignored_columns=None,
                 target_column=None,
                 numerical_columns=None,
                 categorical_columns=None,
                 categorical_filter=None,
                 fit_dictionary_path=None,
                 transform_path=None,
                 test_percentage=None,
                 validate_percentage=None,
                 train_path=None,
                 validate_path=None,
                 test_path=None):

        super().__init__()
        # Dataset load attributes.
        self.non_csv_path = non_csv_path
        self.csv_path = csv_path
        self.header = header
        self.columns = columns
        self.delimiter = delimiter
        self.splitby = splitby
        self.nrows = nrows
        self.lcount = lcount
        self.hcount = hcount
        self.filler = filler
        self.dtype_dict = dtype_dict
        self.ignored_columns = ignored_columns
        self.data_df = None

        # Dataset access attributes.
        self.target_column = target_column
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns

        # Dataset transformation attributes.
        self.categorical_filter = categorical_filter
        self.fit_dict = None
        self.fit_dictionary_path = fit_dictionary_path
        self.transform_path = transform_path

        # Dataset split attributes.
        self.test_percentage = test_percentage
        self.validate_percentage = validate_percentage
        self.train_path = train_path
        self.validate_path = validate_path
        self.test_path = test_path

        # after preprocess info
        self.info = []

    def format_dataset(self):
        """ (Optional) Convert dataset into CSV format.

        # Note
            User should implement this function to convert non-CSV dataset into CSV format.
        """
        raise NotImplementedError

    def load_dataset(self, timestamp): # pragma: no cover
        """ Load CSV data as a Pandas DataFrame object.
        """
        self.data_df = pd.read_csv(self.csv_path, sep=',', nrows=self.nrows, index_col=0,
                              engine='python')
        self.info.append(('Original data shape', self.data_df.shape))
        #self.data_df.drop(columns=self.ignored_columns, inplace=True)
        self.data_df.fillna(self.filler, inplace=True)

        df_cat_tmp = self.data_df['gender'].astype('category')
        df_cat_tmp = df_cat_tmp.unique()
        cat_to_int = {word: ii for ii, word in enumerate(df_cat_tmp, 1)}
        self.data_df['gender'] = self.data_df['gender'].map(cat_to_int)

        scaler = MinMaxScaler(feature_range=(0, 1))
        self.data_df[[timestamp]] = scaler.fit_transform(self.data_df[[timestamp]])

        tmp = self.data_df['rating']
        self.data_df = self.data_df.drop(columns=['rating'])
        self.data_df = pd.concat([self.data_df, tmp], axis=1)
        del tmp

        reframed = self.data_df.sort_values(by=['userID', timestamp], ascending=(True, False))
        reframed = reframed.reset_index(drop=True)
        counts = reframed['userID'].value_counts(dropna=False)
        reframed = reframed[~reframed['userID'].isin(counts[counts < self.lcount].index)]
        reframed = reframed[~reframed['userID'].isin(counts[counts > self.hcount].index)]
        self.data_df = reframed.reset_index(drop=True)

        # print(self.data_df)
        # datao = self.data_df.loc[self.data_df['userID'].isin([15])]
        # datax = self.data_df.loc[self.data_df['userID'].isin([10])]
        #
        # import matplotlib.pyplot as plt
        # datax.plot.scatter(x="timestamp", y="userID")
        # plt.show()
        # sys.exit()

        # drop timestamp
        # self.data_df = self.data_df.drop([self.splitby], 1)
        # self.data_df = self.data_df.reset_index(drop=True)

        self.info.append(('Counts low & high', [counts.iloc[-1], counts.iloc[0]]))
        self.info.append(('After counts data shape', reframed.shape))
        # topMovies = self.data_df.movieID.value_counts().head(600).index.tolist()
        # self.data_df = self.data_df.loc[self.data_df.movieID.isin(topMovies)]

    def merge_datasets(self, timestep):
        mnames = ['movieID', 'title', 'genre1']
        rnames = ['userID', 'movieID', 'rating', 'timestamp']
        unames = ['userID', 'gender', 'age', 'occupation', 'zipcode']

        movies = pd.read_csv(Path(os.path.join(self.csv_path, 'movies.dat')), sep='::', header=None, names=mnames, engine='python')
        ratings = pd.read_csv(Path(os.path.join(self.csv_path, 'ratings.dat')), sep='::', header=None, names=rnames, engine='python')
        users = pd.read_csv(Path(os.path.join(self.csv_path, 'users.dat')), sep='::', header=None, names=unames, engine='python')

        tmp = movies['genre1'].str.split('|', expand=True)
        tmp = tmp.stack().str.get_dummies().sum(level=0)
        # pd.set_option('display.max_columns', None)
        # print(tmp)
        movies = pd.concat([movies, tmp], axis=1)
        movies = movies.drop('genre1', 1)
        # df_cat_tmp = movies['genres'].astype('category')
        # df_cat_tmp = df_cat_tmp.unique()
        # cat_to_int = {word: ii for ii, word in enumerate(df_cat_tmp, 1)}
        # movies['genres'] = movies['genres'].map(cat_to_int)

        df_tmp = movies['title'].str.extract('(\(\d.{3})', expand=False)
        movies['releaseDate'] = df_tmp.str.extract('(\d+)', expand=False)

        ratings = self.convert_timestamp(ratings, timestep)

        movies = movies.drop(columns=['title'])
        users = users.drop(columns=['zipcode'])

        Merged_movies_ratings = pd.merge(movies, ratings, on="movieID")
        final_merge = pd.merge(users, Merged_movies_ratings, on="userID")
        df = pd.DataFrame(final_merge)

        tmp = 'combined_{}.csv'.format(timestep)
        filepath = Path(os.path.join(self.csv_path, tmp))

        if not os.path.isfile(filepath):
            df.to_csv(filepath)
            del df
        else:  # else it exists so append without writing the header
            df.to_csv(filepath)
            del df

    def convert_timestamp(self, ratings, condition='timestamp'):
        hourly = 3600
        daily = 86400  # second to day
        yearly = 31536000
        minute = 60
        seconds = 0
        # choose 1) hourly, 2)daily, 3) yearly
        if condition == 'hourly':
            ratings[condition] = np.ceil(ratings['timestamp'] / hourly)
            ratings = ratings.sort_values([condition], ascending=True).drop('timestamp', 1)
        elif condition == 'daily':
            ratings[condition] = np.ceil(ratings['timestamp'] / daily)
            ratings = ratings.sort_values([condition], ascending=True).drop('timestamp', 1)
        elif condition == 'yearly':
            ratings[condition] = np.ceil(ratings['timestamp'] / yearly)
            ratings = ratings.sort_values([condition], ascending=True).drop('timestamp', 1)
        elif condition == 'minute':
            ratings[condition] = np.ceil(ratings['timestamp'] / minute)
            ratings = ratings.sort_values([condition], ascending=True).drop('timestamp', 1)
        else:
            pass

        return ratings



################ MovieLens ################


class MovielensPreprocessor(BasePreprocessor):
    """ Preprocess the Movielens 1M dataset for rating prediction.
    """

    def __init__(self,
                 non_csv_path=None,
                 csv_path='./example_datasets/movielens',
                 header=None,  # inferred in load_data()
                 columns=None,
                 delimiter=',',
                 splitby='timestamp',
                 nrows=100000,
                 lcount=100,
                 hcount=1000,
                 filler=0.0,
                 dtype_dict=None,
                 ignored_columns=None,
                 target_column='Rating',
                 numerical_columns=None,
                 categorical_columns=None,
                 categorical_filter=0,  # no grouping, simply renumber
                 fit_dictionary_path=None,
                 transform_path=None,
                 test_percentage=0.1,
                 validate_percentage=0.1,
                 train_path=None,
                 validate_path=None,
                 test_path=None):

        self.full = pd.DataFrame()
        if ignored_columns is None:
            ignored_columns = ['title', 'zipcode']
        if columns is None:
            columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']
        if dtype_dict is None:
            dtype_dict = {'UserID': np.int32, 'MovieID': np.int32, 'Rating': np.int32, 'Timestamp': np.int32}
        if ignored_columns is None:
            ignored_columns = ['title', 'zipcode']
        if numerical_columns is None:
            numerical_columns = []
        if categorical_columns is None:
            categorical_columns = columns[:2]

        super().__init__(non_csv_path=non_csv_path,
                         csv_path=csv_path,
                         header=header,
                         delimiter=delimiter,
                         splitby=splitby,
                         nrows=nrows,
                         lcount=lcount,
                         hcount=hcount,
                         filler=filler,
                         dtype_dict=dtype_dict,
                         columns=columns,
                         ignored_columns=ignored_columns,
                         target_column=target_column,
                         numerical_columns=numerical_columns,
                         categorical_columns=categorical_columns,
                         categorical_filter=categorical_filter,
                         fit_dictionary_path=fit_dictionary_path,
                         transform_path=transform_path,
                         test_percentage=test_percentage,
                         validate_percentage=validate_percentage,
                         train_path=train_path,
                         validate_path=validate_path,
                         test_path=test_path)

    def preprocess(self):
        """
        """
        # check if ratings.dat, users.dat, movie.dat files exist
        filenames = ['movies.dat', 'ratings.dat', 'users.dat']
        for i in range(len(filenames)):
            filepath = Path(os.path.join(self.csv_path, filenames[i]))
            try:
                filepath_ = filepath.resolve(strict=True)
            except FileNotFoundError:
                print('FileNotFoundError: Check if file " {} " exists '.format(filepath))
                print('\t -> Provide the path where the files are')
                sys.exit('File Path Error')
            else:
                pass

        # combine ratings.dat, users.dat, movie.dat files
        combpath = Path(os.path.join(self.csv_path, 'combined_{}.csv'.format(self.splitby)))
        if combpath.exists():
            pass
        else:
            # timestep = 'yearly', 'monthly', 'daily', 'minute', 'timestamp'
            self.merge_datasets(timestep=self.splitby)
        self.csv_path = combpath

        # Step 1: Load data for fit and transform categorical data.
        self.load_dataset(self.splitby)

        # Step 2:
        self.train, self.test, self.val = self.split_id()
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)



        # random x_val y_val
        x_validate = self.val
        y_validate = self.val
        # randomly shuffle test set
        # self.test = self.test.sample(frac=1, random_state=42).reset_index(drop=True)
        # self.test = self.test.sort_values(by=['userID'], ascending=(True))


        train = self.train.values
        test = self.test.values


        x_train, y_train = train[:, :-1], train[:, -1]
        x_test, y_test = test[:, :-1], test[:, -1]

        pprint.pprint(self.info)

        return x_train, y_train, x_validate, y_validate, x_test, y_test

    def split_id(self):
        # self.data_df.userID = self.data_df.userID.rank(method='dense', ascending=True).astype(int)
        # self.data_df.movieID = self.data_df.movieID.rank(method='dense', ascending=True).astype(int)
        name = self.data_df['userID'].unique()
        dfs = dict(tuple(self.data_df.groupby('userID')))
        train_x = pd.DataFrame()
        test_x = pd.DataFrame()
        val_x = pd.DataFrame()
        pad = 20
        length=[]
        all_movies = self.data_df['movieID'].unique()
        print('# items:', len(all_movies))
        print('# users', len(self.data_df['userID'].unique()))
        print('# interactions', self.data_df.shape[0])



        self.users_item={}
        for x in name:
            length.append(dfs[x].shape[0])
        avg = int(sum(length)/len(length))

        batch_item = 100
        batch_item = batch_item

        next_prediction = False
        random_num = []
        check = []
        zz = 0
        movies_genre = self.data_df[['movieID', 'releaseDate', 'Action', 'Adventure',
                                       'Animation', "Children's", 'Comedy', "Crime", 'Documentary', 'Drama',
                                       'Fantasy', "Film-Noir", 'Horror', 'Musical', 'Mystery', 'Romance',
                                       'Sci-Fi', 'Thriller', 'War', 'Western']]

        result_df = movies_genre.drop_duplicates()



        for x in name:
            train = dfs[x].head(-5)
            test = dfs[x].tail(5)

            neg_movies = set(all_movies) - set(dfs[x]['movieID'].unique())
            random_test = random.sample(neg_movies, (batch_item - 1))
            b = result_df[result_df.movieID.isin(random_test)]

            concatt = test[['movieID', 'releaseDate', 'Action', 'Adventure',
             'Animation', "Children's", 'Comedy', "Crime", 'Documentary', 'Drama',
             'Fantasy', "Film-Noir", 'Horror', 'Musical', 'Mystery', 'Romance',
             'Sci-Fi', 'Thriller', 'War', 'Western']]
            # test_x = test_x.reset_index(drop=True)
            test = test.reset_index(drop=True)
            end = pd.concat([concatt, b], axis=0)
            user_data = pd.DataFrame()
            user_data = user_data.append([test[['userID', 'age', 'gender', 'occupation', 'rating']].iloc[1,:]]*end.shape[0])
            user_data = user_data.reset_index(drop=True)
            end = end.reset_index(drop=True)

            test = pd.concat([user_data, end], axis=1)
            val = test.sample(frac=1, random_state=42).reset_index(drop=True)

            train_x = train_x.append(train.tail(50))
            test_x = test_x.append(test)
            val_x = val_x.append(val)
            # self.full = self.full.append(dfs[x])
        # self.random_num = random_num
        # self.check = check
        # test_x = test_x.reset_index(drop=True)
        self.info.append(('After padding data shape', (train_x.shape[0]+test_x.shape[0])))
        return train_x, test_x, val_x



# def convert_timestamp(ratings, condition='timestamp'):
#     hourly = 3600
#     daily = 86400 # second to day
#     yearly = 31536000
#     minute = 60
#     seconds = 0
#     # choose 1) hourly, 2)daily, 3) yearly
#     if condition == 'hourly':
#         ratings[condition]=np.ceil(ratings['timestamp']/hourly)
#         ratings=ratings.sort_values([condition], ascending=True).drop('timestamp',1)
#     elif condition == 'daily':
#         ratings[condition]=np.ceil(ratings['timestamp']/daily)
#         ratings =ratings.sort_values([condition], ascending=True).drop('timestamp',1)
#     elif condition == 'yearly':
#         ratings[condition]=np.ceil(ratings['timestamp']/yearly)
#         ratings =ratings.sort_values([condition], ascending=True).drop('timestamp',1)
#     elif condition == 'minute':
#         ratings[condition]=np.ceil(ratings['timestamp']/minute)
#         ratings =ratings.sort_values([condition], ascending=True).drop('timestamp',1)
#     else:
#         pass
#
#     return ratings











################ LastFM ################




class LastFMPreprocessor(BasePreprocessor):
    """ Preprocess the Movielens 1M dataset for rating prediction.
    """

    def __init__(self,
                 non_csv_path=None,
                 csv_path='./example_datasets/movielens',
                 header=None,  # inferred in load_data()
                 columns=None,
                 delimiter='::',
                 splitby='timestamp',
                 nrows=100000,
                 lcount=200,
                 hcount=1000,
                 filler=0.0,
                 dtype_dict=None,
                 ignored_columns=None,
                 target_column='Rating',
                 numerical_columns=None,
                 categorical_columns=None,
                 categorical_filter=0,  # no grouping, simply renumber
                 fit_dictionary_path=None,
                 transform_path=None,
                 test_percentage=0.1,
                 validate_percentage=0.1,
                 train_path=None,
                 validate_path=None,
                 test_path=None):

        if ignored_columns is None:
            ignored_columns = ['userID', 'artistID', 'tagID', 'timestamp']
        if columns is None:
            columns = ['userID', 'artistID', 'tagID', 'timestamp']
        if dtype_dict is None:
            dtype_dict = {'userID': np.int32, 'artistID': np.int32, 'tagID': np.int32, 'timestamp': np.int32}
        if numerical_columns is None:
            numerical_columns = []
        if categorical_columns is None:
            categorical_columns = columns[:2]

        super().__init__(non_csv_path=non_csv_path,
                         csv_path=csv_path,
                         header=header,
                         delimiter=delimiter,
                         splitby=splitby,
                         nrows=nrows,
                         lcount=lcount,
                         hcount=hcount,
                         filler=filler,
                         dtype_dict=dtype_dict,
                         columns=columns,
                         ignored_columns=ignored_columns,
                         target_column=target_column,
                         numerical_columns=numerical_columns,
                         categorical_columns=categorical_columns,
                         categorical_filter=categorical_filter,
                         fit_dictionary_path=fit_dictionary_path,
                         transform_path=transform_path,
                         test_percentage=test_percentage,
                         validate_percentage=validate_percentage,
                         train_path=train_path,
                         validate_path=validate_path,
                         test_path=test_path)

    def preprocess(self):
        """f
        """
        # check if ratings.dat, users.dat, movie.dat files exist
        # Step 1: Load data for fit and transform categorical data.

        self.data_df = pd.read_csv(self.csv_path, sep='\t',
                           header=0,
                           names=self.columns, nrows=self.nrows, engine='python')

        # Step 2:
        self.train, self.test = self.split_id()
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)


        # print(self.train)
        print('\n\n start')
        # print(self.test)

        print(self.train.shape, self.test.shape)
        # random x_val y_val
        x_validate = self.test
        y_validate = self.test

        train = self.train.values
        test = self.test.values
        x_train, y_train = train[:, :-1], train[:, -1]
        x_test, y_test = test[:, :-1], test[:, -1]

        return x_train, y_train, x_validate, y_validate, x_test, y_test

    def split_id(self):
        name = self.data_df['userID'].unique()
        dfs = dict(tuple(self.data_df.groupby('userID')))
        train_x = pd.DataFrame()
        test_x = pd.DataFrame()
        for x in name:
            train, test = train_test_split(dfs[x], test_size=0.2, shuffle=False)
            train_x = train_x.append(train)
            test_x = test_x.append(test)
        return train_x, test_x




class NetflixPrizePreprocessor(BasePreprocessor):
    """ Preprocess the Netflix dataset for rating prediction.

    # Note
        To obtain the Netflix dataset, visit: https://www.kaggle.com/netflix-inc/netflix-prize-data
        The Netflix dataset has 4 data columns: MovieID, CustomerID, Rating, and Date.

    # Arguments
        non_csv_path (str): Path to convert the dataset into CSV format.
        csv_path (str): Path to save and load the CSV dataset.
        header (int): Row number to use as column names.
        columns (list): String names associated with the columns of the dataset.
        delimiter (str): Separator used to parse lines.
        filler (float): Filler value used to fill missing data.
        dtype_dict (dict): Map string column names to column data type.
        ignored_columns (list): String names associated with the columns to ignore.
        target_column (str): String name associated with the columns containing target data for prediction, e.g.,
            rating column for rating prediction and label column for click-through rate (CTR) prediction.
        numerical_columns (list): String names associated with the columns containing numerical data.
        categorical_columns (list): String names associated with the columns containing categorical data.
        categorical_filter (int): Filter used to group infrequent categories in one column as the same category.
        fit_dictionary_path (str): Path to the fit dictionary for categorical data.
        transform_path (str): Path to the transformed dataset.
        test_percentage (float): Percentage for the test set.
        validate_percentage (float): Percentage for the validation set.
        train_path (str): Path to the training set.
        validate_path (str): Path to the validation set.
        test_path (str): Path to the test set.
    """

    def __init__(self,
                 non_csv_path='./example_datasets/netflix/combined_data_1-300k.txt',
                 csv_path='./example_datasets/netflix/combined_data_1-300k.csv',
                 header=None,  # inferred in load_data()
                 columns=None,
                 delimiter=',',
                 filler=0.0,
                 dtype_dict=None,
                 ignored_columns=None,
                 target_column='Rating',
                 numerical_columns=None,
                 categorical_columns=None,
                 categorical_filter=0,  # no grouping, simply renumber
                 fit_dictionary_path=None,
                 transform_path=None,
                 test_percentage=0.1,
                 validate_percentage=0.1,
                 train_path=None,
                 validate_path=None,
                 test_path=None):

        if columns is None:
            columns = ['movieID', 'userID', 'rating', 'timestamp']
        if dtype_dict is None:
            dtype_dict = {'movieID': np.int32, 'userID': np.int32, 'rating': np.int32, 'timestamp': np.int32}
        if ignored_columns is None:
            ignored_columns = columns[3]
        if numerical_columns is None:
            numerical_columns = []
        if categorical_columns is None:
            categorical_columns = columns[:2]

        super().__init__(non_csv_path=non_csv_path,
                         csv_path=csv_path,
                         header=header,
                         delimiter=delimiter,
                         filler=filler,
                         dtype_dict=dtype_dict,
                         columns=columns,
                         ignored_columns=ignored_columns,
                         target_column=target_column,
                         numerical_columns=numerical_columns,
                         categorical_columns=categorical_columns,
                         categorical_filter=categorical_filter,
                         fit_dictionary_path=fit_dictionary_path,
                         transform_path=transform_path,
                         test_percentage=test_percentage,
                         validate_percentage=validate_percentage,
                         train_path=train_path,
                         validate_path=validate_path,
                         test_path=test_path)

    def format_dataset(self):
        """ Convert the Netflix Prize dataset into CSV format and save it as a new file.

        # Note:
            This is an example showing the function which converts dataset into the CSV format as provided by user.
        """
        with open(self.non_csv_path, 'r') as rf, open(self.csv_path, 'w') as wf:
            for line in rf:
                if ':' in line:
                    movie = line.strip(":\n")
                else:
                    wf.write(movie + ',' + line)

    def preprocess(self):
        """ Apply all preprocessing steps to the Netflix Prize dataset.

        # Returns
            6-tuple of ndarray training input data, training output data, validation input data, validation output data,
                testing input data, and testing output data.
        """
        # Step 1: Convert Netflix dataset to CSV format.
        self.format_dataset()

        self.data_df = pd.read_csv(self.csv_path, sep=',',
                           header=None,
                           names=self.columns, nrows=self.nrows, engine='python')
        # self.data_df['timestamp'] = pd.to_datetime(self.data_df['timestamp'], format='%Y-%m-%d')

        # self.data_df['timestamp'] = pd.to_datetime(self.data_df.timestamp, format='%Y-%m-%d')
        # self.data_df[['timestamp']] = self.data_df['timestamp'].dt.strftime("%Y%m%d")

        reframed = self.data_df.sort_values(by=['movieID', 'timestamp'], ascending=(True, True))
        counts = reframed['movieID'].value_counts(dropna=False)
        reframed = reframed[~reframed['movieID'].isin(counts[counts < 300].index)]
        reframed = reframed[~reframed['movieID'].isin(counts[counts > 2000].index)]
        self.data_df = reframed.reset_index(drop=True)
        self.data_df = self.data_df.drop(columns=['timestamp'])

        scaler = MinMaxScaler(feature_range=(0, 100))
        self.data_df[['userID']] = scaler.fit_transform(self.data_df[['userID']])

        self.data_df.rename(columns={"movieID": "userID", "userID": "movieID"}, inplace=True)

        self.train, self.test = self.split_id()


        x_validate = self.test
        y_validate = self.test

        train = self.train.values
        test = self.test.values
        x_train, y_train = train[:, :-1], train[:, -1]
        x_test, y_test = test[:, :-1], test[:, -1]

        # Step 2: Load data for fit and transform categorical data.

        # Step 3: Transform categorical data.

        return x_train, y_train, x_validate, y_validate, x_test, y_test

    def split_id(self):
        name = self.data_df['userID'].unique()
        dfs = dict(tuple(self.data_df.groupby('userID')))
        train_x = pd.DataFrame()
        test_x = pd.DataFrame()
        for x in name:
            train, test = train_test_split(dfs[x], test_size=0.2, shuffle=False)
            train_x = train_x.append(train)
            test_x = test_x.append(test)
        return train_x, test_x




################ Amazon Beauty ################

class AmazonBeautyPreprocessor(BasePreprocessor):
    """ Preprocess the Movielens 1M dataset for rating prediction.
    """

    def __init__(self,
                 non_csv_path="http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Sports_and_Outdoors.csv",
                 csv_path=None,
                 header=None,  # inferred in load_data()
                 columns=None,
                 delimiter='::',
                 splitby='timestamp',
                 nrows=100000,
                 lcount=30,
                 hcount=1000,
                 filler=0.0,
                 dtype_dict=None,
                 ignored_columns=None,
                 target_column='Rating',
                 numerical_columns=None,
                 categorical_columns=None,
                 categorical_filter=0,  # no grouping, simply renumber
                 fit_dictionary_path=None,
                 transform_path=None,
                 test_percentage=0.1,
                 validate_percentage=0.1,
                 train_path=None,
                 validate_path=None,
                 test_path=None):

        if ignored_columns is None:
            ignored_columns = ['userID', 'artistID', 'tagID', 'timestamp']
        if columns is None:
            columns = ['userID', 'artistID', 'tagID', 'timestamp']
        if dtype_dict is None:
            dtype_dict = {'userID': np.int32, 'artistID': np.int32, 'tagID': np.int32, 'timestamp': np.int32}
        if numerical_columns is None:
            numerical_columns = []
        if categorical_columns is None:
            categorical_columns = columns[:2]

        super().__init__(non_csv_path=non_csv_path,
                         csv_path=csv_path,
                         header=header,
                         delimiter=delimiter,
                         splitby=splitby,
                         nrows=nrows,
                         lcount=lcount,
                         hcount=hcount,
                         filler=filler,
                         dtype_dict=dtype_dict,
                         columns=columns,
                         ignored_columns=ignored_columns,
                         target_column=target_column,
                         numerical_columns=numerical_columns,
                         categorical_columns=categorical_columns,
                         categorical_filter=categorical_filter,
                         fit_dictionary_path=fit_dictionary_path,
                         transform_path=transform_path,
                         test_percentage=test_percentage,
                         validate_percentage=validate_percentage,
                         train_path=train_path,
                         validate_path=validate_path,
                         test_path=test_path)

    def preprocess(self):
        """f
        """
        # check if ratings.dat, users.dat, movie.dat files exist
        # Step 1: Load data for fit and transform categorical data.

        self.data_df = pd.read_csv(self.non_csv_path, names=["userID", "movieID", "rating", "timestamp"], nrows=self.nrows)

        reframed = self.data_df.sort_values(by=['userID', 'timestamp'], ascending=(True, True))
        reframed = reframed.reset_index(drop=True)
        counts = reframed['userID'].value_counts(dropna=False)
        reframed = reframed[~reframed['userID'].isin(counts[counts < self.lcount].index)]
        reframed = reframed[~reframed['userID'].isin(counts[counts > self.hcount].index)]
        self.data_df = reframed.reset_index(drop=True)


        # TODO: labelencoder ".values.ravel()" warning if not used
        from sklearn.preprocessing import LabelEncoder
        labelencoder = LabelEncoder()  # initializing an object of class LabelEncoder
        self.data_df[['userID']] = labelencoder.fit_transform(self.data_df[['userID']].values.ravel())  # fitting and transforming the desired categorical column.
        self.data_df[['movieID']] = labelencoder.fit_transform(self.data_df[['movieID']].values.ravel())  # fitting and transforming the desired categorical column.
        self.data_df[['movieID']] += 1



        # Step 2:
        self.train, self.test, self.val = self.split_id()
        # pd.set_option('display.max_columns', None)
        # pd.set_option('display.max_rows', None)


        # print(self.train)
        # print(self.test)

        print(self.train.shape, self.test.shape)
        # random x_val y_val
        x_validate = self.val
        y_validate = self.val

        train = self.train.values
        test = self.test.values
        x_train, y_train = train[:, :-1], train[:, -1]
        x_test, y_test = test[:, :-1], test[:, -1]

        return x_train, y_train, x_validate, y_validate, x_test, y_test

    def split_id(self):
        name = self.data_df['userID'].unique()
        print(name)
        print('avg user', self.data_df.shape[0] / name.shape[0])
        dfs = dict(tuple(self.data_df.groupby('userID')))
        train_x = pd.DataFrame()
        test_x = pd.DataFrame()
        all_movies = self.data_df['movieID'].unique()
        batch_item = 20

        movies_genre = self.data_df[['movieID']]

        result_df = movies_genre.drop_duplicates()
        val_x = pd.DataFrame()

        print(self.data_df.shape)


        for x in name:
            print('11')
            train = dfs[x].head(-5)
            test = dfs[x].tail(5)
            neg_movies = set(all_movies) - set(dfs[x]['movieID'].unique())
            random_test = random.sample(neg_movies, (batch_item - 1))

            b = result_df[result_df.movieID.isin(random_test)]

            concatt = test[['movieID']]
            # test_x = test_x.reset_index(drop=True)
            test = test.reset_index(drop=True)
            end = pd.concat([concatt, b], axis=0)
            user_data = pd.DataFrame()
            user_data = user_data.append([test[['userID', 'rating', 'timestamp']].iloc[1,:]]*end.shape[0])
            user_data = user_data.reset_index(drop=True)
            end = end.reset_index(drop=True)

            test = pd.concat([user_data, end], axis=1)
            val = test.sample(frac=1, random_state=42).reset_index(drop=True)

            train_x = train_x.append(train.tail(20))
            test_x = test_x.append(test)
            val_x = val_x.append(val)

        return train_x, test_x, val_x