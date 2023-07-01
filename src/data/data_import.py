'''
Module: data_import.py

Date: 30-Dec-2022
Author: Amandeep Singh

Module Classes:
    Data:
        - Imports the data
        - Performs the preprocessing

'''

import pandas as pd


class Data():
    '''
    class to setup data for modeling performs the following actions
        - import data
        - create target varaible
        - encode categorical variables
        - return predictors and target dataset
    ...

    Attributes
    ----------
    data_file_pth: path of csv file to read data from

    Methods
    ----------
    create_target_column():
        create a binary target column

    encode_categorical_columns():
        encode categorical columns to the proportion of response variable

    get_dataframe():
        returns the imported dataframe

    get_headrows():
        returns the top rows of the dataframe

    get_dataframe_columns():
        returns the dataframe column names as list

    get_predictors():
        returns the X (selected predictor columns) from the dataset

    get_target():
        returns the target column as pandas series
    '''

    __df = None
    __categorical_columns = None
    __predictor_columns = None
    __target = None

    def __init__(self, data_file_pth, categorical_cols,
                 target_col, predictor_cols):
        self.__import_data(data_file_pth)
        self.__categorical_columns = categorical_cols
        self.__target = target_col
        self.__predictor_columns = predictor_cols

    def __import_data(self, data_file_pth):
        '''read dataframe from the csv file at path provided'''
        self.__df = pd.read_csv(data_file_pth)

    def get_dataframe(self):
        '''
        method to get the data frame

        output:
            pandas dataframe
        '''
        return self.__df

    def get_headrows(self, nrows=10):
        '''
        returns the top rows of the dataframe

        Attributes:
            nrows (optional, default=10): number of rows to return

        output:
            pandas dataframe
        '''
        return self.__df.head(nrows)

    def get_dataframe_columns(self):
        '''
        method to get the column names of dataframe

        output: list of column names
        '''
        return self.__df.columns

    def get_dataframe_shape(self):
        '''
        method to get the data frame shape

        output: tuple specifying rows and columns
        '''
        return self.__df.shape

    def create_target_column(self, col_name, specified_value):
        '''
        method will create a target column based on values specified for given
        column, specified value will be 0 other as 1

        output: None
        '''

        self.__df[self.__target] = self.__df[col_name].apply(
            lambda val: 0 if val == specified_value else 1)

    def encode_categorical_columns(self):
        '''
        method to turn each categorical column into a new column with
        propotion of target column for each category

        -- uses the data frame, target variable and categorical columns
        initiazlied while createing the object

        output: None
        '''

        # create new encoded columns for categorical columns
        for col in self.__categorical_columns:
            new_col = col + '_' + self.__target
            self.__df[new_col] = (self.__df
                                  .groupby(col)[self.__target]
                                  .transform('mean'))

    def get_predictors(self):
        '''method returns the pandas dataframe of predictor variables'''
        return self.__df.loc[:, self.__predictor_columns]

    def get_target(self):
        '''method returns the pandas dataframe of target variable'''
        return self.__df[self.__target]
