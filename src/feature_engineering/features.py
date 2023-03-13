'''
Module: features.py

Date: 30-Dec-2022
Authoer: Amandeep Singh

Module Classes:
    FeatureEngineering:
        - split data into train test

'''

from sklearn.model_selection import train_test_split


class FeatureEngineering():
    '''
    class to perform feature engineering, save and return data
    ...

    Attributes
    ----------
        predictors: data of predictor variables as pandas dataframe
        target: data of target variable as pandas dataframe

    Methods
    ----------
    perform_feature_engineering:
        method to perform the feature engineering on model data
        split the data into train and test

    get_train_data:
        method to get the train data after feature engineering

    save_train_data:
        method to save the train data after feature engineering

    get_test_data:
        method to get the test data after feature engineering

    get_test_data:
        method to save the test data after feature engineering

    get_train_target:
        method to get the train target variable data

    get_save_target:
        method to get the train target variable data

    get_test_target:
        method to get the test target variable data

    get_save_target:
        method to get the test target variable data

    '''

    __predictors = None
    __target = None
    __predictors_train = None
    __predictors_test = None
    __target_train = None
    __target_test = None

    def __init__(self, predictors, target):
        self.__predictors = predictors
        self.__target = target

    def perform_feature_engineering(
            self, test_size=0.3, random_state=42):
        '''
        Arguments:
            test_size: proportion of data to be kept for test
            random_state: random seed value

        output: None
        '''
        (self.__predictors_train, self.__predictors_test,
         self.__target_train, self.__target_test) = train_test_split(
            self.__predictors,
            self.__target,
            test_size=test_size,
            random_state=random_state)

    def get_train_data(self):
        '''method returns the train predictors variables data'''
        return self.__predictors_train

    def save_train_data(self, pth_to_save):
        '''
        method returns the save predictors variables data

        Arguments:
            pth_to_save: path and name of csv file to save data

        Output: None
        '''
        self.__predictors_train.to_csv(pth_to_save, index=False)

    def get_test_data(self):
        '''method returns the test predictor variables data'''
        return self.__predictors_test

    def save_test_data(self, pth_to_save):
        '''
        method returns the save test predictor variables data

        Arguments:
            pth_to_save: path and name of csv file to save data

        Output: None
        '''
        self.__predictors_test.to_csv(pth_to_save, index=False)

    def get_train_target(self):
        '''method returns the target variable from train dataset'''
        return self.__target_train

    def save_train_target(self, pth_to_save):
        '''
        method returns the target variable from train dataset

        Arguments:
            pth_to_save: path and name of csv file to save data

        Output: None
        '''
        self.__target_train.to_csv(pth_to_save, index=False)

    def get_test_target(self):
        '''method returns the target variable from test dataset'''
        return self.__target_test

    def save_test_target(self, pth_to_save):
        '''
        method returns the target variable from test dataset

        Arguments:
            pth_to_save: path and name of csv file to save data

        Output: None
        '''
        self.__target_test.to_csv(pth_to_save, index=False)
