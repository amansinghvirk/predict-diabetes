'''
Module: random_forest.py

Date: 30-Dec-2022
Author: Amandeep Singh

Module Classes:
    RandomForestModel:
        - setup, train and save Random Forest Model

'''

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# pylint: disable=too-many-arguments


class RandomForestModel():
    '''
    class to create model random forest model,
    train and save models
    ...

    Attributes
    ----------
        predictors: data of predictor variables as pandas dataframe
        target: data of target variable as pandas dataframe
        random_state: random seed value
        param_grid: hyperparameter options
        cross_fold (default: 5): cross validation fold values

    Methods
    ----------
    get_model:
        method to get the grid search cv model object

    train_model:
        method to train the random forest model using grid search cv

    save_model:
        save the random forest model to specified path
    '''

    __rfc = None
    __cv_rfc = None
    __predictors = None
    __target = None

    def __init__(self, predictors, target, random_state,
                 param_grid, cross_fold=5):
        self.__predictors = predictors
        self.__target = target
        self.__rfc = RandomForestClassifier(
            random_state=random_state)
        self.__cv_rfc = GridSearchCV(
            estimator=self.__rfc,
            param_grid=param_grid,
            cv=cross_fold,
            verbose=False)

    def get_model(self):
        '''method returns the grid search model object'''
        return self.__cv_rfc.best_estimator_

    def train_model(self):
        '''method to train the random forest model using grid search'''
        self.__cv_rfc.fit(
            self.__predictors,
            self.__target)

    def save_model(self, pth_to_save):
        '''
        method to save the trained random forest model as pickle file

        Arguments:
            pth_to_save: path specified to save the model file

        Output:
            pickle file saved at path
        '''
        joblib.dump(
            self.__cv_rfc.best_estimator_,
            pth_to_save)
