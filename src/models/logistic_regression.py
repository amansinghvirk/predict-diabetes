'''
Module: logistic_regression.py

Date: 30-Dec-2022
Authoer: Amandeep Singh

Module Classes:
    LogisticRegressionModel:
        - setup, train and save Logistic Regression Model

'''

import joblib
from sklearn.linear_model import LogisticRegression

# pylint: disable=too-many-arguments


class LogisticRegressionModel():
    '''
    class to create logistic regression model, train and save models
    ...

    Attributes
    ----------
        predictors: data of predictor variables as pandas dataframe
        target: data of target variable as pandas dataframe
        solver (default: lbfgs): solver to be used
        max_iter (default: 3000): maximum interactions for convergence

    Methods
    ----------

    get_model:
        method to get the logistic regression model object

    train_model:
        method to train the logistic regression classifer model

    save_model:
        save the logistic regression model to specified path
    '''

    __lrc = None
    __predictors = None
    __target = None

    def __init__(self, predictors, target,
                 solver='lbfgs', max_iter=3000):
        self.__lrc = None
        self.__predictors = predictors
        self.__target = target
        self.__lrc = LogisticRegression(
            solver=solver,
            max_iter=max_iter,
            verbose=False)

    def get_model(self):
        '''method returns the logistic regression model object'''
        return self.__lrc

    def train_model(self):
        '''method to train the logistic regression model'''
        self.__lrc.fit(
            self.__predictors,
            self.__target)

    def save_model(self, pth_to_save):
        '''
        method to save the trained logistic regression model as pickle file

        Arguments:
            pth_to_save: path specified to save the model file

        Output:
            pickle file saved at path
        '''
        joblib.dump(
            self.__lrc,
            pth_to_save)
