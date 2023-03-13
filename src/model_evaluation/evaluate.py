
'''
Module: model_analysis.py

Date: 30-Dec-2022
Authoer: Amandeep Singh

Module Classes:

    ModelEvaluation:
        - load model from specified path
        - makes prediction based on model, provide model report

'''

import shap
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import plot_roc_curve

# pylint: disable=too-many-arguments


class ModelEvaluationRandomForest():
    '''
    class to do the random forests model predicitons 
    and evaluate the model results
    ...

    Attributes
    ----------
        model (default=None): trained model object
        model_path (default=None): string specifying path of model pickle file
            to load model.

        Either of the attribute needs to be provided

    Methods
    ----------
    get_model_predictions:
        method returns the predictions based on predictor variables data

    get_roc_curve:
        method generates the roc curve plot and save it to specified path

    get_tree_explainer:


    feature_importance_plot:
        method create plot of important features based on random forest model
        and save the plot to specified path

    '''

    def __init__(self, model=None, model_path=None):
        self.__model = None
        if model is not None:
            self.__model = model
        elif model_path is not None:
            self.__load_model_from_path(
                model_path=model_path)

    def __load_model_from_path(self, model_path):
        self.__model = joblib.load(model_path)

    def get_model_predictions(self, data):
        '''
        method to get the predicted values

        Arguments:
            data: pandas dataframe on which prediction is required

        Output: predicted values
        '''
        return self.__model.predict(data)

    def get_roc_curve(self, predictors, target, pth_to_save=None):
        '''
        method to return the roc curve plot

        Arguments:
            save_to_pth (default=None): path to save the figure
            predictors: pandas data frame of predictor variables
            target: target variable values

        Output: plot
        '''
        plt.figure()
        plt.title("ROC curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plot_roc_curve(self.__model, predictors, target)
        if pth_to_save is not None:
            plt.savefig(pth_to_save)
        plt.close()

    def get_tree_explainer(self, data, pth_to_save=None):
        '''
        method uses Tree SHAP algorithms to explain the output of
        ensemble tree models and save plot to path specified

        Arguments:
            save_to_pth (default=None): path to save the figure

        Output: plot
        '''
        plt.figure()
        explainer = shap.TreeExplainer(self.__model)
        shap_values = explainer.shap_values(data)
        shap.summary_plot(shap_values, data,
                          plot_type="bar", show=False)
        if pth_to_save is not None:
            plt.savefig(pth_to_save)
        plt.close()

    def feature_importance_plot(self, train_data, pth_to_save=None):
        '''
        creates and stores the feature importances and save as image to path
        Arguments:
            train_data: pandas dataframe of X values
            pth_to_save: path to store the figure

        output: None
        '''
        # Calculate feature importances
        importances = self.__model.feature_importances_
        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]
        # Rearrange feature names so they match the sorted feature importances
        names = [train_data.columns[i] for i in indices]
        # Create plot
        plt.figure(figsize=(20, 5))
        # Create plot title
        plt.title("Feature Importance")
        plt.ylabel('Importance')
        # Add bars
        plt.bar(range(train_data.shape[1]), importances[indices])
        # Add feature names as x-axis labels
        plt.xticks(range(train_data.shape[1]), names, rotation=90)
        if pth_to_save is not None:
            plt.savefig(pth_to_save)
        plt.close()


class ModelEvaluationLogisticRegression():
    '''
    class to do the logistic regression 
    model predicitons and evaluate the model results
    ...

    Attributes
    ----------
        model (default=None): trained model object
        model_path (default=None): string specifying path of model pickle file
            to load model.

        Either of the attribute needs to be provided

    Methods
    ----------
    get_model_predictions:
        method returns the predictions based on predictor variables data

    get_roc_curve:
        method generates the roc curve plot and save it to specified path

    '''

    def __init__(self, model=None, model_path=None):
        self.__model = None
        if model is not None:
            self.__model = model
        elif model_path is not None:
            self.__load_model_from_path(
                model_path=model_path)

    def __load_model_from_path(self, model_path):
        self.__model = joblib.load(model_path)

    def get_model_predictions(self, data):
        '''
        method to get the predicted values

        Arguments:
            data: pandas dataframe on which prediction is required

        Output: predicted values
        '''
        return self.__model.predict(data)

    def get_roc_curve(self, predictors, target, pth_to_save=None):
        '''
        method to return the roc curve plot

        Arguments:
            save_to_pth (default=None): path to save the figure
            predictors: pandas data frame of predictor variables
            target: target variable values

        Output: plot
        '''
        plt.figure()
        plt.title("ROC curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plot_roc_curve(self.__model, predictors, target)
        if pth_to_save is not None:
            plt.savefig(pth_to_save)
        plt.close()


