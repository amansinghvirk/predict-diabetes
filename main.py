'''
Module: main.py
    - Module to load, preprocess, perform explorator analysis, prepares data for
        model, train and evalue model.

Date: 30-Dec-2022
Author: Amandeep Singh

Functions:
    model_report:
        function creates the classification report for train and test

    process_data:
        import and process data using Data class

    exploratory_analysis:
        performs exploratory analysis using Exploratory Analysis class

    feature_engineering:
        prepare data for model by creating features using
        FeatureEngineering class

    train_random_forest_model:
        train and save random forest model

    train_logistic_regression_model:
        train and save logistic regression model

    evaluate_random_forest_model:
        evaluates random forest model using ModelEvaluation class

    evaluate_logistic_regression_model:
        evaluates logistic regression model using ModelEvaluation class
'''


# import libraries
import os
import logging
import matplotlib.pyplot as plt


import parameters.constants as constants
from src.data.data_import import Data
from src.exploratory_analysis.eda import ExploratoryAnalysis
from src.feature_engineering.features import FeatureEngineering
from src.models.random_forest import RandomForestModel
from src.models.logistic_regression import LogisticRegressionModel
from src.model_evaluation.evaluate import (ModelEvaluationRandomForest,
                                           ModelEvaluationLogisticRegression)
from src.utils.utilities import model_report


os.environ['QT_QPA_PLATFORM'] = 'offscreen'
# use of custom style sheet for matplotlib plots
plt.style.use('./images/mplstyle/projmplstyle.mplstyle')

# pylint: disable=too-many-arguments

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(asctime)s %(process)d %(message)s',
    filename="logs/execution.log",
    filemode='w')


def process_data(data_file_pth):
    '''
    function initiate the class Data to import data,
    perform encoding and will return model, predictors and target data

    Attributes:
        data_file_pth: path of csv data file to import the data for modelling

    Output:
        model_data: imported model data from csv file as pandas dataframe
        predictors: predictors variables as pandas dataframe for model training
        target: target variable for model training
    '''

    logging.info("Import and process data!")
    data = Data(
        data_file_pth=data_file_pth,
        categorical_cols=constants.CAT_COLUMNS,
        target_col=constants.TARGET_COLUMN,
        predictor_cols=constants.PREDICTOR_COLUMNS)
    data.encode_categorical_columns()
    # save processed data for analysis and modelling
    model_data = data.get_dataframe()
    predictors = data.get_predictors()
    target = data.get_target()
    logging.info(f"Data setup with {model_data.shape[0]} rows")

    return model_data, predictors, target


def exploratory_analysis(model_data):
    '''
        function initiate the ExploratoryAnalysis class,
        perform exploratory analysis and save the results

        Attributes:
            model_data: pandas data frame on which exploratory analysis needs
                to be performed

        Output: None

        '''
    logging.info("Exploratory Analysis!")
    eda = ExploratoryAnalysis(dataframe=model_data)

    logging.info("Exploratory analysis for categorical variables!")
    for cat_var in constants.CAT_COLUMNS:
        eda.get_categorical_dist(
            colname=cat_var,
            pth_to_save='./images/eda/' + cat_var + '_dist.png')

    logging.info("Exploratory analysis for quantative variables!")
    for quant_var in constants.QUANT_COLUMNS:
        eda.get_hist(
            colname=quant_var,
            density=True,
            pth_to_save='./images/eda/' + quant_var + '_dist.png')
    eda.get_heatmap(pth_to_save='./images/eda/correlation_heatmap.png')


def feature_engineering(predictors, target):
    '''
    function perform feature engineering

    Attributes:
        predictors: predictor variables data set
        target: target variable values

    Output:
        train_data: predictors split as train dataset for trianing model
        train_target: target split as train target for training model
        test_data: predictors split as test dataset for model testing
        test_target: target split as test target for model testing
    '''

    logging.info("Feature Engineering process started!")
    feature_eng = FeatureEngineering(
        predictors=predictors,
        target=target)
    feature_eng.perform_feature_engineering(
        test_size=constants.TRAIN_TEST_SPLIT,
        random_state=constants.RANDOM_STATE)
    # save the train and test data for further evaluations
    train_data = feature_eng.get_train_data()
    test_data = feature_eng.get_test_data()
    train_target = feature_eng.get_train_target()
    test_target = feature_eng.get_test_target()
    logging.info("Data split into train and test!")
    logging.info(f"Train dateset has {len(train_data)} rows!")
    logging.info(f"Test dateset has {len(test_data)} rows!")

    return train_data, test_data, train_target, test_target


def train_random_forest_model(predictors, target):
    '''
    function train random forest model

    Attributes:
        predictors: predictor variables data set
        target: target variable values

    Output: model object (which can be used to get the model for evaluation)
        model: Random Forest Model
    '''

    logging.info("Training random forest model!")
    model = RandomForestModel(
        predictors=predictors,
        target=target,
        random_state=constants.RANDOM_STATE,
        param_grid=constants.PARAM_GRID,
        cross_fold=constants.CROSS_VALIDATION_FOLD)
    model.train_model()
    model.save_model('./models/rfc_model.pkl')

    return model.get_model()


def train_logistic_regressionmodel(predictors, target):
    '''
    function train logistic regression model

    Attributes:
        predictors: predictor variables data set
        target: target variable values

    Output: model object (which can be used to get the model for evaluation)
        model: Logistic Regression Model
    '''

    logging.info("Training logistic regression model!")
    model = LogisticRegressionModel(
        predictors=predictors,
        target=target,
        solver=constants.LOGISTIC_REGRESSION_SOLVER,
        max_iter=constants.LOGISTIC_REGRESSION_MAX_ITER)

    model.train_model()
    model.save_model('./models/logistic_model.pkl')

    return model.get_model()


def evaluate_random_forest_model(train_data, test_data, train_target,
                                 test_target, model=None, model_path=None):
    '''
    function initiate the ModelEvaluation class,
    perform model analyis for random forest model and save the results
    as images

    Attributes: (either model or model_path needs to be provide)
        train_data: training data used for model training
        test_data: test data kept for model evaluation
        train_target: target variable used for model training
        test_target: target variable in test data for model evaluation
        model (default: None): model object result from Model class
        model_path (default: None): model path to load the model

    Output: None
    '''

    logging.info("Evaluate random forest model performance!")
    if model is not None:
        rfc_model = ModelEvaluationRandomForest(model=model)
    else:
        rfc_model = ModelEvaluationRandomForest(model_path=model_path)
    # get predictions from random forest best estimator
    y_train_preds_rf = rfc_model.get_model_predictions(data=train_data)
    y_test_preds_rf = rfc_model.get_model_predictions(data=test_data)
    rfc_model.get_roc_curve(
        predictors=test_data,
        target=test_target,
        pth_to_save='./images/results/RandomForest_roc_plot.png')
    model_report(
        model_name='Random Forest',
        train_target=train_target,
        test_target=test_target,
        train_predictions=y_train_preds_rf,
        test_predictions=y_test_preds_rf,
        pth_to_save='./images/results/RandomForest_model_report.png')
    rfc_model.get_tree_explainer(
        data=test_data,
        pth_to_save='./images/results/RandomForest_TreeExplainer.png')
    rfc_model.feature_importance_plot(
        train_data=train_data,
        pth_to_save='./images/results/FeatureImportance_plot.png')


def evaluate_logistic_regression_model(
        train_data,
        test_data,
        train_target,
        test_target,
        model=None,
        model_path=None):
    '''
    function initiate the ModelEvaluation class,
    perform model analyis for random forest model and save the results
    as images

    Attributes: (either model or model_path needs to be provide)
        train_data: training data used for model training
        test_data: test data kept for model evaluation
        train_target: target variable used for model training
        test_target: target variable in test data for model evaluation
        model (default: None): model object result from Model class
        model_path (default: None): model path to load the model

    Output: None
    '''

    logging.info("Evaluate logistic regression model performance!")
    if model is not None:
        lrc_model = ModelEvaluationLogisticRegression(model=model)
    else:
        lrc_model = ModelEvaluationLogisticRegression(model_path=model_path)
    # get predictions from logistic regression model
    y_train_preds_lrc = lrc_model.get_model_predictions(data=train_data)
    y_test_preds_lrc = lrc_model.get_model_predictions(data=test_data)
    lrc_model.get_roc_curve(
        predictors=test_data,
        target=test_target,
        pth_to_save='./images/results/LogisticRegression_roc_plot.png')
    model_report(
        model_name='Logistic Regression',
        train_target=train_target,
        test_target=test_target,
        train_predictions=y_train_preds_lrc,
        test_predictions=y_test_preds_lrc,
        pth_to_save='./images/results/LogisticRegression_model_report.png')


def main(data_file_pth):
    '''
    main function to perform the sequential steps for data processing,
    exploratory analysis, model building, training and evaluation

    Arguments:
        data_file_pth: path of csv data file to import the data for modelling

    Output: None
    '''

    try:

        # Import and encode data
        model_data, predictors, target = process_data(data_file_pth)

        # Exploratory Analysis
        exploratory_analysis(model_data)

        # Perform feature engineering
        (train_data, test_data,
         train_target, test_target) = feature_engineering(predictors, target)

        # build and train random forest model
        rfc_model = train_random_forest_model(train_data, train_target)

        # build and train logistic regression model
        lrc_model = train_logistic_regressionmodel(train_data, train_target)

        # Evaluate random forest model
        evaluate_random_forest_model(train_data=train_data,
                                     test_data=test_data,
                                     train_target=train_target,
                                     test_target=test_target,
                                     model=rfc_model)

        # Evaluate logistic regression model
        evaluate_logistic_regression_model(train_data=train_data,
                                           test_data=test_data,
                                           train_target=train_target,
                                           test_target=test_target,
                                           model=lrc_model)

    except Exception as e:
        logging.exception("Execution failed due to %s", e)


if __name__ == "__main__":
    main(data_file_pth='./data/diabetes.csv')