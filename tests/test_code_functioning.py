'''
Module: test_code_functioning

Date: 30-Dec-2022
Author: Amandeep Singh
'''


import logging
import pytest

import parameters.constants as constants
from src.data.data_import import Data
from src.exploratory_analysis.eda import ExploratoryAnalysis
from src.feature_engineering.features import FeatureEngineering
from src.models.random_forest import RandomForestModel
from src.models.logistic_regression import LogisticRegressionModel
from src.model_evaluation.evaluate import (ModelEvaluationRandomForest,
                                           ModelEvaluationLogisticRegression)
from src.utils.utilities import model_report


# pylint: disable=no-value-for-parameter

logging.basicConfig(
    filename='logs/tests.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


@pytest.fixture(name='data_path')
def fixture_data_path():
    '''defines path to data for model training'''
    return './data/diabetes.csv'


def test_import_data(data_path):
    '''to test the wokring of data import in Data class'''
    try:
        _ = Data(
            data_file_pth=data_path,
            categorical_cols=constants.CAT_COLUMNS,
            target_col=constants.TARGET_COLUMN,
            predictor_cols=constants.PREDICTOR_COLUMNS)
        logging.info("Data: Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Data: Testing import_data: The file wasn't found")
        raise err


@pytest.fixture(name='process_data')
def fixture_process_data(data_path):
    '''fixture to initiate the Data class and import data'''
    return Data(
        data_file_pth=data_path,
        categorical_cols=constants.CAT_COLUMNS,
        target_col=constants.TARGET_COLUMN,
        predictor_cols=constants.PREDICTOR_COLUMNS)


def test_rows(process_data):
    '''test if there are data rows in dataset'''
    try:
        assert process_data.get_dataframe_shape()[0] > 0
        logging.info("Data: Testing rows: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Data: Testing rows: The file doesn't appear to have rows")
        raise err


def test_columns(process_data):
    '''test if dataset contains columns'''
    try:
        assert process_data.get_dataframe_shape()[1] > 0
        logging.info("Data: Testing columns: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Data: Testing columsns: The file doesn't appear to have columns")
        raise err


@pytest.fixture(name='encode_data')
def fixture_encode_data(process_data):
    '''text the encoding of categorical columns'''
    try:
        process_data.encode_categorical_columns()
        logging.info('Data: encode columns: SUCCESS')
    except Exception as err:
        logging.info(
            'Data: encode columns: Failed encoding categorical columns!')
        raise err

    return process_data


def test_model_data(encode_data):
    '''test if the data prepared for model has rows and columns'''
    try:
        assert encode_data.get_dataframe_shape()[0] > 0
        assert encode_data.get_dataframe_shape()[1] > 0
        logging.info("Data: model data columns and rows: SUCCESS")
    except AssertionError as err:
        logging.error("Data: model data columns and rows: no rows and columns")
        raise err


def test_target_column(encode_data):
    '''validate that target column is in data for training model'''
    try:
        assert (constants.TARGET_COLUMN
                in encode_data.get_dataframe().columns)
        logging.info('Data: Create Target column: SUCCESS')
    except AssertionError as err:
        logging.info(
            'Data: Create Target column: Target column creation failed!')
        raise err


def test_predictors_target(encode_data):
    '''validate if rows in predictor and target data frame are same'''
    try:
        assert (len(encode_data.get_predictors())
                == len(encode_data.get_target()))
        logging.info("Data: same length predictors and target: SUCCESS")
    except AssertionError as err:
        logging.error("Data: same length predictors and target: FAILED")
        raise err


def test_predictor_variables(encode_data):
    '''validate that dataset for training model contains all the
    predictor variables specified in constants file
    '''
    try:
        assert (len(list(set(constants.PREDICTOR_COLUMNS).intersection(
            encode_data.get_predictors().columns)))
            == len(constants.PREDICTOR_COLUMNS))
        logging.info("Data: test predictors: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Data: test predictors: Not all predictors in train dataset!")
        raise err


@pytest.fixture(name='exploratory_data')
def fixture_exploratory_data(encode_data):
    '''initialize the Exploratory Analysis class'''
    return ExploratoryAnalysis(dataframe=encode_data.get_dataframe())


def test_categorical_dist(exploratory_data):
    '''test functioning of categorical distribution plot function'''
    try:
        for cat_var in constants.CAT_COLUMNS:
            exploratory_data.get_categorical_dist(
                colname=cat_var,
                pth_to_save='./images/eda/' + cat_var + '_dist.png')
        logging.info("EDA: Categorical Dist Plot: SUCCESS")
    except KeyError as err:
        logging.error(
            "EDA: Categorical Dist Plot: Categorical column not found!")
        raise err
    except Exception as err:
        logging.error("EDA: Categorical Dist Plot: FAILED")
        raise err


def test_hist(exploratory_data):
    '''test functioning of histogram plot function'''
    try:
        for quant_var in constants.QUANT_COLUMNS:
            exploratory_data.get_hist(
                colname=quant_var,
                density=True,
                pth_to_save='./images/eda/' + quant_var + '_dist.png')
        logging.info("EDA: Histogram Plot: SUCCESS")
    except KeyError as err:
        logging.error("EDA: Histogram Plot: column not found!")
        raise err
    except Exception as err:
        logging.error("EDA: Histogram Plot: FAILED")
        raise err


def test_heatmap(exploratory_data):
    '''test functioning of heat map plot function'''
    try:
        exploratory_data.get_heatmap(
            pth_to_save='./images/eda/correlation_heatmap.png')
    except Exception as err:
        logging.error("EDA: Heatmap Plot: FAILED")
        raise err


@pytest.fixture(name='feature_eng')
def fixture_feature_eng(encode_data):
    '''initiate the Feature Engineering class'''
    return FeatureEngineering(
        predictors=encode_data.get_predictors(),
        target=encode_data.get_target())


@pytest.fixture(name='features')
def fixture_features(feature_eng):
    ''' test the function of feature engineering creation function'''
    try:
        feature_eng.perform_feature_engineering(
            test_size=constants.TRAIN_TEST_SPLIT,
            random_state=constants.RANDOM_STATE)
        logging.info('Model: Feature Engineering: SUCCESS')
    except Exception as err:
        logging.error('Model: Feature Engineering: FAILED')
        raise err

    return feature_eng


def test_train_shape(features):
    ''' Validate the rows and columns in train dataset'''
    try:
        assert len(features.get_train_data()) > 0
        assert len(features.get_train_target()) > 0
        logging.info('Model: Compare length training and target data: SUCCESS')
    except AssertionError as err:
        logging.error('Model: Compare length training and target data: FAILED')
        raise err


def test_train_size(features):
    '''validate the lenght of train predictors and target is matching'''
    try:
        assert len(
            features.get_train_data()) == len(
            features.get_train_target())
        logging.info('Model: Match length training and target data: SUCCESS')
    except AssertionError as err:
        logging.error('Model: Match length training and target data: FAILED')
        raise err


def test_test_shape(features):
    ''' Validate the rows and columns in test dataset'''
    try:
        assert len(features.get_test_data()) > 0
        assert len(features.get_test_target()) > 0
        logging.info('Model: Compare length test and target data: SUCCESS')
    except AssertionError as err:
        logging.error('Model: Compare length test and target data: FAILED')
        raise err


def test_test_size(features):
    '''validate the lenght of train predictors and target is matching'''
    try:
        assert len(features.get_test_data()) == len(features.get_test_target())
        logging.info('Model: Match length test and target data: SUCCESS')
    except AssertionError as err:
        logging.error('Model: Match length test and target data: FAILED')
        raise err


@pytest.fixture(name='random_forest_model')
def fixture_random_forest_model(features):
    '''initiate and validate the Random Forest Model'''
    try:
        model = RandomForestModel(
            predictors=features.get_train_data(),
            target=features.get_train_target(),
            random_state=constants.RANDOM_STATE,
            param_grid=constants.PARAM_GRID,
            cross_fold=constants.CROSS_VALIDATION_FOLD)
        logging.info('RF Model: Setup random forest model: SUCCESS')
    except Exception as err:
        logging.error('RF Model: Setup random forest model: FAILED')
        raise err

    return model


def test_random_forest_model(random_forest_model):
    '''test train and save functionality of random forest model'''
    try:
        random_forest_model.train_model()
        logging.info('RF Model: Train random forest model: SUCCESS')
    except Exception as err:
        logging.error('RF Model: Train random forest model: FAILED')
        raise err

    try:
        random_forest_model.save_model('./models/rfc_model.pkl')
        logging.info('RF Model: save random forest model: SUCCESS')
    except Exception as err:
        logging.error('RF Model: save random forest model: FAILED')
        raise err


@pytest.fixture(name='logistic_regression_model')
def fixture_logistic_regression_model(features):
    '''initiate and validate the logistic regression model'''
    try:
        model = LogisticRegressionModel(
            predictors=features.get_train_data(),
            target=features.get_train_target(),
            solver=constants.LOGISTIC_REGRESSION_SOLVER,
            max_iter=constants.LOGISTIC_REGRESSION_MAX_ITER)
        logging.info('LR Model: Setup random forest model: SUCCESS')
    except Exception as err:
        logging.error('LR Model: Setup random forest model: FAILED')
        raise err

    return model


def test_logistic_regression_model(logistic_regression_model):
    '''test train and save functionality of logistic regression model'''
    try:
        logistic_regression_model.train_model()
        logging.info('LR Model: Train random forest model: SUCCESS')
    except Exception as err:
        logging.error('LR Model: Train random forest model: FAILED')
        raise err

    try:
        logistic_regression_model.save_model('./models/logistic_model.pkl')
        logging.info('LR Model: save random forest model: SUCCESS')
    except Exception as err:
        logging.error('LR Model: save random forest model: FAILED')
        raise err


@pytest.fixture(name='random_forest_evaluation')
def fixture_random_forest_evaluation():
    '''initialize the ModelEvaluation class for random forest model'''
    try:
        rfc_model = ModelEvaluationRandomForest(
            model_path='./models/rfc_model.pkl')
        logging.info('Model Evaluation RF: load the model: SUCCESS')
    except Exception as err:
        logging.error('Model Evaluation RF: load the model: FAILED')
        raise err

    return rfc_model


def test_random_forest_predictions(
        random_forest_evaluation, features):
    ''' test the prediction funciton for random forest'''
    try:
        _ = random_forest_evaluation.get_model_predictions(
            data=features.get_train_data())
        logging.info(
            'Model Evaluation RF: get predicitons on train data: SUCCESS')
    except Exception as err:
        logging.error(
            'Model Evaluation RF: get predicitons on train data: FAILED')
        raise err


def test_random_forest_roc(
        random_forest_evaluation, features):
    '''test the roc curve function for random forest'''
    try:
        random_forest_evaluation.get_roc_curve(
            predictors=features.get_test_data(),
            target=features.get_test_target(),
            pth_to_save='./images/results/RandomForest_roc_plot.png')
        logging.info('Model Evaluation RF: get roc curve: SUCCESS')
    except Exception as err:
        logging.error('Model Evaluation RF: get roc curve: FAILED')
        raise err


@pytest.fixture(name='logistic_regression_evaluation')
def fixture_logistic_regression_evaluation():
    '''initialize the ModelEvaluation class for logistic regression'''

    try:
        lrc_model = ModelEvaluationLogisticRegression(
            model_path='./models/logistic_model.pkl')
        logging.info('Model Evaluation LR: load the model: SUCCESS')
    except Exception as err:
        logging.error('Model Evaluation LR: load the model: FAILED')
        raise err

    return lrc_model


def test_logistic_regression_predictions(
        logistic_regression_evaluation, features):
    ''' test the prediction functionality of logistic regression'''
    try:
        _ = (
            logistic_regression_evaluation.get_model_predictions(
                data=features.get_train_data()))
        logging.info(
            'Model Evaluation LR: get predicitons on train data: SUCCESS')
    except Exception as err:
        logging.error(
            'Model Evaluation LR: get predicitons on train data: FAILED')
        raise err


def test_logistic_regression_roc(
        logistic_regression_evaluation, features):
    '''test the roc curve function for logistic regression'''
    try:
        logistic_regression_evaluation.get_roc_curve(
            predictors=features.get_test_data(),
            target=features.get_test_target(),
            pth_to_save='./images/results/LogisticRegression_roc_plot.png')
        logging.info('Model Evaluation LR: get roc curve: SUCCESS')
    except Exception as err:
        logging.error('Model Evaluation LR: get roc curve: FAILED')
        raise err


def test_model_report(random_forest_evaluation, features):
    '''test the model report function'''
    try:
        y_train_preds = random_forest_evaluation.get_model_predictions(
            data=features.get_train_data())
        y_test_preds = random_forest_evaluation.get_model_predictions(
            data=features.get_test_data())
        model_report(
            model_name='Random Forest',
            train_target=features.get_train_target(),
            test_target=features.get_test_target(),
            train_predictions=y_train_preds,
            test_predictions=y_test_preds,
            pth_to_save='./images/results/RandomForest_model_report.png')
        logging.info('Model Report: generate model report: SUCCESS')
    except Exception as err:
        logging.error('Model Report: generate model report: FAILED')
        raise err


if __name__ == "__main__":
    # Arrange
    test_import_data()
    test_rows()
    test_columns()
    test_model_data()
    test_target_column()
    test_predictors_target()
    test_predictor_variables()
    test_categorical_dist()
    test_hist()
    test_heatmap()
    test_train_shape()
    test_train_size()
    test_test_shape()
    test_test_size()
    test_random_forest_model()
    test_logistic_regression_model()
    test_random_forest_predictions()
    test_random_forest_roc()
    test_logistic_regression_predictions()
    test_logistic_regression_roc()
    test_model_report()
