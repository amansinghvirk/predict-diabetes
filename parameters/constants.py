# constants.py

"""This module defines project-level constants"""

# Categorical columns in dataset
CAT_COLUMNS = [               
]
# Quantative columns in dataset
QUANT_COLUMNS = [
    'Pregnancies', 'Glucose', 'BloodPressure', 
    'SkinThickness', 'Insulin',
    'BMI', 'DiabetesPedigreeFunction', 'Age'
]
# Target column to be used for prediction
TARGET_COLUMN = 'Outcome'
# Variables to be used as predcitors for target variable
PREDICTOR_COLUMNS = [
    'Pregnancies', 'Glucose', 'BloodPressure', 
    'SkinThickness', 'Insulin',
    'BMI', 'DiabetesPedigreeFunction', 'Age'
]
# Random intialization seed value
RANDOM_STATE = 42
# Ratio of train and test split
TRAIN_TEST_SPLIT = 0.3
# Parameters for Logistic Regression
LOGISTIC_REGRESSION_SOLVER = 'lbfgs'
LOGISTIC_REGRESSION_MAX_ITER = 3000
# Parameters for Random Forest
CROSS_VALIDATION_FOLD = 5
PARAM_GRID = { 
    'n_estimators': [200, 500, 1000],
    'max_features': ['auto', 'sqrt'],
    'max_depth' : [4, 5, 10, 100],
    'criterion' :['gini', 'entropy']
}