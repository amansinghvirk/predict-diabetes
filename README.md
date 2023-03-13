# Introduction

Objective of the project is to predict the diabetes based on several health indicatiors.


# Installation

Project uses the python version 3.8. Before executing the code it is require to have the required packages installed. Project repository includes requirements.txt file which specifies the required packages. Below are the steps to install the required packages.

<div style="color:#021829;background:#e9f1f7">


<h4>for python 3.8:</h4>
> pip install -r requirements_py.txt
</div>

# Execution

<h5>To execute the code run the following command:</h5>

<div style="color:#021829;background:#e9f1f7">
> python main.py
</div>

<p>
Above command will execute the code from main which will prepare the data, perform exploratory analysis, train the model and save the model results
</p>

<h5>To execute the test script run the following command:</h5>

<div style="color:#021829;background:#e9f1f7">
> python -m pytest -p no:logging tests/
</div>



<div style="color:#021829;background:#e9f1f7">
<h5>if need to print the status of each test on screen:</h5>
> python -m pytest -p no:logging tests/ -v
</div>

Above command will perform the series of tests and saves the log in logs folder

### Directory structure
- **data**: *data for modeling will be used from this directory*
- **images**
    - **eda**: *results of exploratory analysis will be saved in this directory*
    - **results**: *results of model analysis will be saved in this directory*
- **models**: *trained models will be saved in this directory*

### Project parameters

The project parameters are saved in ***constants.py*** file.


<details>
<summary>Code Structure</summary>

Project code is organized as classes which can work independently of each other. Below is the descriptin of each class

| Class                    | Description                                                           |
|-------------------------:|-----------------------------------------------------------------------|
| Data                     | Import the data from csv file and preforms the preprocessing of data  |
| ExploratoryAnalysis      | Performs the exploratory analysis and save the resutls                |
| FeatureEngineering       | Performs the feature engineering e.g. train, test split               |
| RandomForestModel        | Setup, train and saves the random forest model.                       |
| LogisticRegressionModel  | Setup, train and saves the random forest model.                       |
| ModelEvaluation          | Evaluate model resutls and save the analysis                          |
  
</details>