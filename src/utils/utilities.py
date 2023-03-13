'''
Module: utilities.py
    - Module common utilities functions.

Date: 23-Jan-2022
Author: Amandeep Singh

Functions:
    logger: to setup loger for logging the model detaisl
    
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

def model_report(
        model_name,
        train_target,
        test_target,
        train_predictions,
        test_predictions,
        pth_to_save=None):
    '''
    produces classification report for training and testing results
    and stores report as image in images folder

    Arguments:
    model_name: string value specifying the name of the model to be used
        in report
    train_target: actual target values from train dataset
    test_target: actual target values from test dataset
    train_predictions: predicted values from model using train dataset
    test_predictions: predicted values from model using test dataset
    save_to_pth (default=None): path to save the figure

    Output: plot
    '''

    plt.figure()
    plt.text(0.01, 1.25,
             str(model_name + ' Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05,
             str(classification_report(
                 train_target,
                 train_predictions)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6,
             str(model_name + ' Test'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7,
             str(classification_report(
                 test_target,
                 test_predictions)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    if pth_to_save is not None:
        plt.savefig(pth_to_save)
    plt.close()