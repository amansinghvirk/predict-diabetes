'''
Module: eda.py

Date: 30-Dec-2022
Author: Amandeep Singh

Module Classes:
    ExploratoryAnalysis:
        - Performs exploratory data analysis on given dataset.

'''
import matplotlib.pyplot as plt
import seaborn as sns


class ExploratoryAnalysis():
    '''
    class to perform exploratory analysis on pandas dataframe
    ...

    Attributes
    ----------
    dataframe: pandas dataframe object

    Methods
    ----------
    get_null_count:
        method returns the count of null values in each column

    get_descriptive_stats:
        method returns the descriptive stats of pandas dataframe columns
        using default describe pandas function

    get_categorical_dist:
        method to return bar plot of categorical values

    get_hist:
        method to return histogram plot of quant columns

    get_heatmap:
        method to return heatmap of column correlation values
    '''

    __df = None

    def __init__(self, dataframe):
        self.__df = dataframe

    def get_null_count(self):
        '''method to get the count of null in each column'''
        return self.__df.isnull().sum()

    def get_descriptive_stats(self):
        '''method to get the descriptive stats of dataframe'''
        return self.__df.describe()

    def get_categorical_dist(self, colname, pth_to_save=None):
        '''
        method to plot distribution of categorical column
        and optionally save png figure to path

        Arguments:
            colname: Name of the categorical column
            pth_to_save (default=None): path to save the figure

        Output: Plot
        '''
        plt.figure()
        plt.title("Distribution of " + colname.replace('_', ' '))
        plt.xlabel(colname)
        plt.ylabel('Proportion of each category')
        (self.__df[colname]
         .value_counts('normalize')
         .plot(kind='bar'))
        if pth_to_save is not None:
            plt.savefig(pth_to_save)
        plt.close()

    def get_hist(self, colname, bins=10, density=False, pth_to_save=None):
        '''
        method to plot histogram
        and optionally save png figure to path

        Arguments:
            colname: Name of the quant column
            density (default=False): set True if needs to plot density
            pth_to_save (default=None): path to save the figure

        Output: plot
        '''

        plt.figure()
        plt.title("Distribution of " + colname.replace('_', ' '))
        plt.xlabel(colname)
        if density:
            sns.histplot(
                self.__df[colname],
                bins=bins,
                stat='density',
                kde=True)
        else:
            self.__df[colname].hist(bins=bins)
        if pth_to_save is not None:
            plt.savefig(pth_to_save)
        plt.close()

    def get_heatmap(self, pth_to_save=None):
        '''
        method to plot heatmap of correlation values of columns

        Arguments:
            save_to_pth (default=None): path to save the figure

        Output: plot
        '''
        plt.figure()
        plt.title("Heatmap showing correlation between variables")
        sns.heatmap(
            self.__df.corr(),
            annot=False,
            cmap='Dark2_r'
        )
        if pth_to_save is not None:
            plt.savefig(pth_to_save)
        plt.close()
