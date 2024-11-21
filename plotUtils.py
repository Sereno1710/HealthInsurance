#######################################################
# A utils file that contains methods to plot the data #
#######################################################

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Markdown
from IPython.core.display import HTML

class Utils:
    
    def __init__(self, df):
        HTML("""
            <style>
            .output_png {
                display: table-cell;
                text-align: center;
                vertical-align: middle;
            }
            </style>
            """)
        self.df = df
    
    def print_statistics(self, feature):
        table = '| Statistics | Value |\n'
        table += '| --- | --- |\n'
        table += '| Mean | {:.2f} |\n'.format(self.df[feature].mean())
        table += '| Standard Deviation | {:.2f} |\n'.format(self.df[feature].std())
        table += '| Minimum | {:.2f} |\n'.format(self.df[feature].min())
        table += '| 25th percentile | {:.2f} |\n'.format(self.df[feature].quantile(0.25))
        table += '| Median | {:.2f} |\n'.format(self.df[feature].median())
        table += '| 75th percentile | {:.2f} |\n'.format(self.df[feature].quantile(0.75))
        table += '| Maximum | {:.2f} |'.format(self.df[feature].max())
        return Markdown(table)

    def plot_statistics(self, feature, with_target_value=False):
        plt.figure(figsize=(8, 4))
        
        if with_target_value:
            sns.boxplot(x=self.df[feature], palette='viridis', hue=self.df['health_ins'])
        else:
            sns.boxplot(x=self.df[feature])
            
        plt.title(f'{feature} boxplot')
        plt.show()
        
    def plot_numeric_distribution(self, feature, plot_outliers = True):
        # Calculate outliers using the IQR method
        Q1 = self.df[feature].quantile(0.25)
        Q3 = self.df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Split the data into inliers and outliers
        inliers = self.df[(self.df[feature] >= lower_bound) & (self.df[feature] <= upper_bound)][feature]
        outliers = self.df[(self.df[feature] < lower_bound) | (self.df[feature] > upper_bound)][feature]

        # Plot inliers (blue) and outliers (light red)
        sns.histplot(inliers, kde=True, bins=20, color='blue', label='Inliers') 
        if plot_outliers:
            sns.histplot(outliers, kde=True, bins=20, color='lightcoral', label='Outliers')
            plt.axvline(lower_bound, color='red', linestyle='--', label=f'Lower Bound: {lower_bound:.2f}')
            plt.axvline(upper_bound, color='red', linestyle='--', label=f'Upper Bound: {upper_bound:.2f}')

        plt.legend()
        plt.title(f'{feature} Distribution')
        plt.show()
        
    def plot_categorical_distribution(self, feature):
        sns.countplot(x=feature, data=self.df, palette='viridis')
        plt.title(f'{feature} Distribution')
        plt.show()

    def plot_distribution(self, feature, plot_outliers = True):
        if self.df[feature].dtype in ['int64', 'float64']:
            self.plot_numeric_distribution(feature, plot_outliers)
        else:
            self.plot_categorical_distribution(feature)