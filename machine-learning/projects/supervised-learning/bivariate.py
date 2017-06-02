# import libraries
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.stats import chi2_contingency
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Pearson Correlation Test Table
def pearson_table(df):
    p_table = pd.DataFrame(index = df.columns, columns = df.columns)
    
    for index, row in p_table.iterrows():
        for i in np.arange(len(df.columns)):
            temp_tuple = pearsonr(df[index], df.iloc[:,i])
            p_table.ix[index, i] = (round(temp_tuple[0], 4), round(temp_tuple[1], 4))
    
    print p_table

# Chi-Square Test Table
def chi2_table(df):
    p_table = pd.DataFrame(index = df.columns, columns = df.columns)
    
    for index, row in p_table.iterrows():
        for i in np.arange(len(df.columns)):
            try:
                chi2, p, dof, expected = chi2_contingency(pd.concat([df[index], df.iloc[:,i]], axis = 1))
                p_table.ix[index, i] = (round(chi2, 4), round(p, 4))
            except ValueError:
                p_table.ix[index, i] = np.nan
    
    print p_table

# takes continuous variable, then categorical variable
def anova(series_continuous, series_categorical):
    f_value, p_value = f_oneway(series_continuous, series_categorical)
    print (f_value, p_value)

    print pairwise_tukeyhsd(series_continuous, series_categorical, alpha = 0.05)
