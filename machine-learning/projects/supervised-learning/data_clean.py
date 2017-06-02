import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def categorical_clean(col, numeric, dummies):
    
    col_clean = pd.DataFrame(col)
    col_name = col_clean.columns[0]
    
    if numeric == True:
        labelencoder = LabelEncoder()
        col_clean.ix[:,col_name] = labelencoder.fit_transform(col_clean.ix[:,col_name])
        
    if dummies == True:
        col_clean = pd.get_dummies(col_clean.ix[:,col_name])
        col_clean = col_clean.iloc[:,:-1]
        
        col_name_new = []
        for i in np.arange(len(col_clean.columns)):
             col_name_new.insert(i, str(col_name) + '_' + str(i))
        col_clean.columns = col_name_new
    
    return col_clean

# mean normalization using numpy:
# dataset_clean_temp.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
def mean_norm(col):
    col_clean = pd.DataFrame(col).apply(lambda x: (x - x.mean()) / (x.max() - x.min()))
    return col_clean
