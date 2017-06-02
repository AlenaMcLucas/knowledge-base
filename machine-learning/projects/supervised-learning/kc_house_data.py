#### King County House Sale Prices ####

'''
This dataset contains house sale prices for King County, which includes
Seattle. It includes homes sold between May 2014 and May 2015.

This dataset is hosted on Kaggle at
https://www.kaggle.com/harlfoxem/housesalesprediction released Under CC0:
Public Domain License.

The goal of this project is to apply basic machine learning principles to
predict housing sales prices in this county.
'''

# import libraries
import numpy as np
import pandas as pd
from prettytable import PrettyTable

# libraries I wrote lawl
import univariate as uni
import linear_reg as linreg

# import dataset
dataset = pd.read_csv('kc_house_data.csv')

### Data Exploration ###

'''
The dependent variable is price.

The independent variables are the rest.
'''

# summary table
t = PrettyTable()
t.add_column('Variable', dataset.columns)
t.add_column('Data Type', dataset.dtypes)
t.add_column('In/Dependent', ['n/a','d','i','d','d','d','d','d','d','d','d',
'd','d','d','d','d','d','d','d','d','d'])
t.add_column('Category', ['id','date','continuous','categorical',
'categorical','continuous','continuous','categorical','categorical',
'categorical','categorical','categorical','continuous','continuous',
'categorical','categorical','categorical','categorical','categorical',
'continuous','continuous'])
print t

## Univariate Analysis ##

# total_count
total_count = len(dataset.index)

# all categorical variables
uni.categorical(dataset.bedrooms, total_count)
uni.categorical(dataset.bathrooms, total_count)
uni.categorical(dataset.floors, total_count)
uni.categorical(dataset.waterfront, total_count)
uni.categorical(dataset.view, total_count)
uni.categorical(dataset.condition, total_count)
uni.categorical(dataset.grade, total_count)
uni.categorical(dataset.yr_built, total_count)
uni.categorical(dataset.yr_renovated, total_count)
uni.categorical(dataset.zipcode, total_count)
uni.categorical_nrm(dataset.lat, total_count)
uni.categorical_nrm(dataset.long, total_count)

# all continuous variables

continuous_var = pd.concat([dataset['price'], dataset['sqft_living'],
                            dataset['sqft_lot'], dataset['sqft_above'],
                            dataset['sqft_basement'], dataset['sqft_living15'],
                            dataset['sqft_lot15']], axis = 1)

uni.continuous_all(continuous_var, total_count)

uni.continuous(dataset.price, 10, total_count)
uni.continuous(dataset.sqft_living, 10, total_count)
uni.continuous(dataset.sqft_lot, 10, total_count)
uni.continuous(dataset.sqft_above, 10, total_count)
uni.continuous(dataset.sqft_basement, 10, total_count)
uni.continuous(dataset.sqft_living15, 10, total_count)
uni.continuous(dataset.sqft_lot15, 10, total_count)

# delete id, date, sqft_living15, sqft_lot15
del dataset['id']
del dataset['date']
del dataset['sqft_living15']
del dataset['sqft_lot15']

# create X and y
X = pd.DataFrame(dataset.iloc[:,1:])
#X = pd.DataFrame(dataset.iloc[:,1])
y = pd.DataFrame(dataset.iloc[:,0])

(beta, costs) = linreg.linear_regression(X, y)
print beta



# notes
# bedrooms (numerical -> discrete (which is so far same as categorical -> nominal ^ ordinal))
# are measures of central tendency and dispersion important?
# print frequency table (count and/or count%)
# print bar graph (count and/or count%)


# junkyard

# check whole dataset for missing values and return new dataframe of them
# dataset_missing = dataset.id[pd.isnull(dataset.id).any(axis=1)]

# check one variable for missing values and return just those values
# dataset.id[pd.isnull(dataset.id)]
