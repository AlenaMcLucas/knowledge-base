######################
### Student Grades ###
######################

# imports
import pandas as pd
import numpy as np
from prettytable import PrettyTable
from sklearn.cross_validation import train_test_split

# libraries I wrote lawl
import univariate as uni
import linear_reg as linreg
import data_clean as clean

dataset = pd.read_csv('student-mat.csv')

# summary table
t = PrettyTable()
t.add_column('Variable', dataset.columns)
t.add_column('Data Type', dataset.dtypes)
t.add_column('In/Dependent', ['i','i','i','i','i','i','i','i','i','i','i',
'i','i','i','i','i','i','i','i','i','i','i','i','i','i','i','i','i','i','i',
'd','d','d'])
t.add_column('Category', ['categorical','categorical','continuous','categorical',
'categorical','categorical','categorical','categorical','categorical',
'categorical','categorical','categorical','categorical','categorical',
'continuous','categorical','categorical','categorical','categorical',
'categorical','categorical','categorical','categorical','categorical',
'categorical','categorical','categorical','categorical','categorical',
'continuous','continuous','continuous','continuous'])
print t

## Data Exploration ##
######################

total_count = len(dataset)

# uni.categorical(dataset.G3, total_count)

## Feature Engineering ##
#########################

dataset_clean = dataset.copy(deep=True)   # new dataset to clean

# school - GP = 0, MS = 1
col_name = 'school'
dataset_clean[col_name] = clean.categorical_clean(dataset_clean.ix[:,col_name], True, False)

# sex - F = 0, M = 1
col_name = 'sex'
dataset_clean[col_name] = clean.categorical_clean(dataset_clean.ix[:,col_name], True, False)

# address - R = 0, U = 1
col_name = 'address'
dataset_clean[col_name] = clean.categorical_clean(dataset_clean.ix[:,col_name], True, False)

# famsize - GT3 = 0, LE3 = 1
col_name = 'famsize'
dataset_clean[col_name] = clean.categorical_clean(dataset_clean.ix[:,col_name], True, False)

# Pstatus - A = 0, T = 1
col_name = 'Pstatus'
dataset_clean[col_name] = clean.categorical_clean(dataset_clean.ix[:,col_name], True, False)

# Mjob - at_home = 0, health = 1, other = 2, services = 3, teacher = 4
col_name = 'Mjob'
variable_clean = clean.categorical_clean(dataset_clean.ix[:,col_name], True, True)
[dataset_clean.insert(dataset_clean.columns.get_loc(col_name), variable_clean.columns[i], variable_clean.ix[:,i]) for i in np.arange(len(variable_clean.columns))]
del dataset_clean[col_name]

# Fjob - at_home = 0, health = 1, other = 2, services = 3, teacher = 4
col_name = 'Fjob'
variable_clean = clean.categorical_clean(dataset_clean.ix[:,col_name], True, True)
[dataset_clean.insert(dataset_clean.columns.get_loc(col_name), variable_clean.columns[i], variable_clean.ix[:,i]) for i in np.arange(len(variable_clean.columns))]
del dataset_clean[col_name]

# reason - course = 0, home = 1, other = 2, repuation = 3
col_name = 'reason'
variable_clean = clean.categorical_clean(dataset_clean.ix[:,col_name], True, True)
[dataset_clean.insert(dataset_clean.columns.get_loc(col_name), variable_clean.columns[i], variable_clean.ix[:,i]) for i in np.arange(len(variable_clean.columns))]
del dataset_clean[col_name]

# guardian - father = 0, mother = 1, other = 2
col_name = 'guardian'
variable_clean = clean.categorical_clean(dataset_clean.ix[:,col_name], True, True)
[dataset_clean.insert(dataset_clean.columns.get_loc(col_name), variable_clean.columns[i], variable_clean.ix[:,i]) for i in np.arange(len(variable_clean.columns))]
del dataset_clean[col_name]

# schoolsup - no = 0, yes = 1
col_name = 'schoolsup'
dataset_clean[col_name] = clean.categorical_clean(dataset_clean.ix[:,col_name], True, False)

# famsup - no = 0, yes = 1
col_name = 'famsup'
dataset_clean[col_name] = clean.categorical_clean(dataset_clean.ix[:,col_name], True, False)

# paid - no = 0, yes = 1
col_name = 'paid'
dataset_clean[col_name] = clean.categorical_clean(dataset_clean.ix[:,col_name], True, False)

# activities - no = 0, yes = 1
col_name = 'activities'
dataset_clean[col_name] = clean.categorical_clean(dataset_clean.ix[:,col_name], True, False)

# nursery - no = 0, yes = 1
col_name = 'nursery'
dataset_clean[col_name] = clean.categorical_clean(dataset_clean.ix[:,col_name], True, False)

# higher - no = 0, yes = 1
col_name = 'higher'
dataset_clean[col_name] = clean.categorical_clean(dataset_clean.ix[:,col_name], True, False)

# internet - no = 0, yes = 1
col_name = 'internet'
dataset_clean[col_name] = clean.categorical_clean(dataset_clean.ix[:,col_name], True, False)

# romantic - no = 0, yes = 1
col_name = 'romantic'
dataset_clean[col_name] = clean.categorical_clean(dataset_clean.ix[:,col_name], True, False)

## Train Test Split ##
######################

X = dataset_clean.ix[:,:'absences']
y = pd.DataFrame(dataset_clean.ix[:,'G3'])

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.2,
                                                    random_state = 0)

## Machine Learning ##
######################

(beta, costs) = linreg.gradient_descent(X, y, 0.0001, 1000)
print beta

# to calculate cost at converged number either do this:
(X_t, y_t, theta_t, m_t) = linreg.reg_prep(dataset_clean.ix[:,:'absences'], y)
print linreg.cost_function(X_t, y_t, beta, m_t)

# OR
costs[(len(costs) - 1)]
