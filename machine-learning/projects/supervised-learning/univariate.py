# import libraries
from prettytable import PrettyTable
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd

# print univariate analysis for a numerical continuous variable
# pass it a vectorized variable
def continuous(variable, hist_bins, total_count):
    continuous_c(variable)
    continuous_d(variable)
    continuous_h(variable, hist_bins)
    continuous_b(variable)
    continuous_m(variable, total_count)

# measures of central tendency
def continuous_c(variable):
    t = PrettyTable([variable.name,''])
    t.add_row(['mean', np.mean(variable)])
    t.add_row(['median', np.median(variable)])
    t.add_row(['mode', sp.stats.mode(variable)[0]])
    t.add_row(['min', min(variable)])
    t.add_row(['max', max(variable)])
    print t

# measures of dispersion
def continuous_d(variable):
    t = PrettyTable([variable.name,''])
    t.add_row(['range', max(variable)-min(variable)])
    t.add_row(['25%', np.percentile(variable, 25)])
    t.add_row(['50%', np.percentile(variable, 50)])
    t.add_row(['75%', np.percentile(variable, 75)])
    t.add_row(['variance', np.var(variable, ddof = 1)])
    t.add_row(['standard deviation', np.std(variable, ddof = 1)])
    t.add_row(['coefficient of deviation', sp.stats.variation(variable)])
    t.add_row(['skewness', sp.stats.skew(variable)])
    t.add_row(['kurtosis', sp.stats.kurtosis(variable)])
    print t

# histogram
def continuous_h(variable, hist_bins):
    plt.hist(variable, bins = hist_bins)
    plt.xlabel(variable.name.title())
    plt.title(variable.name.title() + ' Histogram')
    plt.grid(True)
    plt.show()

# boxplot
def continuous_b(variable):
    plt.boxplot(variable)
    plt.ylabel(variable.name.title())
    plt.title(variable.name.title() + ' Boxplot')
    plt.grid(True)
    plt.show()

# missing values
def continuous_m(variable, total_count):
    print 'Missing values: '+ str(total_count - variable.count())
    print variable[pd.isnull(variable)]

# get all of the above (no visualizations) for all continuous variables at once
def continuous_all(variable, total_count):
    blanks = ['' for x in np.arange(len(variable.columns))]
    t = PrettyTable([''] + list(variable.columns.values))
    
    t.add_row(['Central Tendency'] + blanks)   # central tendency
    
    t.add_row(['mean'] + [np.mean(x) for x in [variable.iloc[:,i] for i in np.arange(len(variable.columns))]])
    t.add_row(['median'] + [np.median(x) for x in [variable.iloc[:,i] for i in np.arange(len(variable.columns))]])
    t.add_row(['mode'] + [sp.stats.mode(x)[0] for x in [variable.iloc[:,i] for i in np.arange(len(variable.columns))]])
    this_min = [min(x) for x in [variable.iloc[:,i] for i in np.arange(len(variable.columns))]]
    this_max = [max(x) for x in [variable.iloc[:,i] for i in np.arange(len(variable.columns))]]
    t.add_row(['min'] + this_min)
    t.add_row(['max'] + this_max)
    
    t.add_row([''] + blanks)
    t.add_row(['Dispersion'] + blanks)   # dispersion
    
    this_range = []
    this_count = 0
    for i in this_max:
        this_range.append(this_max[this_count] - this_min[this_count])
        this_count += 1
            
    t.add_row(['range'] + this_range)
    t.add_row(['25%'] + [np.percentile(x, 25) for x in [variable.iloc[:,i] for i in np.arange(len(variable.columns))]])
    t.add_row(['50%'] + [np.percentile(x, 50) for x in [variable.iloc[:,i] for i in np.arange(len(variable.columns))]])
    t.add_row(['75%'] + [np.percentile(x, 75) for x in [variable.iloc[:,i] for i in np.arange(len(variable.columns))]])
    t.add_row(['variance'] + [np.var(x, ddof = 1) for x in [variable.iloc[:,i] for i in np.arange(len(variable.columns))]])
    t.add_row(['standard deviation'] + [np.std(x, ddof = 1) for x in [variable.iloc[:,i] for i in np.arange(len(variable.columns))]])
    t.add_row(['coefficient of deviation'] + [sp.stats.variation(x) for x in [variable.iloc[:,i] for i in np.arange(len(variable.columns))]])
    t.add_row(['skewness'] + [sp.stats.skew(x) for x in [variable.iloc[:,i] for i in np.arange(len(variable.columns))]])
    t.add_row(['kurtosis'] + [sp.stats.kurtosis(x) for x in [variable.iloc[:,i] for i in np.arange(len(variable.columns))]])
    
    t.add_row([''] + blanks)
    t.add_row(['Missing Values'] + blanks)   # missing values
    t.add_row(['count'] + [total_count - x.count() for x in [variable.iloc[:,i] for i in np.arange(len(variable.columns))]])
    
    print t

# print univariate analysis for a categorical variable
# pass it a vectorized variable
def categorical(variable, total_count):
    categorical_nrm(variable, total_count)
    categorical_c(variable, total_count)
    categorical_b(variable)
    categorical_p(variable)

# number of categories and range
def categorical_nrm(variable, total_count):
    print '\nNumber of categories: ' + str(len(variable.value_counts()))
    
    this_min = min(variable)
    this_max = max(variable)
    
    t = PrettyTable([variable.name,''])
    t.add_row(['min', this_min])
    t.add_row(['max', this_max])
    t.add_row(['range', this_max-this_min])
    print t
    
    continuous_m(variable, total_count)

# count and count% table for categorical variable
def categorical_c(variable, total_count):
    categories = variable.value_counts().to_frame()
    categories.columns = ['count']
    categories['count%'] = np.nan
    
    count = 0
    for index, row in categories.iterrows():
        categories.iloc[count,1] = row['count']/total_count
        count += 1
    
    print categories

# bar graph
def categorical_b(variable):
    variable.value_counts().sort_index(axis = 0, ascending = True).plot(kind='bar')
    plt.show()

# pie chart
def categorical_p(variable):
    plt.axis('equal')
    variable.value_counts().sort_index(axis = 0, ascending = True).plot(kind='pie')
    plt.show()
