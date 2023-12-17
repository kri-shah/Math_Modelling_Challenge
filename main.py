#api declarations
import pandas_datareader.data as web
import datetime
import pandas as pd 
import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import seaborn as sns

#set-up for graph
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
sns.set()

#check that everything installed/loaded correctly
print("Connection was succesful!")

#read file, and signify that the first column is the date
data = pd.read_csv("Data.csv")
data.index = pd.to_datetime(data['Date'], format='%Y')
del data['Date']

#establish test vs train data
train = data[data.index < pd.to_datetime("2019", format='%Y')]
test = data[data.index > pd.to_datetime("2021", format='%Y')]

#create SARIMAX model
y = train['Mining, logging, construction'] 

ARMAmodel = SARIMAX(y, order = (1, 0, 1))
ARMAmodel = ARMAmodel.fit() 
