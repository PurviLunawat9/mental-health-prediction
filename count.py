
import warnings
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pandas.plotting import autocorrelation_plot
import matplotlib.pyplot as plt
from datetime import datetime
from math import sqrt
from sklearn.metrics import mean_squared_error

# Import CSV to Python
def parser(x):
    return datetime.strptime(x,'%y/%m/%d')
dataset=pd.read_csv("H:\\purvi\\Research Project\\data\\condition\\condition_1.csv")
data=dataset.dropna()
print(data)
data1=data.loc[:,["date","activity"]]
print(data1)
print(dataset.info())
count = (dataset["activity"] == 0).sum()
print('Count of zeros in Column  activity : ', count)
