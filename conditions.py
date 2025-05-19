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
dataset=pd.read_csv("H:\\purvi\\Research Project\\Mental Health Prediction\\condition_1.csv")
data=dataset.dropna()
print(data)
plt.plot(dataset["date"],data["activity"],color="green")
plt.xticks(dataset["date"],rotation="vertical")
plt.xlabel("date")
plt.ylabel("activity")
plt.show()
data1=data.loc[:,["date","activity"]]
print(data1)
print(dataset.info())

#Plotting Autocorrelation plot
autocorrelation_plot(data1["activity"],color="orange")
plt.title("Autocorrelation Plot")
plt.grid()
plt.show()

#fit model
model=ARIMA(data1["activity"].values,order=(5,1,0))
model_fit=model.fit()

#summary of fit model
print(model_fit.summary())

# line plot of residuals
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
plt.title("ARMA Fit Residual Error Line Plot")
plt.show()

# density plot of residuals
residuals.plot(kind='kde')
plt.title("ARMA Fit Residual Error Density Plot")
plt.show()

# summary stats of residuals
print(residuals.describe())

X = data1["activity"].values

# Drop zero values
X = X[X != 0]
size = int(len(X) * 0.66)
train, test = X[:size], X[size:]
history = train.tolist()
predictions = []

# walk-forward validation
for t in range(len(test)):
    model = ARIMA(history, order=(5, 1, 0))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))

# evaluate forecasts
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)

# plot forecasts against actual outcomes
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()
