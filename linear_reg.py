import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv(r"C:\Users\star2\Downloads\Salary_Data.csv")

x = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison)

plt.scatter(x_test, y_test, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('Salary vs Experience (Test set')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

m = regressor.coef_
print(m)

c = regressor.intercept_
print(c)

y_12 = m*12+c
print(y_12)

exp_12_future_pred = 9312 * 12 + 26780
exp_12_future_pred

bias = regressor.score(x_train, y_train)
print(bias)

variance = regressor.score(x_test, y_test)
print(variance)

dataset.mean()
dataset['Salary'].mean()
dataset.median()
dataset['Salary'].mode()
dataset.var()
dataset['Salary'].var()
dataset.std()
dataset['Salary'].std()

from scipy.stats import variation #coefficient of variance
variation(dataset.values)
variation(dataset['Salary'])

dataset.corr()
dataset['Salary'].corr(dataset['YearsExperience'])

dataset.skew()
dataset['Salary'].skew()

dataset.sem()  #standard error
dataset['Salary'].sem()

import scipy.stats as stats

dataset.apply(stats.zscore)
stats.zscore(dataset['Salary'])

y_mean = np.mean(y)
SSR = np.sum((y_pred-y_mean)**2)
print(SSR)

y = y[0:6]
SSE = np.sum((y-y_pred)**2)
print(SSE)

mean_total = np.mean(dataset.values)
SST = np.sum((dataset.values-mean_total)**2)
print(SST)

r_square = 1 - (SSR/SST)
r_square

from sklearn.metrics import mean_squared_error
train_mse = mean_squared_error(y_train, regressor.predict(x_train))
test_mse = mean_squared_error(y_test, y_pred)

print(train_mse)
print(test_mse)

import pickle
filename = 



