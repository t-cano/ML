import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Student_performance_data _.csv')
from sklearn.linear_model import LinearRegression
X = dataset['Absences'].values
y= dataset['GPA'].values

#when we used one column that "[:, None]" part necessary for arry format
#If use more than one column just delete it 
X = X[:, None]
print(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

#Create our model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#let's sketch the graph
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('derse girmeme vs GPA (Training set)')
plt.xlabel('ABSENT')
plt.ylabel('GPA')
plt.show()