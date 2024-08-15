
import pandas as pd
import numpy as np

#those columns has one to one correlation with world hapiness score so I just drop them
df = pd.read_csv('world-happiness-report-2021.csv')
df.drop(df.columns[[0,1,3,4,5,12,13,14,15,16,17,18,19]],axis=1, inplace =True)

X = df.iloc[:, 1:].values
y = df.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

#future scale for better result
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Create our model
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train)

#Score 
print("Accuracy on training set: {:.3f}".format(regressor.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(regressor.score(X_test, y_test)))