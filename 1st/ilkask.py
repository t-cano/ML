import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv('/content/Student_performance_data _.csv')

# try to make absences column to Attendance so firstly find max value absent 
maxValue = df['Absences'].max()

# after that subtract all rows from max number
for index, row in df.iterrows():
    df.at[index, 'Absences'] =  29 - row['Absences'] 
#rename Absences
df = df.rename(columns={'Absences': 'Attendance'})


X = df[['Attendance']].values
y= df['GPA'].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

#build model for correlation line
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

#sketch the graph
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Attendance / GPA Correlation')
plt.xlabel('Absences')
plt.ylabel('GPA')
plt.show()