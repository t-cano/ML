import numpy as np
import pandas as pd


#read cvs data also drop id column
df=pd.read_csv('Iris.csv')
df.drop(df.columns[[0]], axis=1, inplace = True)
X = df.iloc[:, 0:4].values
y = df.iloc[:, -1].values


#spicies were str data. make them numeric for clasification
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

#train our model whit decision tree
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=3, random_state=0)
tree.fit(X_train, y_train)

#time to skecht the tree
from sklearn.tree import export_graphviz
export_graphviz(tree, out_file="tree.dot", class_names=["Versicolour", "Setosa","Virginica"], feature_names=["SepalLengthCm	", "SepalWidthCm","PetalLengthCm",	"PetalWidthCm"], impurity=False, filled=True)

import graphviz
with open("tree.dot") as f:
  dot_graph = f.read()
graphviz.Source(dot_graph)


#lets check our score numeric way 
print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))
