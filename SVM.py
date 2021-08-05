import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, plot_confusion_matrix, confusion_matrix

#get iris data
data = datasets.load_iris()
X = data.data
y = data.target

#slip train test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

#get model and its parameter
clf = SVC(C=0.1)

#train model
clf.fit(X_train, y_train)

#predict 
y_pred = clf.predict(X_test)

#find accuracy
acc = accuracy_score(y_test, y_pred)
print(acc)