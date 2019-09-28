from sklearn.preprocessing import Imputer
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

df = pd.read_csv("./MLF_GP1_CreditScore.csv")

df.isnull().any()

X, y = df.iloc[0:1700,0:26], df.InvGrd

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25, random_state=33)

scaler = preprocessing.StandardScaler().fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

param_grid = [
	{
		'C': range(5, 1005, 50),
		'degree': [1,2,3,4,5]
	}
]

kernel_svm = SVC(kernel="poly")

grid_search = GridSearchCV(kernel_svm, )

kernel_svm.fit(X_train, y_train)
y_pred = kernel_svm.predict(X_test)
print(accuracy_score(y_test, y_pred))

kernel_svm = SVC(kernel="rbf", C=1000)
kernel_svm.fit(X_train, y_train)
y_pred = kernel_svm.predict(X_test)
print(accuracy_score(y_test, y_pred))

