#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 09:29:52 2018

@author: siyangzhang
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
import time

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
Insample_score=[]
Out_of_sample_score=[]

start = time.clock()

for i in range(1,11):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=i, stratify=y)
    tree = DecisionTreeClassifier(criterion='gini',
                                  max_depth=4,
                                  random_state=1)
    tree.fit(X_train, y_train)
    y_test_pred = tree.predict(X_test)
    y_train_pred = tree.predict(X_train)    
    IS_score = metrics.accuracy_score(y_train, y_train_pred)
    OS_score = metrics.accuracy_score(y_test, y_test_pred)
    Insample_score.append(IS_score)    
    Out_of_sample_score.append(OS_score)
    print('Random State: %d, In-sample score: %.3f, Out of sample score: %.3f' % (i,IS_score,OS_score))

print("Insample mean:",np.mean(Insample_score),"stddev:",np.std(Insample_score))
print("Out_of_sample mean:",np.mean(Out_of_sample_score),"stddev:",np.std(Out_of_sample_score))

end = time.clock()

print (end-start)



X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.1, 
                                                    random_state=1, 
                                                    stratify=y)


start = time.clock()

kfold = KFold(n_splits=10,random_state=1).split(X_train, y_train)

tree = DecisionTreeClassifier(criterion='gini',
                                  max_depth=4,
                                  random_state=1)

scores = cross_val_score(estimator=tree,
                         X=X_train,
                         y=y_train,
                         cv=10,
                         n_jobs=1)

print('CV accuracy scores: %s' % scores)
print('CV accuracy mean/std: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
    
end = time.clock()

print (end-start)


# # Fine-tuning machine learning models via grid search


# ## Tuning hyperparameters via grid search 

start = time.clock()

param_range = {'random_state':range(0,10,1)}
gsearch1 = GridSearchCV(estimator = tree,
                        param_grid = param_range, 
                        scoring='accuracy',
                        cv=10)

gsearch1.fit(X_train, y_train)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_

end = time.clock()

print (end-start)


print("My name is {Siyang Zhang")
print("My NetID is: {siyangz2}")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")



