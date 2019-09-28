#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 13:11:26 2018

@author: siyangzhang
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 16:07:58 2018

@author: siyangzhang
"""

from sklearn.preprocessing import Imputer
import numpy as np
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import KernelPCA
from sklearn import datasets,decomposition,manifold
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.utils import resample
from sklearn.datasets import load_digits
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
import scipy
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


df = pd.read_csv("/Users/siyangzhang/Desktop/课程/ie598 machine learning in finance/groupproject/MLF_GP1_CreditScore.csv")

df.isnull().any()

X, y = df.iloc[0:1700,0:26], df.Rating

print( X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25, random_state=42)
print( X_train.shape, y_train.shape)

# Standardize the features


sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)


#print summary of data frame
summary = X.describe()
print(summary)

print(df.head())
print(df.tail())

# Principal component analysis in scikit-learn


def plot_decision_regions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.6, 
                    c=cmap(idx),
                    edgecolor='black',
                    marker=markers[idx], 
                    label=cl)

#PCA
        
pca = PCA()
X_train_pca = pca.fit_transform(X_train_std)
pca.explained_variance_ratio_



pca.explained_variance_ratio_.shape      

plt.bar(range(0, 26), pca.explained_variance_ratio_, alpha=0.5, align='center')
plt.step(range(0, 26), np.cumsum(pca.explained_variance_ratio_), where='mid')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')

plt.show()



pca = PCA(n_components=15)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)



knn = KNeighborsClassifier(n_neighbors=1, 
                           p=2, 
                           metric='minkowski')
knn.fit(X_train_pca, y_train)

print('Training accuracy:', knn.score(X_train_pca, y_train))
print('Test accuracy:', knn.score(X_test_pca, y_test))


#y_train_pred = pca.predict(X_train_pca)
#print( metrics.accuracy_score(y_train, y_train_pred) )


# ## LDA via scikit-learn


lda = LDA()
X_train_lda = lda.fit_transform(X_train_std, y_train)
X_test_lda = lda.transform(X_test_std)

lda.explained_variance_ratio_.shape      

plt.bar(range(0, 15), lda.explained_variance_ratio_, alpha=0.5, align='center')
plt.step(range(0, 15), np.cumsum(lda.explained_variance_ratio_), where='mid')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')

plt.show()


lda = LDA()
X_train_lda = lda.fit_transform(X_train_std, y_train)
X_test_lda = lda.transform(X_test_std)




knn= KNeighborsClassifier(n_neighbors=1, 
                        p=2, 
                        metric='minkowski',
                        algorithm='auto',
                        leaf_size=1,
                        weights='uniform')
knn.fit(X_train_lda, y_train)

print('Training accuracy:', knn.score(X_train_lda, y_train))
print('Test accuracy:', knn.score(X_test_lda, y_test))




#plot_decision_regions(X_train_lda, y_train, classifier=knn)
#plt.xlabel('LD 1')
#plt.ylabel('LD 2')
#plt.legend(loc='lower left')
#plt.tight_layout()
#plt.show()

#
#
#plot_decision_regions(X_test_lda, y_test, classifier=knn)
#plt.xlabel('LD 1')
#plt.ylabel('LD 2')
#plt.legend(loc='lower left')
#plt.tight_layout()
#plt.show()



# ## Kernel principal component analysis in scikit-learn

kpca = KernelPCA()
X_train_kpca = kpca.fit_transform(X_train_std, y_train)
X_test_kpca = kpca.transform(X_test_std)

knn= KNeighborsClassifier(n_neighbors=1, 
                        p=2, 
                        metric='minkowski',
                        algorithm='auto',
                        leaf_size=1,
                        weights='uniform')
knn.fit(X_train_kpca, y_train)

print('Training accuracy:', knn.score(X_train_kpca, y_train))
print('Test accuracy:', knn.score(X_test_kpca, y_test))



#decomposition.IncrementalPCA
#
#def plot_KPCA(*data):
#    X,y = data
#    kernels = ['linear','poly','rbf','sigmoid']
#    fig = plt.figure()
#
#    for i,kernel in enumerate(kernels):
#        kpca = decomposition.KernelPCA(n_components=15, kernel=kernel)
#        kpca.fit(X_train)
#        X_r = kpca.transform(X_train_std)
#        ax = fig.add_subplot(2, 2, i+1)
#        for label in np.unique(y):
#            position = y_train == label
#            ax.scatter(X_r[position,0],X_r[position,1],label="target=%d"%label)
#            ax.set_xlabel('x[0]')
#            ax.set_ylabel('x[1]')
#            ax.legend(loc='best')
#            ax.set_title('kernel=%s'% kernel)
#    plt.suptitle("KPCA")
#    plt.show()
#plot_KPCA(X_train, y_train)

#KNN before hypertuning

knn = KNeighborsClassifier(n_neighbors=1, 
                           p=2, 
                           metric='minkowski')
knn.fit(X_train_std, y_train)

print('Training accuracy:', knn.score(X_train_std, y_train))
print('Test accuracy:', knn.score(X_test_std, y_test))



#Use GridSearchCV to hypertune the parameters

knn = KNeighborsClassifier()
k_range = list(range(1,10))
leaf_range = list(range(1,2))
weight_options = ['uniform','distance']
algorithm_options = ['auto','ball_tree','kd_tree','brute']
param_gridknn = dict(n_neighbors = k_range,weights = weight_options,algorithm=algorithm_options,leaf_size=leaf_range)
gridKNN = GridSearchCV(knn,param_gridknn,cv=10,scoring='accuracy',verbose=1)
gridKNN.fit(X_train,y_train)
print('best score is:',str(gridKNN.best_score_))
print('best params are:',str(gridKNN.best_params_))


knn= KNeighborsClassifier(n_neighbors=1, 
                        p=2, 
                        metric='minkowski',
                        algorithm='auto',
                        leaf_size=1,
                        weights='uniform')
knn.fit(X_train_std, y_train)

print('Training accuracy:', knn.score(X_train_std, y_train))
print('Test accuracy:', knn.score(X_test_std, y_test))




y_train_pred = knn.predict(X_train_std)
print( metrics.accuracy_score(y_train, y_train_pred) )
#
#
y_pred = knn.predict(X_test_std)
print( metrics.accuracy_score(y_test, y_pred) )
#
#
print( metrics.classification_report(y_test, y_pred) )
#
#
print( metrics.confusion_matrix(y_test, y_pred) )




# ## Combining transformers and estimators in a pipeline



pipe_knn = make_pipeline(StandardScaler(),
                        PCA(n_components=15),
                        KNeighborsClassifier(n_neighbors=1, 
                        p=2, 
                        metric='minkowski',
                        algorithm='auto',
                        leaf_size=1,
                        weights='uniform'))

pipe_knn.fit(X_train_pca, y_train)
y_pred = pipe_knn.predict(X_test_pca)
print('Test Accuracy: %.3f' % pipe_knn.score(X_test_pca, y_test))


    
 ## K-fold cross-validation




#kfold = StratifiedKFold(n_splits=10,
#                        random_state=1).split(X_train, y_train)
#
#scores = []
#for k, (train, test) in enumerate(kfold):
#    pipe_knn.fit(X_train[train], y_train[train])
#    score = pipe_knn.score(X_train[test], y_train[test])
#    scores.append(score)
#    print('Fold: %2d, Class dist.: %s, Acc: %.3f' % (k+1,
#          np.bincount(y_train[train]), score))
#    
#print('\nCV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))





scores = cross_val_score(estimator=pipe_knn,
                         X=X_train_pca,
                         y=y_train,
                         cv=10,
                         n_jobs=1)
print('CV accuracy scores: %s' % scores)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))




# # Debugging algorithms with learning curves


# ## Diagnosing bias and variance problems with learning curves





train_sizes, train_scores, test_scores = learning_curve(estimator=pipe_knn,
                               X=X_train_pca,
                               y=y_train,
                               train_sizes=np.linspace(0.1, 1.0, 10),
                               cv=10,
                               n_jobs=1)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean,
         color='blue', marker='o',
         markersize=5, label='training accuracy')

plt.fill_between(train_sizes,
                 train_mean + train_std,
                 train_mean - train_std,
                 alpha=0.15, color='blue')

plt.plot(train_sizes, test_mean,
         color='green', linestyle='--',
         marker='s', markersize=5,
         label='validation accuracy')

plt.fill_between(train_sizes,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha=0.15, color='green')

plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.0, 1.0])
plt.tight_layout()
plt.show()



# ## Bagging

tree = DecisionTreeClassifier(criterion='entropy', 
                              max_depth=None,
                              random_state=1)

bag = BaggingClassifier(base_estimator=tree,
                        n_estimators=500, 
                        max_samples=1.0, 
                        max_features=1.0, 
                        bootstrap=True, 
                        bootstrap_features=False, 
                        n_jobs=1, 
                        random_state=1)


tree = tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)

tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)
print('Decision tree train/test accuracies %.3f/%.3f'
      % (tree_train, tree_test))


bag = bag.fit(X_train, y_train)
y_train_pred = bag.predict(X_train)
y_test_pred = bag.predict(X_test)

bag_train = accuracy_score(y_train, y_train_pred) 
bag_test = accuracy_score(y_test, y_test_pred) 
print('Bagging train/test accuracies %.3f/%.3f'
      % (bag_train, bag_test))





