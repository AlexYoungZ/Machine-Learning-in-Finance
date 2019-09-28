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


df = pd.read_csv("/Users/siyangzhang/Desktop/课程/ie598 machine learning in finance/groupproject/MLF_GP1_CreditScore.csv")

df.isnull().any()

X, y = df.iloc[0:1700,0:26], df.InvGrd

print( X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25, random_state=42)
print( X_train.shape, y_train.shape)

# Standardize the features

#scaler = preprocessing.StandardScaler().fit(X_train)
#X_train = scaler.transform(X_train)
#X_test = scaler.transform(X_test)

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


pca = PCA()
X_train_pca = pca.fit_transform(X_train_std)
pca.explained_variance_ratio_


#
#plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1])
#plot_decision_regions(X_train_pca, y_train, classifier=lr)
#plt.xlabel('PC 1')
#plt.ylabel('PC 2')
#plt.legend(loc='lower left')
#plt.show()


pca.explained_variance_ratio_.shape      

plt.bar(range(0, 26), pca.explained_variance_ratio_, alpha=0.5, align='center')
plt.step(range(0, 26), np.cumsum(pca.explained_variance_ratio_), where='mid')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')

plt.show()





pca = PCA()
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.fit_transform(X_test_std)

lr = LogisticRegression()
lr.fit(X_train_pca,y_train)

print('Training accuracy:', lr.score(X_train_pca, y_train))
print('Test accuracy:', lr.score(X_test_pca, y_test))





# ## LDA via scikit-learn




lda = LDA()
X_train_lda = lda.fit_transform(X_train_std, y_train)
X_test_lda = lda.transform(X_test_std)



lr = LogisticRegression()
lr = lr.fit(X_train_lda, y_train)
print('Training accuracy:', lr.score(X_train_lda, y_train))
print('Test accuracy:', lr.score(X_test_lda, y_test))





#plot_decision_regions(X_train_lda, y_train, classifier=lr)
#plt.xlabel('LD 1')
#plt.ylabel('LD 2')
#plt.legend(loc='lower left')
#plt.tight_layout()
#plt.show()
#
#
#plot_decision_regions(X_test_lda, y_test, classifier=lr)
#plt.xlabel('LD 1')
#plt.ylabel('LD 2')
#plt.legend(loc='lower left')
#plt.tight_layout()
#plt.show()



# ## Kernel principal component analysis in scikit-learn

kpca = KernelPCA(n_components=15, kernel='sigmoid', gamma=10)
X_train_kpca = kpca.fit_transform(X_train_std, y_train)
X_test_kpca = kpca.transform(X_test_std)

lr = LogisticRegression()
lr = lr.fit(X_train_kpca, y_train)
print('Training accuracy:', lr.score(X_train_kpca, y_train))
print('Test accuracy:', lr.score(X_test_kpca, y_test))



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
#

#Fit a logistic classifier model and print accuracy score


lr = LogisticRegression()
lr.fit(X_train_std, y_train)
print('Training accuracy:', lr.score(X_train_std, y_train))
print('Test accuracy:', lr.score(X_test_std, y_test))



#clf = SGDClassifier(loss='squared_loss', penalty='l2', random_state=42)
#clf.fit(X_train, y_train)

#y_train_pred = clf.predict(X_train)
#print( metrics.accuracy_score(y_train, y_train_pred) )
#
#
#y_pred = clf.predict(X_test)
#print( metrics.accuracy_score(y_test, y_pred) )
#
#
#print( metrics.classification_report(y_test, y_pred) )
#
#
#print( metrics.confusion_matrix(y_test, y_pred) )


#lr.intercept_
#np.set_printoptions(8)
#lr.coef_[lr.coef_!=0].shape
#lr.coef_


#fig = plt.figure()
#ax = plt.subplot(111)
#    
#colors = ['blue', 'green', 'red', 'cyan', 
#          'magenta', 'yellow', 'black', 
#          'pink', 'lightgreen', 'lightblue', 
#          'gray', 'indigo', 'orange']
#
#weights, params = [], []
#for c in np.arange(-4., 6.):
#    lr = LogisticRegression(penalty='l1', C=10.**c, random_state=0)
#    lr.fit(X_train_std, y_train)
#    weights.append(lr.coef_[1])
#    params.append(10**c)
#
#weights = np.array(weights)
#
#for column, color in zip(range(weights.shape[1]), colors):
#    plt.plot(params, weights[:, column],
#             label=df.columns[column + 1],
#             color=color)
#plt.axhline(0, color='black', linestyle='--', linewidth=3)
#plt.xlim([10**(-5), 10**5])
#plt.ylabel('weight coefficient')
#plt.xlabel('C')
#plt.xscale('log')
#plt.legend(loc='upper left')
#ax.legend(loc='upper center', 
#          bbox_to_anchor=(1.38, 1.03),
#          ncol=1, fancybox=True)
#plt.show()



# ## Combining transformers and estimators in a pipeline



pipe_lr = make_pipeline(StandardScaler(),
                        PCA(n_components=15),
                        LogisticRegression(random_state=1))

pipe_lr.fit(X_train, y_train)
y_pred = pipe_lr.predict(X_test)
print('Test Accuracy: %.3f' % pipe_lr.score(X_test, y_test))



#Use GridSearchCV to hypertune the parameters
tuned_parameters={'C':scipy.stats.expon(scale=100),
                  'multi_class':['ovr','multinomial']}
clf=RandomizedSearchCV(LogisticRegression(penalty='l2',solver='lbfgs',tol=1e-6),
                   tuned_parameters,cv=10,scoring='accuracy',n_iter=100)
 
clf.fit(X_train,y_train)
print('best parameters:',clf.best_estimator_)
print(classification_report(y_test,clf.predict(X_test)))
print(metrics.confusion_matrix(y_test,clf.predict(X_test)))

print(clf.best_score_)
print(clf.best_params_)

print('Test accuracy: %.3f' % clf.score(X_test, y_test))
    
#After tuning

lr = LogisticRegression(C=55.992073973395804, class_weight=None, dual=False,
          fit_intercept=True, intercept_scaling=1, max_iter=100,
          multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
          solver='lbfgs', tol=1e-06, verbose=0, warm_start=False)
lr.fit(X_train_std, y_train)
print('Training accuracy:', lr.score(X_train_std, y_train))
print('Test accuracy:', lr.score(X_test_std, y_test))




# ## K-fold cross-validation




#kfold = StratifiedKFold(n_splits=10,
#                        random_state=1).split(X_train, y_train)
#
#scores = []
#for k, (train, test) in enumerate(kfold):
#    pipe_lr.fit(X_train[train], y_train[train])
#    score = pipe_lr.score(X_train[test], y_train[test])
#    scores.append(score)
#    print('Fold: %2d, Class dist.: %s, Acc: %.3f' % (k+1,
#          np.bincount(y_train[train]), score))
#    
#print('\nCV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))





scores = cross_val_score(estimator=pipe_lr,
                         X=X_train,
                         y=y_train,
                         cv=10,
                         n_jobs=1)
print('CV accuracy scores: %s' % scores)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))




# # Debugging algorithms with learning curves


# ## Diagnosing bias and variance problems with learning curves



pipe_lr = make_pipeline(StandardScaler(),
                        LogisticRegression(C=55.992073973395804, class_weight=None, dual=False,
          fit_intercept=True, intercept_scaling=1, max_iter=100,
          multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
          solver='lbfgs', tol=1e-06, verbose=0, warm_start=False))

train_sizes, train_scores, test_scores = learning_curve(estimator=pipe_lr,
                               X=X_train,
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
#plt.savefig('images/06_05.png', dpi=300)
plt.show()



# ## Addressing over- and underfitting with validation curves

param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
train_scores, test_scores = validation_curve(
                estimator=pipe_lr, 
                X=X_train, 
                y=y_train, 
                param_name='logisticregression__C', 
                param_range=param_range,
                cv=10)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(param_range, train_mean, 
         color='blue', marker='o', 
         markersize=5, label='training accuracy')

plt.fill_between(param_range, train_mean + train_std,
                 train_mean - train_std, alpha=0.15,
                 color='blue')

plt.plot(param_range, test_mean, 
         color='green', linestyle='--', 
         marker='s', markersize=5, 
         label='validation accuracy')

plt.fill_between(param_range, 
                 test_mean + test_std,
                 test_mean - test_std, 
                 alpha=0.15, color='green')

plt.grid()
plt.xscale('log')
plt.legend(loc='lower right')
plt.xlabel('Parameter C')
plt.ylabel('Accuracy')
plt.ylim([0.0, 1.0])
plt.tight_layout()
# plt.savefig('images/06_06.png', dpi=300)
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





