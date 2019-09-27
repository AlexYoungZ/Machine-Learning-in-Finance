import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import KernelPCA
import seaborn as sns
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn import datasets,decomposition,manifold


df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
                      'machine-learning-databases/wine/wine.data',
                      header=None)

df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines', 'Proline']

df_wine.head()
#print head and tail of data frame
print(df_wine.head())
print(df_wine.tail())

#print summary of data frame
summary = df_wine.describe()
print(summary)

# ## Visualizing the important characteristics of a dataset

cols =  ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines', 'Proline']

#pairplotmap

sns.pairplot(df_wine[cols], size=2.5)
plt.tight_layout()
# plt.savefig('images/10_03.png', dpi=300)
plt.show()



#heatmap

cm = np.corrcoef(df_wine[cols].values.T)
hm = sns.heatmap(cm,
                 cbar=True,
                 annot=True,
                 square=True,
                 fmt='.2f',
                 annot_kws={'size': 4},
                 yticklabels=cols,
                 xticklabels=cols)

plt.tight_layout()
plt.show()


#Split data into training and test sets.

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.2, 
                                                   stratify=y,random_state=42)

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)


#Fit a logistic classifier model and print accuracy score


lr = LogisticRegression(penalty='l1', C=1.0)
lr.fit(X_train_std, y_train)
print('Training accuracy:', lr.score(X_train_std, y_train))
print('Test accuracy:', lr.score(X_test_std, y_test))





#clf = SGDClassifier(loss='squared_loss', penalty='l2', random_state=42)
#clf.fit(X_train, y_train)
#
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




lr.intercept_
np.set_printoptions(8)
lr.coef_[lr.coef_!=0].shape
lr.coef_


fig = plt.figure()
ax = plt.subplot(111)
    
colors = ['blue', 'green', 'red', 'cyan', 
          'magenta', 'yellow', 'black', 
          'pink', 'lightgreen', 'lightblue', 
          'gray', 'indigo', 'orange']

weights, params = [], []
for c in np.arange(-4., 6.):
    lr = LogisticRegression(penalty='l1', C=10.**c, random_state=0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)

weights = np.array(weights)

for column, color in zip(range(weights.shape[1]), colors):
    plt.plot(params, weights[:, column],
             label=df_wine.columns[column + 1],
             color=color)
plt.axhline(0, color='black', linestyle='--', linewidth=3)
plt.xlim([10**(-5), 10**5])
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.xscale('log')
plt.legend(loc='upper left')
ax.legend(loc='upper center', 
          bbox_to_anchor=(1.38, 1.03),
          ncol=1, fancybox=True)
plt.show()

#Fit a SVM classifier model and print accuracy score


svm = SVC(kernel='linear', C=1.0, random_state=1)
svm.fit(X_train_std, y_train)
print('Training accuracy:', svm.score(X_train_std, y_train))
print('Test accuracy:', svm.score(X_test_std, y_test))

#clf = SGDClassifier(loss='squared_loss', penalty='l2', random_state=42)
#clf.fit(X_train, y_train)
#
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



# Principal component analysis in scikit-learn
        
 

pca = PCA()
X_train_pca = pca.fit_transform(X_train_std)
pca.explained_variance_ratio_




plt.bar(range(1, 26), pca.explained_variance_ratio_, alpha=0.5, align='center')
plt.step(range(1, 26), np.cumsum(pca.explained_variance_ratio_), where='mid')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')

plt.show()




pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
lr = LogisticRegression()
lr.fit(X_train_pca,y_train)
print('Training accuracy:', lr.score(X_train_pca, y_train))
print('Test accuracy:', lr.score(X_test_pca, y_test))



svm = SVC(kernel='linear', C=1.0, random_state=1)
svm.fit(X_train_pca, y_train)
print('Training accuracy:', svm.score(X_train_pca, y_train))
print('Test accuracy:', svm.score(X_test_pca, y_test))


#y_train_pred = pca.predict(X_train_pca)
#print( metrics.accuracy_score(y_train, y_train_pred) )
##0.821428571429
#
#y_pred = clr.predict(X_test)
#print( metrics.accuracy_score(y_test, y_pred) )
##0.684210526316
#
#print( metrics.classification_report(y_test, y_pred, target_names=iris.target_names) )
##precision recall f1-score support
##setosa 1.00 1.00 1.00 8
##versicolor 0.43 0.27 0.33 11
##virginica 0.65 0.79 0.71 19
##avg / total 0.66 0.68 0.66 38
#
#print( metrics.confusion_matrix(y_test, y_pred) )



plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1])
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.show()
       


plot_decision_regions(X_train_pca, y_train, classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()


plot_decision_regions(X_test_pca, y_test, classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()




# ## LDA via scikit-learn

lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)
X_test_lda = lda.transform(X_test_std)


lr = LogisticRegression()
lr = lr.fit(X_train_lda, y_train)
print('Training accuracy:', lr.score(X_train_lda, y_train))
print('Test accuracy:', lr.score(X_test_lda, y_test))


svm = SVC(kernel='rbf', C=1, random_state=1)
svm.fit(X_train_lda, y_train)
print('Training accuracy:', svm.score(X_train_lda, y_train))
print('Test accuracy:', svm.score(X_test_lda, y_test))


plot_decision_regions(X_train_lda, y_train, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()


plot_decision_regions(X_test_lda, y_test, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()



# ## Kernel principal component analysis in scikit-learn

kpca = KernelPCA(n_components=2, kernel='poly', gamma=10)
X_train_kpca = kpca.fit_transform(X_train_std, y_train)
X_test_kpca = kpca.transform(X_test_std)

lr = LogisticRegression()
lr = lr.fit(X_train_kpca, y_train)
print('Training accuracy:', lr.score(X_train_kpca, y_train))
print('Test accuracy:', lr.score(X_test_kpca, y_test))


svm = SVC(kernel='rbf', C=1.0, random_state=42)
svm.fit(X_train_kpca, y_train)
print('Training accuracy:', svm.score(X_train_kpca, y_train))
print('Test accuracy:', svm.score(X_test_kpca, y_test))


decomposition.IncrementalPCA

def plot_KPCA(*data):
    X,y = data
    kernels = ['linear','poly','rbf','sigmoid']
    fig = plt.figure()

    for i,kernel in enumerate(kernels):
        kpca = decomposition.KernelPCA(n_components=2, kernel=kernel)
        kpca.fit(X)
        X_r = kpca.transform(X)
        ax = fig.add_subplot(2, 2, i+1)
        for label in np.unique(y):
            position = y == label
            ax.scatter(X_r[position,0],X_r[position,1],label="target=%d"%label)
            ax.set_xlabel('x[0]')
            ax.set_ylabel('x[1]')
            ax.legend(loc='best')
            ax.set_title('kernel=%s'% kernel)
    plt.suptitle("KPCA")
    plt.show()
plot_KPCA(X, y)

print("My name is {Siyang Zhang}")
print("My NetID is: {siyangz2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")