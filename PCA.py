"""
@author: hichem felouat
"""
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.decomposition import PCA

dat = datasets.load_breast_cancer()
X = dat.data
Y = dat.target
print("Examples = ",X.shape ," Labels = ", Y.shape)

pca = PCA(n_components = 5)
X_pca = pca.fit_transform(X)
print("Examples = ",X_pca.shape ," Labels = ", Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, 
              Y, test_size= 0.20, random_state=100)

X_train_pca, X_test_pca, Y_train_pca, Y_test_pca = train_test_split(X_pca, 
              Y, test_size= 0.20, random_state=100)

clf = svm.SVC(kernel='rbf', gamma= 0.001, C=5)
clf.fit(X_train, Y_train)
y_pred1 = clf.predict(X_test)
print(" Accuracy     :",metrics.accuracy_score(Y_test, y_pred1))

clf_pca = svm.SVC(kernel='rbf', gamma= 0.001, C=5)
clf_pca.fit(X_train_pca, Y_train_pca)
y_pred_pca = clf_pca.predict(X_test_pca)
print(" Accuracy pca :",metrics.accuracy_score(Y_test_pca, y_pred_pca))











