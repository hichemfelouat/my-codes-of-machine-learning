"""@author: hichem
"""
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn import model_selection
from sklearn import metrics

dat = datasets.load_breast_cancer()
print("Examples = ",dat.data.shape ," Labels = ", dat.target.shape)
print("Example 0 = ",dat.data[0])
print("Label 0 =",dat.target[0])
print(dat.target)
X = dat.data
Y = dat.target
# Make a train/test split using 20% test size
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.20, 
                                                    random_state=100)
print("X_test = ",X_test.shape)

print("Without Validation : *********")
model_1 = svm.SVC(kernel='linear', C=10.0, gamma= 0.1)
model_1.fit(X_train, Y_train)
y_pred1 = model_1.predict(X_test)
print("Accuracy 1 :",metrics.accuracy_score(Y_test, y_pred1))

print("K-fold Cross-Validation : *********")
from sklearn.model_selection import KFold
kfold = KFold(n_splits=10, random_state=100)
model_2 = svm.SVC(kernel='linear', C=10.0, gamma= 0.1)
results_model_2 = cross_val_score(model_2, X, Y, cv=kfold)
accuracy2 = results_model_2.mean()
print("Accuracy 2 :", accuracy2)

print("Stratified K-fold Cross-Validation : *********")
from sklearn.model_selection import StratifiedKFold
skfold = StratifiedKFold(n_splits=3, random_state=100)
model_3 = svm.SVC(kernel='linear', C=10.0, gamma= 0.1)
results_model_3 = cross_val_score(model_3, X, Y, cv=skfold)
accuracy3  = results_model_3.mean()
print("Accuracy 3 :", accuracy3)

print("Leave One Out Cross-Validation : *********")
from sklearn.model_selection import LeaveOneOut
loocv = model_selection.LeaveOneOut()
model_4 = svm.SVC(kernel='linear', C=10.0, gamma= 0.1)
results_model_4 = cross_val_score(model_4, X, Y, cv=loocv)
accuracy4  = results_model_4.mean()
print("Accuracy 4 :", accuracy4)

print("Repeated Random Test-Train Splits : *********")
from sklearn.model_selection import ShuffleSplit
kfold2 = model_selection.ShuffleSplit(n_splits=10, test_size=0.30, 
                                      random_state=100)
model_5 = svm.SVC(kernel='linear', C=10.0, gamma= 0.1)
results_model_5 = cross_val_score(model_5, X, Y, cv=kfold2)
accuracy5  = results_model_5.mean()
print("Accuracy 5 :", accuracy5)











