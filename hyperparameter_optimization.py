
"""
@author: hichem felouat
"""
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

dat = datasets.load_breast_cancer()
print("Examples = ",dat.data.shape ," Labels = ", dat.target.shape)
X_train, X_test, Y_train, Y_test = train_test_split(dat.data, 
              dat.target, test_size= 0.20, random_state=100)

param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [2, 20, 200, 2000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},]
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3) 
grid.fit(X_train, Y_train)
print('The best parameter after tuning :',grid.best_params_)  
print('our model looks after hyper-parameter tuning',grid.best_estimator_)
grid_predictions = grid.predict(X_test)  
print(classification_report(Y_test, grid_predictions))


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import expon
param_rdsearch = {'C': expon(scale=100), 'gamma': expon(scale=.1),
  'kernel': ['rbf'], 'class_weight':['balanced', None]}
clf_rds = RandomizedSearchCV(SVC(), param_rdsearch, n_iter=100)
clf_rds.fit(X_train, Y_train)
print("Best: %f using %s" % (clf_rds.best_score_, 
                             clf_rds.best_params_))
rds_predictions = clf_rds.predict(X_test)  
print(classification_report(Y_test, rds_predictions))















