from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
dat = load_breast_cancer()
X = dat.data
Y = dat.target
print("Examples = ",X.shape ," Labels = ", Y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 
                                                    0.3, random_state=100)
model_pipeline = Pipeline(steps=[
  ("feature_union", FeatureUnion([
    ('missing_values',SimpleImputer(missing_values=np.nan, strategy='mean')),
    ('scale', StandardScaler()),
    ("reduce_dim", PCA(n_components=10)),
  ])),
  ('clf', SVC(kernel='rbf', gamma= 0.001, C=5)) ])
model_pipeline.fit(X_train, y_train)
predictions = model_pipeline.predict(X_test)
print(" Accuracy :",metrics.accuracy_score(y_test, predictions))
