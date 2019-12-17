from sklearn.datasets.samples_generator import make_blobs
from sklearn import preprocessing 
from mirapy.visualization import visualize_3d

X, y = make_blobs(n_samples= 50000, centers=4, 
                  n_features=3)
print("Examples = ",X.shape ," Labels = ", y.shape)

visualize_3d(X,y)

poly = preprocessing.PolynomialFeatures(degree=3, 
                         interaction_only=True)
X_1 = poly.fit_transform(X)




















print("new features : ", X_1.shape)
 



