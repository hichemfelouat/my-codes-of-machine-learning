import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA

iris = datasets.load_iris()
X = iris.data
y = iris.target
X_pca = PCA(n_components=3).fit_transform(X)
kpca = KernelPCA(n_components=3, kernel='rbf')
X_kpca = kpca.fit_transform(X)

fig = plt.figure(figsize=(18, 8))
ax = fig.add_subplot(1,2,1, projection='3d')
ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y, marker='o',cmap=plt.cm.Set1, edgecolor='k', s=40)
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])

ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.scatter(X_kpca[:, 0], X_kpca[:, 1], X_kpca[:, 2], c=y, marker='o',cmap=plt.cm.Set1, edgecolor='k', s=40)
ax.set_title("First three KPCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])
plt.show()
