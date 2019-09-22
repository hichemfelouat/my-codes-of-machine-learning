"""
@author: hichem
"""
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn import metrics

dat = datasets.load_iris()
print("Examples = ",dat.data.shape ," Labels = ", 
      dat.target.shape)
print("Example 0 = ",dat.data[0])
print("Label 0 =",dat.target[0])
# Make a train/test split using 20% test size
X_train, X_test, Y_train, Y_test = train_test_split(dat.data, 
                dat.target, test_size= 0.20, random_state=100)

# Create adaboost classifer object
abc = AdaBoostClassifier(n_estimators=200, learning_rate=1)
# Train Adaboost Classifer
model1 = abc.fit(X_train, Y_train)
#Predict the response for test dataset
y_pred1 = model1.predict(X_test)
# Model Accuracy, how often is the classifier correct?
print("Accuracy1 : ", metrics.accuracy_score(Y_test, y_pred1))

model2 = GradientBoostingClassifier(n_estimators=200, 
        learning_rate= 0.5, max_features=4, max_depth=2, 
        random_state=0)
model2.fit(X_train, Y_train)
print("Accuracy2 : ", model2.score(X_test, Y_test))











