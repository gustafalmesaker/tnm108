# Import scikit-learn dataset library
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


# Load dataset
wine = datasets.load_wine()

# print the names of the features
#print(wine.feature_names)
# print the label species(class_0, class_1, class_2)
#print(wine.target_names)
# print the wine data (top 5 records)
#print(wine.data[0:5])
# print the wine labels (0:Class_0, 1:Class_1, 2:Class_3)
#print(wine.target)
# print data(feature)shape
#print(wine.data.shape)
# print target(or label)shape
#print(wine.target.shape)

#SPLITTING DATA - To understand the model performance you can divide the dataset into a training set and a test set.
# Import train_test_split function
# Split dataset into training set and test set: 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3)
# Create KNN Classifier
knn = KNeighborsClassifier(n_neighbors=5)
# Train the model using the training sets
knn.fit(X_train, y_train)
# Predict the response for test dataset
y_pred = knn.predict(X_test)

#GENERATING MODEL FOR K=5
# Create KNN Classifier
knn = KNeighborsClassifier(n_neighbors=5)
# Train the model using the training sets
knn.fit(X_train, y_train)
# Predict the response for test dataset
y_pred = knn.predict(X_test)
#MODEL EVALUTION FOR K=5
# Import scikit-learn metrics module for accuracy calculation
# Model Accuracy, how often is the classifier correct?
print("Accuracy for k=5: ",metrics.accuracy_score(y_test, y_pred))

#REGENERATING MODEL FOR K=7
knn = KNeighborsClassifier(n_neighbors=13)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
#Model Evaluation for k=7
print("Accuracy for k=7: ",metrics.accuracy_score(y_test, y_pred))