# Import scikit-learn dataset library
from sklearn import datasets
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