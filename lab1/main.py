# Dependencies test
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Load the train and test datasets to create two DataFrames
train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
train = pd.read_csv(train_url)
test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"
test = pd.read_csv(test_url)

# Fill missing values with mean column values in the train set 
train.fillna(train.mean(numeric_only=True), inplace=True)
# Fill missing values with mean column values in the test set
test.fillna(test.mean(numeric_only=True), inplace=True)

#TESTING
'''
print("***** Train_Set *****")
print(train.head())
print("\n")
print("***** Test_Set *****")
print(test.head())

print("\n")
print("***** Train_Set *****")
print(train.describe())
print(train.columns.values)
'''

# For the train set
train.isna().head()
# For the test set
test.isna().head()

#the total number of missing values in both datasets
'''
print("*****In the train set*****")
print(train.isna().sum())
print("\n")
print("*****In the test set*****")
print(test.isna().sum())
'''

#print(train['Ticket'].head())
#print(train['Cabin'].head())

"""
print("\n*****Pclass*****")
print(train[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean(). sort_values(by='Survived', ascending=False))

print("\n*****Sex*****")
print(train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived',ascending=False))

print("\n*****Sibsp*****")
print(train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived',ascending=False))
"""

#quick-plotting: "Age vs Survived"
"""
g = sns.FacetGrid(train, col='Survived') 
g.map(plt.hist, 'Age', bins=20) 
plt.show()

#how the Pclass and Survived features are related to each other with a graph:
grid = sns.FacetGrid(train, col='Survived', row='Pclass', aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()
plt.show()
"""

train = train.drop(['Name','Ticket', 'Cabin','Embarked'], axis=1)
test = test.drop(['Name','Ticket', 'Cabin','Embarked'], axis=1)

labelEncoder = LabelEncoder()
labelEncoder.fit(train['Sex'])
labelEncoder.fit(test['Sex'])
train['Sex'] = labelEncoder.transform(train['Sex']) 
test['Sex'] = labelEncoder.transform(test['Sex'])

X = np.array(train.drop(['Survived'],axis=1).astype(float))
y = np.array(train['Survived'])

train.info()
test.info()


kmeans = KMeans(n_clusters=2, max_iter=600, algorithm = 'lloyd')
kmeans.fit(X)
KMeans(algorithm='lloyd', copy_x=True, init='k-means++', max_iter=600, n_clusters=2, n_init=10, random_state=None, tol=0.0001, verbose=0)
correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1 
print(correct/len(X))


scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
kmeans.fit(X_scaled)
KMeans(algorithm='lloyd', copy_x=True, init='k-means++', max_iter=600,
n_clusters=2, n_init=10, random_state=None, tol=0.0001, verbose=0)
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1
print(correct/len(X))


