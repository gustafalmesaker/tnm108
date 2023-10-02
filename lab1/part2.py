import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering


#X = np.array([[5,3], [10,15], [15,12], [24,10], [30,30], [85,70], [71,80], [60,78], [70,55], [80,91] ])

# Import shopping dataset
customer_data = pd.read_csv('lab1\shopping_data.csv')

# Check number of records and attributes
print('\n')
print('Number of records and attributes: ')
print(customer_data.shape)

# Check 2 last columns annual income ($k) and spending score (1-100)
data = customer_data.iloc[:, 3:5].values
print('\n')
print('Last two columns: ')
print(data)

X = data


labels = range(1, 11)
plt.figure(figsize=(10, 7))
plt.subplots_adjust(bottom=0.1)
plt.scatter(X[:,0],X[:,1], label='True Position')

for label, x, y in zip(labels, X[:, 0], X[:, 1]):
    plt.annotate(label,xy=(x, y),xytext=(-3, 3),textcoords='offset points', ha='right',va='bottom')
plt.show()


linked = linkage(X, 'ward')
labelList = range(1, len(X)+1)
plt.figure(figsize=(10 , 7))
dendrogram(linked, orientation='top', labels=labelList, distance_sort='descending', show_leaf_counts=True)
plt.show()


cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
cluster.fit_predict(X)
plt.scatter(X[:,0],X[:,1], c=cluster.labels_, cmap='rainbow')
plt.show()
