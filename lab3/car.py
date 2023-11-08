import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score



df = pd.read_csv('C:\Users\guffi\Documents\GitHub\tnm108\lab3\data_cars.csv',header=None)
for i in range(len(df.columns)):
 df[i] = df[i].astype('category')
df.head()