import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


d1 = "The sky is blue."
d2 = "The sun is bright."
d3 = "The sun in the sky is bright."
d4 = "We can see the shining sun, the bright sun."
Z = (d1,d2,d3,d4)


vectorizer = CountVectorizer()

#stop words
my_stop_words={"the","is"}
my_vocabulary={'blue': 0, 'sun': 1, 'bright': 2, 'sky': 3}
vectorizer=CountVectorizer(stop_words=my_stop_words,vocabulary=my_vocabulary)

#print the stop words list and the vocabulary
#print(vectorizer.vocabulary)
#print(vectorizer.stop_words)

#use our vectorizer to print the sparse matrix of the document set, Scipy sparse matrix with elements stored in a coordinate format
smatrix = vectorizer.transform(Z)
#print(smatrix)

#convert it into a dense format, Note that the matrix created has the format |Z| x F.
matrix = smatrix.todense()
#print(matrix)

#COMPUTING THE TF-IDF SCORE
#To calculate the tf-idf value of our documents we will use the TfidfTransformer module provided by
#Scikit.learn on the word counts (smatrix) we computed earlier.
tfidf_transformer = TfidfTransformer(norm="l2")
tfidf_transformer.fit(smatrix)

# print idf values
#feature_names = vectorizer.get_feature_names()
feature_names = vectorizer.get_feature_names_out()

df_idf=pd.DataFrame(tfidf_transformer.idf_, index=feature_names,columns=["idf_weights"])
# sort ascending
df_idf.sort_values(by=['idf_weights'])
#print(df_idf)

# tf-idf scores
tf_idf_vector = tfidf_transformer.transform(smatrix)

# get tfidf vector for first document
first_document = tf_idf_vector[0] # first document "The sky is blue."

# print the scores
df=pd.DataFrame(first_document.T.todense(), index=feature_names, columns=["tfidf"])
df.sort_values(by=["tfidf"],ascending=False)
#print(df)


#DOCUMENT SIMILARITY
#Cosine Similarity for Vector Space Models
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(Z)
print(tfidf_matrix.shape)

cos_similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix)
print(cos_similarity)

# Take the cos similarity of the third document (cos similarity=0.52)
angle_in_radians = math.acos(cos_similarity)
print(math.degrees(angle_in_radians))

