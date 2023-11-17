import numpy as np
from sklearn import metrics
import sklearn
from sklearn.datasets import load_files
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier

#nltk.download('punkt')


#1. Loading Movie Review Data
# loading all files. 
moviedir = r'lab4\movie_reviews'
movie = load_files(moviedir, shuffle=True)


#2. Splitting Data 
# Split data into training and test sets
movie_data_train, movie_data_test, movie_target_train, movie_target_test = train_test_split(
    movie.data, 
    movie.target, 
    test_size = 0.20, 
    random_state = 12, 
)


#3. Text Classification Pipeline Setup 
# Create classifier with a pipeline, using Naive Bayes as classifier
"""text_clf = Pipeline([(
    'vect', CountVectorizer()), 
    ('tfidf', TfidfTransformer()), 
    ('clf', MultinomialNB()),
])"""

# CountVectorizer() converts text data into a matrix of token counts.
# TfidfTransformer() converts count matrix to TF-IDF representation.
# MultinomialNB() is the Naive Bayes classifier.

# Create classifier with a pipeline, using SVM/SGD
text_clf = Pipeline([
 ('vect', CountVectorizer()),
 ('tfidf', TfidfTransformer()),
 ('clf', SGDClassifier(loss='hinge', penalty='l1',alpha=1e-3, random_state=42, max_iter=50, tol=None)),
])
#4. Training and Testing the Model 
# Trains the text classification model using the training data.
text_clf.fit(movie_data_train, movie_target_train)

# Predicts labels for the test set.
predicted = text_clf.predict(movie_data_test)

# Prints accuracy, classification report, and confusion matrix to evaluate model performance.
print('\n Accuracy: ', np.mean(predicted == movie_target_test))
print(metrics.classification_report(movie_target_test, predicted, target_names = movie.target_names))
print('\nConfusion matrix: ', metrics.confusion_matrix(movie_target_test, predicted))

#5. Grid Search for Parameter Tuning
# Parameter tuning using grid search
# Test different parameters for the grid search
# Defines a grid of parameters for the pipeline components (CountVectorizer, TfidfTransformer, and MultinomialNB).

parameters = {
'vect__ngram_range': [(1, 1), (1, 2),(1,3)], #unigrams, bigrams, trigrams?
'tfidf__use_idf': (True, False),
'clf__alpha': (1e-1,1e-2,1e-3) #(1e-3 i less smoother than (1e-2) and the training data will be more sensitive)
}

##RESULT##

#MultinomialNB
#TEST 1: parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False), 'clf__alpha': (1e-2, 1e-3), cv=5}
#Accuracy: 0.8175
#Best Mean Score: 0.74
#tfidf__use_idf: False
#vect__ngram_range: (1, 2) 

#TEST 2: parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False), 'clf__alpha': (1e-2, 1e-3), cv=10}
#Accuracy: 0.8175
#Best Mean Score: 0.7625
#tfidf__use_idf: False
#vect__ngram_range: (1, 2)

#TEST 3: parameters = {'vect__ngram_range': [(1, 1), (1, 2), (1, 3)], 'tfidf__use_idf': (True, False), 'clf__alpha': (1e-2, 1e-3), cv=10}
#Accuracy: 0.8175
#Best Mean Score: 0.7675
#tfidf__use_idf: False
# vect__ngram_range: (1, 3) #trigrams (kollar 3 ord irad som gav bäst mean score?)

#TEST 4: parameters = {'vect__ngram_range': [(1, 1), (1, 2), (1,3), (1,4)], 'tfidf__use_idf': (True, False), 'clf__alpha': (1e-1), (1e-2), (1e-3), (1e-4) cv=10}
#Accuracy: 0.8175
#Best Mean Score: 0.7675
#tfidf__use_idf: False
#vect__ngram_range: (1, 3)
#samma som förra

#SVM/SDG
#TEST 1: parameters = {('vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False), 'clf__alpha': (1e-1, 1e-2), cv=10 || loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)}
#Accuracy: 0.8325
#Best Mean Score: 0.6725
#tfidf__use_idf: False
#vect__ngram_range: (1, 1)
#ridge 


#TEST 2: parameters = {('vect__ngram_range': [(1, 1), (1, 2), (1,3)], 'tfidf__use_idf': (True, False), 'clf__alpha': (1e-1, 1e-2, 1e-3), cv=10 || loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)}
#Accuracy: 0.8325
#Best Mean Score: 0.7625
#tfidf__use_idf: True 
#vect__ngram_range: (1, 1)
#ridge 

#TEST 3: parameters = {('vect__ngram_range': [(1, 1), (1, 2), (1,3)], 'tfidf__use_idf': (True, False), 'clf__alpha': (1e-1, 1e-2, 1e-3), cv=10 || loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=25, tol=None)}
#Accuracy: 0.8325
#Best Mean Score: 0.7675
#tfidf__use_idf: True 
#vect__ngram_range: (1, 2)
#max_iter=25


#TEST 4: parameters = {('vect__ngram_range': [(1, 1), (1, 2), (1,3)], 'tfidf__use_idf': (True, False), 'clf__alpha': (1e-1, 1e-2, 1e-3), cv=10 || loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)}
#Accuracy: 0.715
#Best Mean Score: 0.72
#tfidf__use_idf: True 
#vect__ngram_range: (1, 2)
#Kör med Lasso (l1)

#TEST 3: parameters = {('vect__ngram_range': [(1, 1), (1, 2), (1,3)], 'tfidf__use_idf': (True, False), 'clf__alpha': (1e-1, 1e-2, 1e-3), cv=10 || loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=25, tol=None)}
#Accuracy: 0.
#Best Mean Score: 0.
#tfidf__use_idf: True 
#vect__ngram_range: (1, 2)
#max_iter=50



# Grid search will detect how many CPU cores we have at our disposal n_jobs = (-1)
# Uses GridSearchCV to search for the best combination of parameters using cross-validation.
gs_clf = GridSearchCV(text_clf, parameters, cv=10, n_jobs=-1) #cv 10 had better score compared to 5

# Fits the grid search on a smaller subset of the training data to speed up computation.
gs_clf = gs_clf.fit(movie_data_train[:400], movie_target_train[:400])


#6. Prediction on New Reviews 
# Create fake reviews
reviews_new = ['This movie was excellent', 'Absolute joy ride', 
            'Steven Seagal was terrible', 'Steven Seagal shone through.', 
            'This was certainly a movie', 'Two thumbs up', 'I fell asleep halfway through', 
            "We can't wait for the sequel!!", '!', '?', 'I cannot recommend this highly enough', 
            'instant classic.', 'Steven Seagal was amazing. His performance was Oscar-worthy.']

# Uses the best model found through grid search to predict the sentiment (positive or negative) of these new reviews.
# Best mean score
print('\n Best mean score: ', gs_clf.best_score_)


# And its parameter settings
for param_name in sorted(parameters.keys()):
  print(" %s: %r" % (param_name, gs_clf.best_params_[param_name]))


#7. Print predictions 
# Prints the predicted sentiment for each review.
print('\n')
# Predict the new reviews
predicted = gs_clf.predict(reviews_new)
# Print the predicted results
for review, category in zip(reviews_new, predicted):
    print('\n %r => %s' % (review, movie.target_names[category]))