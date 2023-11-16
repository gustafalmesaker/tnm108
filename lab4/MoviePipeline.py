from matplotlib.pyplot import clf
import sklearn
import nltk
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix


# nltk.download('punkt')

# Load movie_reviews corpus data through sklearn
moviedir = r'lab4\movie_reviews'

# loading all files.
movie = load_files(moviedir, shuffle=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(movie.data, movie.target, test_size=0.2, random_state=42)

# Building a pipeline
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])

# Define the parameters you want to search through
parameters = {
    'vect__ngram_range': [(1, 1), (1, 2), (1, 3), (1,4)],  # unigrams or bigrams
    'tfidf__use_idf': (True, False),
    'clf__alpha': (1e-2, 1e-3),
}




# Create SVM classifier with a pipeline, using SGD as classifier
# text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', SGDClassifier(loss = 'hinge', penalty = 'l2', alpha=1e-3, random_state=42, max_iter=5, tol=None), )])
# hinge gives a linear SVM, l2 is the regularisation parameter, alpha is the learning rate, max_iter is the number of iterations, tol is the stopping criterion

# Create GridSearchCV object
grid_search = GridSearchCV(text_clf, parameters, cv=5, scoring='accuracy')

# Fit the model to the data
grid_search.fit(X_train, y_train)


# Get the best parameters
best_params = grid_search.best_params_

# Train the final model with the best parameters
final_model = Pipeline([
    ('vect', CountVectorizer(ngram_range=best_params['vect__ngram_range'])),
    ('tfidf', TfidfTransformer(use_idf=best_params['tfidf__use_idf'])),
    ('clf', MultinomialNB(alpha=best_params['clf__alpha'])),
])

"""
final_model = Pipeline([
('vect', CountVectorizer()),
('tfidf', TfidfTransformer()),
('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42
,max_iter=5, tol=None)),
])"""


#final_model.fit(X_train, y_train)
grid_search.fit(X_train, y_train)

# Evaluate on the test set
#test_accuracy = final_model.score(X_test, y_test)
test_accuracy = grid_search.score(X_test, y_test)


#gs_clf = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1) #iid = false removed

#gs_clf = gs_clf.fit(.data[:400], twenty_train.target[:400])

# Create a confusion matrix
conf_matrix = confusion_matrix(y_test,X_test)

# Print the confusion matrix
print("Confusion Matrix:")
print(conf_matrix)


print("Best Parameters:", best_params)
print("Test Accuracy:", test_accuracy)

print("Grid search best score: ",grid_search.best_score_)
print("Grid search best estimator: ",grid_search.best_estimator_)


for param_name in sorted(parameters.keys()):
 print("%s: %r" % (param_name, grid_search.best_params_[param_name]))