from preprocess import preprocess
from data import X_df, y_df
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import numpy as np
import time

""" Convert to matrices. """
X = X_df.as_matrix()
y = y_df.as_matrix().astype(int).astype(str)

""" Convert continous values to labels. """
y_ranges = {
    '1':'1-3',
    '2':'1-3',
    '3':'1-3',
    '4':'4-6',
    '5':'4-6',
    '6':'4-6',
    '7':'7-9',
    '8':'7-9',
    '9':'7-9',
    '10':'10'
}

for i in range(0, len(y)):
    y[i] = y_ranges[y[i]]

kf = KFold(n_splits=10)
accuracies = []
for train_indices, test_indices in kf.split(X):
    # Partition
    X_train = [X[i] for i in train_indices]
    X_test = [X[i] for i in test_indices]
    y_train = [y[i] for i in train_indices]
    y_test = [y[i] for i in test_indices]

    # Classification
    classifier = MLPClassifier(activation='logistic', hidden_layer_sizes=(10000,), solver='sgd')
    classifier = classifier.fit(X_train, y_train)
    prediction = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, prediction)
    accuracies = accuracies + [accuracy]
print '\nNumber of folds: ', len(accuracies)
print '\nAccuracies:\n', accuracies
print '\nAverage Accuracy: ', np.mean(accuracies)
