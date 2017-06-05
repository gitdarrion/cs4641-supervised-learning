from preprocess import preprocess
from data import X_df, y_df
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import time

print '\nPredicting...'
start = time.time()

X_train, X_test, y_train, y_test = preprocess(X_df, y_df, 0.1)
classifier = MLPClassifier(activation='logistic', solver='sgd') # Defaults to 100 layers of length 98
classifier = classifier.fit(X_train, y_train)
prediction = classifier.predict(X_test)
accuracy = accuracy_score(y_test, prediction)

print 'Length of X training: ', len(X_train)
print 'Length of Y training: ', len(y_train)

#print 'Prediction: ', prediction
print 'Accuracy: ', accuracy

end = time.time()
run_time = end-start
print 'Finished in ', run_time, ' seconds.'
