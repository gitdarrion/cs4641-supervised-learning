from preprocess import preprocess
from read import X_df, y_df
from sklearn import tree
from sklearn.metrics import accuracy_score
import time

print '\nPredicting...'
start = time.time()

X_train, X_test, y_train, y_test = preprocess(X_df, y_df, 0.1)
classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(X_train, y_train)
prediction = classifier.predict(X_test)
accuracy = accuracy_score(y_test, prediction)

print 'Length of X training: ', len(X_train)
print 'Length of Y training: ', len(y_train)

#print 'Prediction: ', prediction
print 'Accuracy: ', accuracy

with open('imdb.dot', 'w') as f:
    f = tree.export_graphviz(classifier, f)

end = time.time()
run_time = end-start
print 'Finished in ', run_time, ' seconds.'
