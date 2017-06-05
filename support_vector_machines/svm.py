from data import X_df, y_df
from preprocess import preprocess
from sklearn.metrics import accuracy_score
from sklearn import svm
import time

start = time.time()

X_train, X_test, y_train, y_test = preprocess(X_df, y_df, 0.1)
classifier = svm.SVC()
classifier = classifier.fit(X_train, y_train)
prediction = classifier.predict(X_test)
accuracy = accuracy_score(y_test, prediction)

print 'Accuracy: ', accuracy

end = time.time()
run_time = end-start
print 'SVM Classification finished in ', run_time, ' seconds.'
