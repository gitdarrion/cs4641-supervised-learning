from sklearn.neighbors import KNeighborsClassifier
from data import X_df, y_df
from preprocess import preprocess
import numpy as np

print '\nClassifying...\n'

X_train, X_test, y_train, y_test = preprocess(X_df, y_df, 0.1)
classifier = KNeighborsClassifier()
classifier = classifier.fit(X_train, y_train)
print 'Score: ', classifier.score(X_test, y_test)
