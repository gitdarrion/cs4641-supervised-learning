from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from preprocess import preprocess
from data import X_df, y_df


X_train, X_test, y_train, y_test = preprocess(X_df, y_df, 0.1)
estimator = GradientBoostingClassifier(learning_rate=0.1, max_depth=1, random_state=0)
estimator.fit(X_train, y_train)
y_pred = estimator.predict(X_test)
print(classification_report(y_test, y_pred))
