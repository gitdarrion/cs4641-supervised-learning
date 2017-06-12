"""
Author: Scikit-Learn
Modified by: Darrion

"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    print train_scores_mean[-1]
    print test_scores_mean[-1]

    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


from data import X, y
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.ensemble import GradientBoostingClassifier


def nn(n_splits=100, test_size=0.2, random_state=0, activation='logistic', hidden_layer_sizes=(10,46), solver='sgd'):
    title = "Learning Curves: Neural Network Classifier"
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    estimator = MLPClassifier(activation=activation, hidden_layer_sizes=hidden_layer_sizes, solver=solver)
    plot_learning_curve(estimator, title, X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=1)

def svm(n_splits=10, test_size=0.2, random_state=0, kernel='rbf'):
    title = "Learning Curves: SVM Classifier"
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    estimator = SVC(kernel=kernel)
    plot_learning_curve(estimator, title, X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=1)

def knn(neighbors=[1,2,3,4,5,6,7,8,9,10], n_splits=10, test_size=0.2, cv_random_state=0):
    for k in neighbors:
        print 'For k=', k
        title = "Learning Curves: K=%s Nearest Neighbors Classifier" % k
        cv = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=cv_random_state)
        estimator = KNeighborsClassifier(n_neighbors=k)
        plot_learning_curve(estimator, title, X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=1)

def dt(n_splits=10, test_size=0.2, cv_random_state=0):
    title = "Learning Curves: Decision Tree Classifier"
    cv = cv = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=cv_random_state)
    estimator = DecisionTreeClassifier()
    plot_learning_curve(estimator, title, X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=1)

def gb(n_splits=10, test_size=0.2, cv_random_state=0, estimator_random_state=0, learning_rate=0.1, max_depth=1):
    title = "Learning Curves: Gradient Boosted Decision Tree Classifier"
    cv = cv = ShuffleSplit(n_splits=n_splits, test_size=test_size, cv_random_state=random_state)
    estimator = GradientBoostingClassifier(learning_rate=learning_rate, max_depth=max_depth, random_state=estimator_random_state)
    plot_learning_curve(estimator, title, X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=1)

import sys
if len(sys.argv) == 2:
    arg = sys.argv[1]
    if arg == 'nn':
        nn()
    elif arg == 'svm':
        svm()
    elif arg == 'knn':
        knn()
    elif arg == 'dt':
        dt()
    elif arg == 'gb':
        gb()
else:
    print 'Invalid inputs.'

plt.show()
