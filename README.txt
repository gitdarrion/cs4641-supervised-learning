The name of each directory refers to the dataset on which the code within executes.
To execute the code:
	1. CD into one of the two directories.
	2. Run "python main.py <option>" (without the braces). *
		options:
			'nn': neural network
			'svm': support vector machines
			'knn': k-nearest neighbors
			'dt': decision tree (unpruned) **
			'gb': gradient-boosted decision tree (unpruned) **
The script will execute the algorithms on the .csv file within the directory.
The script will output features, number of features, training score, cross-validation score, and execution times.
When the script terminates, it will display a matplotlib graph plotting the training and cross-validation scores over increasing examples.

* To adjust parameters of the algorithms, do so in the main.py file.
** The pruned trees were created in the GUI of WEKA.
