from sklearn.neighbors import NearestNeighbors
from read import X_df
import numpy as np

X = X_df.as_matrix()
classifer = NearestNeighbors(n_neighbors=3)
classifier = classifier.fit(X)
distances, indices = classifier.kneighbors(X)

print 'Distances:\n', distances
print 'Indices:\n', indices
