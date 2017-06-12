import sys
sys.path.append('..')
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize

path_to_csv = 'netflix_shows.csv'

df = pd.read_csv(path_to_csv)
df.replace([np.inf, -np.inf], 0)
df = df.dropna()

X_cols = [col for col in df.columns.values if not col == 'user rating score']
X_df = df[X_cols]
X_df = pd.get_dummies(X_df)

print 'Training on features:'
for feature_num, col in enumerate(X_df.columns.values):
    print feature_num, ' ', col

y_df = df['user rating score']

X = X_df.as_matrix()
y = y_df.as_matrix().astype(int).astype(str)
