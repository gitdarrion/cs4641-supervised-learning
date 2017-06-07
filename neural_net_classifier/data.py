import sys
sys.path.append('..')
import time
import pandas as pd
import numpy as np

path_to_csv = 'clean_movie_metadata.csv'

df = pd.read_csv(path_to_csv)
df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

X_cols = [col for col in df.columns.values if not col == 'imdb_score']
X_df = df[X_cols]
X_df = pd.get_dummies(X_df)

print 'Training on features:'
for feature_num, col in enumerate(X_df.columns.values):
    print feature_num, ' ', col

y_df = df['imdb_score']
