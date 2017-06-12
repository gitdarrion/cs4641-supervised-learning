import sys
sys.path.append('..')
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize

csv_path = 'clean_movie_metadata.csv'

use_cols = ['num_voted_users', 'duration', 'cast_total_facebook_likes', 'budget', 'gross', 'imdb_score']

df = pd.read_csv(csv_path, usecols=use_cols)
df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

X_cols = [col for col in df.columns.values if not col == 'imdb_score']
X_df = df[X_cols]
#X_df = pd.get_dummies(X_df)
y_series = df['imdb_score']

print 'Training on features:'
for feature_num, col in enumerate(X_df.columns.values):
    print feature_num, ' ', col

X = X_df.as_matrix()
y = y_series.as_matrix()

print len(X), ' instances'
