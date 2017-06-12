import sys
sys.path.append('..')
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize

raw_csv_path = 'movie_metadata.csv'

use_cols = ['num_voted_users', 'duration', 'cast_total_facebook_likes', 'budget', 'gross', 'imdb_score']


df = pd.read_csv(raw_csv_path, usecols=use_cols)
df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

X_cols = [col for col in df.columns.values if not col == 'imdb_score']
X_df = df[X_cols]
X_df = pd.get_dummies(X_df)
y_series = df['imdb_score']

X = X_df.as_matrix()
y = y_series.as_matrix().astype(int).astype(str)

y_ranges = {
    '1':'1-3',
    '2':'1-3',
    '3':'1-3',
    '4':'4-6',
    '5':'4-6',
    '6':'4-6',
    '7':'7-9',
    '8':'7-9',
    '9':'7-9',
    '10':'10'
}

for i in range(0, len(y)):
    y[i] = y_ranges[y[i]]
