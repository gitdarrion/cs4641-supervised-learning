import time
import pandas as pd
import numpy as np

start = time.time()
print '\nCleaning...'

path_to_csv = 'movie_metadata.csv'

#use_cols = ['duration', 'director_facebook_likes', 'actor_1_facebook_likes', 'actor_2_facebook_likes', 'actor_3_facebook_likes', 'content_rating', 'country', 'budget', 'gross', 'num_voted_users', 'cast_total_facebook_likes', 'num_user_for_reviews', 'num_critic_for_reviews', 'imdb_score']
use_cols = ['duration', 'director_facebook_likes', 'budget', 'gross', 'num_voted_users', 'cast_total_facebook_likes', 'num_user_for_reviews', 'country', 'imdb_score']

df = pd.read_csv(path_to_csv, usecols=use_cols)
df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()
df = df[df['country']=='USA']

X_cols = [col for col in df.columns.values if not col == 'imdb_score']
X_df = df[X_cols]
X_df = pd.get_dummies(X_df)

print 'Training on features:'
for feature_num, col in enumerate(X_df.columns.values):
    print feature_num, ' ', col

y_df = df['imdb_score']
