import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

start = time.time()
print '\nPreprocessing...'

def preprocess(X_df=pd.DataFrame, y_df=pd.Series, test_size=float):
    X = X_df.as_matrix()
    y = y_df.as_matrix().astype(int).astype(str)
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

    return train_test_split(X, y, test_size=test_size)
