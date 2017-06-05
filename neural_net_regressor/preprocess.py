import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

start = time.time()
print '\nPreprocessing...'

def preprocess(X_df=pd.DataFrame, y_df=pd.Series, test_size=float):
    X = X_df.as_matrix()
    y = y_df.as_matrix()

    return train_test_split(X, y, test_size=test_size)
