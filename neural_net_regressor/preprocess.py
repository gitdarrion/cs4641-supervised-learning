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

def discretize(array=np.array, ranges=dict):
    return None

if __name__ == '__main__':
    from read import X_df, y_df
    X_train, X_test, y_train, y_test = preprocess(X_df, y_df, 0.33)
    end = time.time()
    run_time = end - start
    print 'Finished in', run_time, 'seconds'
else:
    end = time.time()
    run_time = end - start
    print 'Finished in', run_time, 'seconds'
