import pandas as pd
import numpy as np

def make_arrays(df, m, x_cols, y_col):
    """Return two arrays derived from a table to set up a regression."""
    data_array = df[x_cols].to_numpy()
    ones_array = np.ones([1, m])
    x = np.insert(data_array, 0, ones_array, axis = 1)
    y = df[y_col].to_numpy()
    return x, y

def w(x, y):
    """Return the weights for a regression."""
    dp1 = np.linalg.pinv(np.dot(x.T, x))
    dp2 = np.dot(x.T, y)
    w = np.dot(dp1, dp2)
    return w

def J(x, y, w, m):
    """Return the error term of a regression based on the given weights"""
    dp1 = np.dot(x, w) - y
    J = (1/m) * np.dot(dp1.T, dp1)
    return J