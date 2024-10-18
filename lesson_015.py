import pandas as pd
import numpy as np

# Calculate weights
def w(X, Y):
    A = np.linalg.pinv(np.dot(X.T, X))
    B = np.dot(X.T, Y)
    w = np.dot(A, B)
    return w

# Calculate error term
def J(X, Y, w, m):
    A = np.dot(X, w) - Y
    J = (1 / m) * np.dot(A.T, A)
    return J

# Create X and Y arrays
def make_arrays(x_cols, y_col, m, df):
    data_array = df[x_cols].to_numpy()
    ones_array = np.ones([1, m])
    X = np.insert(data_array, 0, ones_array, axis = 1)
    Y = df[y_col].to_numpy()
    return X, Y

df = pd.read_csv('Data/MoreHouseData.csv')

num_rows = len(df)
x_cols = np.array(['Beds', 'Baths', 'Sqft'])
y_col = np.array(['Price'])
X, Y = make_arrays(x_cols, y_col, num_rows, df)
weights = w(X, Y)

print("w0 =", weights[0])
print("w1 =", weights[1])
print("w2 =", weights[2])
print("w3 =", weights[3])

error_term = J(X, Y, weights, num_rows)
print("Error term is:", error_term)