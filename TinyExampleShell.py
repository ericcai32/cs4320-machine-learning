# Example Linear Regression with one variable

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Hypothesis function
def h(w0, w1, single_x):
    return w0 + w1 * single_x

# Error (J) function
def J(w0, w1, m, X, Y):
    return 1/(2*m) * np.sum((h(w0, w1, X) - Y) ** 2)

#Function to update  w0 and w1
def update_ws(w0, w1, alpha, m, X, Y):
    new_w0 = w0 - (alpha * (1/m) * np.sum(w0 + w1*X - Y))
    new_w1 = w1 - (alpha * (1/m) * np.sum((w0 + w1*X - Y) * X))
    return new_w0, new_w1

#Use Tiny Example Data from the notes
df = pd.read_csv("Data/olympic_100_meters_2004.csv")
print(df.head())
m = len(df.index)
X = df['Year'].to_numpy()
Y = df['Men'].to_numpy()

#Choose some intial values
alpha = 0.0000001
w0 = 40
w1 = 10
loops = 30

print ("Initially w0 =", w0, "w1 = ", w1, "J =", J(w0,w1, m, X, Y))
print()
for k in range(loops):
    w0, w1 = update_ws(w0, w1, alpha, m, X, Y)
    the_error = J(w0, w1, m, X, Y)
    if (k > 0):  #Determine what to plot
        plt.scatter(k, the_error)
        plt.xlabel("Number of iterations")
        plt.ylabel("Error (J)")

print("After", loops, "iterations w0 = ", w0, "w1 =", w1, "J=", the_error)