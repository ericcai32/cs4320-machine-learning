import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

def calculate_hypotheses(X, w):
    hypotheses = 1 / (1 + math.e ** (-1 * np.dot(X, w)))
    return hypotheses

def calculate_costs(H, y):
    costs = -1 * (y * np.log(H)) - (1 - y) * np.log(1 - H)
    return costs

def sum_costs(costs, m):
    O1xm = np.array([np.ones(m)])
    costs_sum = (1/m) * np.dot(O1xm, costs)
    return costs_sum

def calculate_new_weights(X, y, w, H, m, alpha):
    new_weights = w - ((alpha/m) * np.dot((H - y).T, X).T)
    return new_weights

def logistic_regression(df, x_columns, y_column, alpha, iterations):
    m = len(df)
    
    feature_list = [np.ones(m)]
    for x_column in x_columns:
        feature_array = df[x_column].to_numpy()
        feature_list.append(feature_array)
    X = np.array(feature_list).T
    y = np.array([df[y_column].to_numpy()]).T
    w = np.array([np.zeros(len(x_columns) + 1)]).T
    
    for i in range(iterations):
        H = calculate_hypotheses(X, w)
        cost = calculate_costs(H, y)
        J = sum_costs(cost, m)
        
        w = calculate_new_weights(X, y, w, H, m, alpha)
        if i > 0.8 * iterations:
            plt.scatter(i, J)
    return w, J

path = input("Enter the path to the training file: ")
df = pd.read_csv(path)

# Do gradient descent on logistic regression.
final_weights, final_J = logistic_regression(
                             df,
                             ['Variance', 'Skewness', 'Curtosis', 'Entropy'],
                             'Genuine=1',
                             1.3,
                             77
                         )

# Print the final weights and write each to a text file.
for i, weight in enumerate(final_weights):
    print(f"w{i}:", *weight)
print("J:", *final_J[0])

np.savetxt('weights.txt', final_weights)