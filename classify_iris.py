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
        plt.scatter(i, J)
    return w, J
    
df = pd.read_csv('Data/iris_data.csv')

# Remove unneeded columns and rows.
df = df.drop(['petal_width', 'sepal_width'], axis=1)
df = df.loc[df['species'].isin(['versicolor', 'virginica'])]

# Create a new column that is equal to 0 if the speices is versicolor and 1 if
# the species is virginica.
species_nums = df['species'] == 'virginica'
df['species_num'] = species_nums.astype(int)

# Split the data into a training set of 80% and a test set of 20%.
training_df = df.sample(frac=0.8, random_state=0)
test_df = df.drop(training_df.index)

# Convert the dataframes to a csv file.
training_df.to_csv('Data/iris_train.csv')
test_df.to_csv('Data/iris_test.csv')

final_weights, final_J = logistic_regression(
    training_df,
    ['sepal_length', 'petal_length'],
    'species_num',
    0.1,
    50000
)

print("Final weights:")
print("w0 =", *final_weights[0])
print("w1 =", *final_weights[1])
print("w2 =", *final_weights[2])
print("Final J:", *final_J[0])