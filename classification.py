import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

def kNN_classify(train_df, val_df, x_col, y_col, cat_col, cats, k):
    """
    Return a dataframe with a 'cats_by_kNN' column with categories based on 
    k-Nearest Neighbor.
    """
    train_x = train_df[x_col].to_numpy()
    train_y = train_df[y_col].to_numpy()
    val_x = val_df[x_col].to_numpy()
    val_y = val_df[y_col].to_numpy()
    cats_by_kNN = np.array([])
    
    # Calculate distances.
    for x, y in zip(val_x, val_y):
        distances = ((x - train_x) ** 2 + (y - train_y) ** 2) ** 0.5
    
        # Add distances to a copy of the table and sort.
        temp_df = train_df.copy()
        temp_df['distances'] = distances
        temp_df = temp_df.sort_values('distances').reset_index()
        
        # Count the classes for the k-nearest neighbors.
        cat_counts = []
        for cat in cats:
            count = 0
            for i in range(k):
                if temp_df.loc[i, cat_col] == cat:
                    count += 1
            cat_counts.append(count)
        
        # Find the categories with the most matches, and add them to an array.
        cat_by_kNN = cats[cat_counts.index(max(cat_counts))]
        cats_by_kNN = np.append(cats_by_kNN, cat_by_kNN)
    
    # Add the cats_by_kNN array to the dataframe.
    val_df['cats_by_kNN'] = cats_by_kNN
    
    return val_df

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