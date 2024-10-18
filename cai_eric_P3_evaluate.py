import numpy as np
import pandas as pd
import math

def make_arrays(df, m, x_cols, y_col):
    """Return two arrays derived from a table to set up a regression."""
    data_array = df[x_cols].to_numpy()
    ones_array = np.ones([1, m])
    x = np.insert(data_array, 0, ones_array, axis = 1)
    y = df[y_col].to_numpy()
    return x, y

def predict_categories(x, y, w):
    hypotheses = 1 / (1 + math.e ** (-1 * np.dot(x, w)))
    predictions = hypotheses > 0.5
    predictions = predictions.astype(int)
    return predictions

def check_predictions(actual_categories, predicted_categories):
    tn, fp, fn, tp = 0, 0, 0, 0
    for prediction, actual in zip(predicted_categories.flatten(), actual_categories):
        if actual == 0:
            if prediction == 0:
                tn += 1
            else:
                fp += 1
        else:
            if prediction == 0:
                fn += 1
            else:
                tp += 1
    return tn, fp, fn, tp

# Read the test file and the weights text file.
df_path = input("Enter the path to the test file: ")
weights_path = input("Enter the path to the weights file: ")
df = pd.read_csv(df_path)
w = np.array([np.loadtxt(weights_path)]).T

# Check the weights based on the test file.
m = len(df)
x, y = make_arrays(df, m, ['Variance', 'Skewness', 'Curtosis', 'Entropy'], 'Genuine=1')
predicted_categories = predict_categories(x, y, w)
tn, fp, fn, tp = check_predictions(y, predicted_categories)

# Calculate and print evaluation values.
accuracy = (tn + tp) / (tn + fp + fn + tp)
precision = tp / (fp + tp)
recall = tp / (fn + tp)
f1_score = 2 * (precision * recall) / (precision + recall)

print()
print(f"True Positives: {tp}")
print(f"False Positives: {fp}")
print(f"True Negatives: {tn}")
print(f"False Negatives: {fn}")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1_score}")