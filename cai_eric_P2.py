import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def make_k_folds(df, k):
    """Split a dataframe into k equal and random folds."""
    num_rows = int(len(df) / k)
    original_df = df
    training_folds = []
    validation_folds = []
    
    for i in range(k):
        validation_fold = df.sample(n=num_rows)
        df = df.drop(validation_fold.index)
        temp_df = original_df.drop(validation_fold.index)
        training_folds.append(temp_df)
        validation_folds.append(validation_fold)
        
    return training_folds, validation_folds

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

def check_incorrect(df):
    """
    Count how many items in the 'Pass' column are equal to its corresponding 
    item in the 'cat_by_kNN' column.
    """
    pass_cats = df['Pass'].to_numpy()
    new_cats = df['cats_by_kNN'].to_numpy()
    num_wrong = sum(np.not_equal(pass_cats, new_cats))
    return num_wrong

training_file = input("What is the name of the training file? ")
test_file = input("What is the name of the test file? ")

df = pd.read_csv(training_file)
test_df = pd.read_csv(test_file)

# Check the test set and print the values for a confusion matrix.
k = 3
tp = 0
ff = 0
fp = 0
tf = 0

classified = kNN_classify(df, test_df, 'Test1', 'Test2', 'Pass', [0, 1], k)
true_cats = classified['Pass'].to_numpy()
kNN_cats = classified['cats_by_kNN'].to_numpy()
for true_cat, kNN_cat in zip(true_cats, kNN_cats):
    if true_cat == 1:
        if kNN_cat == 1:
            tp += 1
        else:
            ff += 1
    else:
        if kNN_cat == 1:
            fp += 1
        else:
            tf += 1

print("TN:", tf, "FP:", fp)
print("FN:", ff, "TP:", tp)
accuracy = (tf + tp) / (tf + fp + ff + tp)
precision = tp / (tp + fp)
recall = tp / (tp + ff)
f1 = 2 * (precision * recall) / (precision + recall)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)
