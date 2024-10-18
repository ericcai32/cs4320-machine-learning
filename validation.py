import pandas as pd

def make_k_folds(df, k):
    """Split a dataframe into k equal and random folds."""
    num_rows = int(len(df) / k)
    original_df = df
    training_folds = []
    validation_folds = []
    
    for i in range(k):
        validation_fold = df.sample(n=num_rows)
        # df = df.drop(validation_fold.index)
        temp_df = original_df.drop(validation_fold.index)
        training_folds.append(temp_df)
        validation_folds.append(validation_fold)
        
    return training_folds, validation_folds