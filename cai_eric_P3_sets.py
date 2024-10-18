import pandas as pd

path = input("Enter the path to the banknote_data.csv file. ")

df = pd.read_csv(path)

# Split the data into a training set of 80% and a test set of 20%.
training_df = df.sample(frac=0.8, random_state=7)
test_df = df.drop(training_df.index)

# Convert the dataframes to a csv file.
training_df.to_csv('banknote_train.csv')
test_df.to_csv('banknote_test.csv')