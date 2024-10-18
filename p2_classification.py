import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from validation import make_k_folds
from classification import kNN_classify

df = pd.read_csv('Data/P2train.csv')
test_df = pd.read_csv('Data/P2test.csv')

error_sums = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
error_percentages = []

def check_incorrect(df):
    """
    Count how many items in the 'Pass' column are equal to its corresponding 
    item in the 'cat_by_kNN' column.
    """
    pass_cats = df['Pass'].to_numpy()
    new_cats = df['cats_by_kNN'].to_numpy()
    num_wrong = sum(np.not_equal(pass_cats, new_cats))
    return num_wrong

t_folds, v_folds = make_k_folds(df, 5)
for i, (t_fold, v_fold) in enumerate(zip(t_folds, v_folds)):
    print(f"\n\nFOLD {i+1}")
    print("------")
    for n, k in enumerate(range(1, 22, 2)):
        to_check = kNN_classify(t_fold, v_fold, 'Test1', 'Test2', 'Pass', [0, 1], k)
        num_wrong = check_incorrect(to_check)
        print(num_wrong)
        error_sums[n] += num_wrong

print("\n\n\n")

error_sums = [37, 24, 27, 27, 33, 36, 37, 37, 44, 41, 40]
# Assign percentages to a list for plotting.
for error_sum in error_sums:
    error_percentages.append(1 - (error_sum / 85))



# Make plot of Accuracy vs. k
plt.plot(range(1, 22, 2), error_percentages)
plt.xlabel("Value of k for kNN")
plt.ylabel("Cross-Validated Accuracy")
plt.xticks(range(1, 22, 2))


# Check the test set and print the values for a confusion matrix.
k = 3
tp = 0
ff = 0
fp = 0
tf = 0

whole = kNN_classify(df, test_df, 'Test1', 'Test2', 'Pass', [0, 1], k)
true_cats = whole['Pass'].to_numpy()
kNN_cats = whole['cats_by_kNN'].to_numpy()
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

print("tf:", tf, "fp:", fp)
print("ff:", ff, "tp:", tp)

full_df = pd.concat([df, test_df])

"""
colors = []
symbols = []
for i in full_df['Pass']:
    if i == 1:
        colors.append('green')
        symbols.append('o')
    else:
        colors.append('red')
        symbols.append('x')

plt.xlabel("Test 1")
plt.ylabel("Test 2")
full_df = full_df.reset_index()
for i in range(len(full_df['Test1'])):
    print(full_df['Test1'][i])
    print("...")
    plt.scatter(full_df['Test1'][i], full_df['Test2'][i], c=colors[i], marker=symbols[i])
"""