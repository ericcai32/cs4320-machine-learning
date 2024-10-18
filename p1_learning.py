import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from regression import w, J, make_arrays
from validation import make_k_folds

df = pd.read_csv('Data/baby.csv')
num_rows = len(df)

# Create the array of column headers that correspond to x and y values.
x_cols = np.array([
    'Gestational Days',
    'Maternal Age',
    'Maternal Height',
    'Maternal Pregnancy Weight',
    'Maternal Smoker'
])
y_col = np.array(['Birth Weight'])
quad_x_cols = np.array([
    'Gestational Days',
    'Maternal Age',
    'Maternal Height',
    'Maternal Pregnancy Weight',
    'Maternal Smoker',
    'Gestational Days Squared',
    'Maternal Age Squared',
    'Maternal Height Squared',
    'Maternal Pregnancy Weight Squared',
    'Maternal Smoker Squared'
])

# Create the table for a quadratic regression.
gestational_days_sq = df['Gestational Days'] ** 2
maternal_age_sq = df['Maternal Age'] ** 2
maternal_height_sq = df['Maternal Height'] ** 2
maternal_pregnancy_weight_sq = df['Maternal Pregnancy Weight'] ** 2
maternal_smoker_sq = df['Maternal Smoker'] ** 2
df['Gestational Days Squared'] = gestational_days_sq
df['Maternal Age Squared'] = maternal_age_sq
df['Maternal Height Squared'] = maternal_height_sq
df['Maternal Pregnancy Weight Squared'] = maternal_pregnancy_weight_sq
df['Maternal Smoker Squared'] = maternal_smoker_sq

print("\n\n\n")

# Set up the k-fold validation.
i = 0
k = 5
lin_t_Js = []
lin_v_Js = []
quad_t_Js = []
quad_v_Js = []

t_folds, v_folds = make_k_folds(df, k)
num_t_rows = len(t_folds[0])
num_v_rows = len(v_folds[0])
for t_fold, v_fold in zip(t_folds, v_folds):
    # Calculate the weights of the linear regression.
    lin_x, y = make_arrays(t_fold, num_t_rows, x_cols, y_col)
    lin_w = w(lin_x, y)
    
    # Calculate the weights of the quadratic regression.
    quad_x, y = make_arrays(t_fold, num_t_rows, quad_x_cols, y_col)
    quad_w = w(quad_x, y)
    
    # Calculate the J terms of the two regressions on the training set.
    lin_t_J = J(lin_x, y, lin_w, num_t_rows)
    quad_t_J = J(quad_x, y, quad_w, num_t_rows)
    
    # Calculate the J terms for validation sets using the previous weights.
    lin_x, y = make_arrays(v_fold, num_v_rows, x_cols, y_col)
    quad_x, y = make_arrays(v_fold, num_v_rows, quad_x_cols, y_col)
    lin_v_J = J(lin_x, y, lin_w, num_v_rows)
    quad_v_J = J(quad_x, y, quad_w, num_v_rows)
    
    # Append J terms to a list for future plotting.
    lin_t_Js.append(lin_t_J)
    lin_v_Js.append(lin_v_J)
    quad_t_Js.append(quad_t_J)
    quad_v_Js.append(quad_v_J)
    print(lin_t_J)
    print(lin_v_J)
    
    # Print all of the calculated values.
    print(f"FOLD {i}")
    i += 1
    print("---")
    print("Training Set J Terms")
    print(f"Linear regression: {round(lin_t_J[0][0], 3)}")
    print(f"Quadratic Regression {round(quad_t_J[0][0], 3)}")
    print("---")
    print("Validation Set J Terms")
    print(f"Linear regression: {round(lin_v_J[0][0], 3)}")
    print(f"Quadratic regression {round(quad_v_J[0][0], 3)}")
    print("\n")

print("\n\n\n")

# Plot all of the calculated values.
plt.scatter(lin_t_Js, lin_v_Js, c='blue', label="Linear")
plt.scatter(quad_t_Js, quad_v_Js, c='red', label="Quadratic")
plt.title("Validation J Terms vs. Training J Terms")
plt.xlabel("Training J Terms")
plt.ylabel("Validation J Terms")
plt.legend()

lin_x, y = make_arrays(df, num_rows, x_cols, y_col)
lin_w = w(lin_x, y)
print(lin_w)