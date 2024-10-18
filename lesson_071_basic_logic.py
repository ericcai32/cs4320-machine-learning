import pandas as pd

from regression import make_arrays, w

and_df = pd.read_csv('Data/AND_data.csv')
or_df = pd.read_csv('Data/OR_data.csv')
bi_df = pd.read_csv('Data/BI_data.csv')
xor_df = pd.read_csv('Data/XOR_data.csv')

m = len(and_df)
x, y = make_arrays(and_df, m, ['x1', 'x2'], 'AND')
weights = w(x, y)
print("AND Weights")
print(weights)

m = len(or_df)
x, y = make_arrays(or_df, m, ['x1', 'x2'], 'OR')
weights = w(x, y)
print("\nOR Weights")
print(weights)

m = len(bi_df)
x, y = make_arrays(bi_df, m, ['x1', 'x2'], 'BI')
weights = w(x, y)
print("\nBI Weights")
print(weights)

m = len(xor_df)
x, y = make_arrays(xor_df, m, ['x1', 'x2'], 'XOR')
weights = w(x, y)
print("\nXOR Weights")
print(weights)