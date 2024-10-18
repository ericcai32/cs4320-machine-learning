import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

test_sepal_length = float(input("Enter a sepal length: "))
test_sepal_width = float(input("Enter a sepal width: "))
k = 3

df = pd.read_csv('Data/iris_data.csv')

# Remove setosa, petal_length, and petal_wdith.
newdf = df.drop(columns = ['petal_length', 'petal_width'])
df100 = newdf[newdf.species != 'setosa']

# Create two tables for coloring.
Versicolor = df100[df100.species == 'versicolor']
Virginica = df100[df100.species == 'virginica']

# Create scatter plot.
plt.scatter(
    Versicolor['sepal_length'], Versicolor['sepal_width'], marker='x', 
    c='green', label="Versicolor")
plt.scatter(
    Virginica['sepal_length'], Virginica['sepal_width'], marker='o', c='blue',
    label="Virginica")
plt.scatter(test_sepal_length, test_sepal_width, c='red')
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend()

# Calculate distances.
y_dist = (df100['sepal_width'].to_numpy() - test_sepal_width) ** 2
x_dist = (df100['sepal_length'].to_numpy() - test_sepal_length) ** 2
dist_array = (x_dist + y_dist) ** 0.5

# Create table for sorting.
df100['distance'] = dist_array
df100 = df100.sort_values('distance')
df100.reset_index(inplace = True)
print(df100.head())

num_virginica = 0
num_versicolor = 0

for j in range(k):
    if df100.loc[j].at['species'] == "versicolor":
        num_versicolor += 1
    else:
        num_virginica += 1
        
print(f"num_verginica = {num_virginica} num_versicolor = {num_versicolor}")

# Pick a winner
if num_versicolor > num_virginica:
    print("This is a Versicolor iris.")
else:
    print("This is a Virginica iris.")
print("\n\n\n")