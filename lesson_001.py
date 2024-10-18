import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv('Data/iris_data.csv')
# print("shape of the DataFrame is:", data.shape)
# print("")
# print(data.head())
# print("")
# print(data.pivot_table(index='species',
#       aggfunc='count'))

# Create a table of each type of iris
Setosa = data[data.species=='setosa']
Versicolor = data[data.species=='versicolor']
Virginica = data[data.species=='virginica']

# Create scatter plots of sepal length vs petal length
# Convert columns in a dataframe to arrays
setosaSL = Setosa['sepal_length'].to_numpy()
setosaPL = Setosa['petal_length'].to_numpy()

# Create the scatter plot
# Plot by turnings columns into arrays
# plt.scatter(setosaSL, setosaPL, marker='v', c='red', label="Setosa")

# Plot by pulling the columns directly
plt.scatter(Setosa['sepal_length'], Setosa['petal_length'], marker='v',
            c='red', label="Setosa")
plt.scatter(Versicolor['sepal_length'], Versicolor['petal_length'],
            marker='x', c='green', label="Versicolor")
plt.scatter(Virginica['sepal_length'], Virginica['petal_length'],
            marker='o', c='blue', label="Virginica")

# Add plot labels
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.title("Sepal vs Petal Length")
plt.legend(loc='lower right')

# Save the plot
plt.savefig('flowers.png')