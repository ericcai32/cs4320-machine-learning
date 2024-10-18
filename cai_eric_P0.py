import matplotlib.pyplot as plt
import pandas as pd

make_plot = True

file_path = input("Enter the path to the data file: ")
data = pd.read_csv(file_path)

features = ("sepal_length", "sepal_width", "petal_length", "petal_width")

Setosa = data[data.species=='setosa']
Versicolor = data[data.species=='versicolor']
Virginica = data[data.species=='virginica']

print("\nCreate a plot for two features of your data set.")
print("\nThe feature codes are...")
print("0 = sepal length\n1 = sepal width\n2 = petal length\n3 = petal width")

def ask_code(axis):
    code = input(f"\nEnter the feature code for your {axis}-axis: ")
    if code not in ["0", "1", "2", "3"]:
        print("Invalid code.")
        return int(ask_code(axis))
    else:
        return int(code)
    
def ask_end():
    another = input("\nWould you like to do another plot? (y/n) ")
    if another == "n":
        return False
    elif another == "y":
        return True
    else:
        print("Answer with y or n.")
        return ask_end()

while make_plot:
    x_code = ask_code('x')
    y_code = ask_code('y')
    
    plt.scatter(Setosa[features[x_code]], Setosa[features[y_code]], marker='v',
                c='red', label="Setosa")
    plt.scatter(Versicolor[features[x_code]], Versicolor[features[y_code]],
                marker='x', c='green', label="Versicolor")
    plt.scatter(Virginica[features[x_code]], Virginica[features[y_code]],
                marker='o', c='blue', label="Virginica")
    
    x_label = features[x_code].replace("_", " ").title()
    y_label = features[y_code].replace("_", " ").title()
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(x_label + " vs " + y_label)
    plt.legend()
    
    plt.show()
    
    make_plot = ask_end()