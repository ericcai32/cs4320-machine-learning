import pandas as pd

times = pd.read_csv('Data/olympic_100_meters_2004.csv')

x = times['Year']
y = times['Men']
xy = x * y
x2 = x ** 2

w1 = (x.mean() * y.mean() - xy.mean()) / (x.mean() ** 2 - x2.mean())
w0 = y.mean() - w1 * x.mean()

given = int(input("What year do y want to predict? "))

h_func = w0 + w1 * given
h_func = round(h_func, 2)

print(f"Predicted winning time for the {given} men's 100m dash: {h_func} seconds.")