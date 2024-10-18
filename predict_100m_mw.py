import pandas as pd
import matplotlib.pyplot as plt

times = pd.read_csv('Data/olympic_100_meters_2004.csv')

while True:
    mw = input("Do you want to the scores for men or women? ").lower()
    if mw == 'men':
        y = times['Men']
        break
    elif mw == 'women':
        times = times[times['Women'] > 0]
        y = times['Women']
        break
    else:
        print("Invalid answer.\n")
x = times['Year']
xy = x * y
x2 = x ** 2

w1 = (x.mean() * y.mean() - xy.mean()) / (x.mean() ** 2 - x2.mean())
w0 = y.mean() - w1 * x.mean()

plt.scatter(x, y)
plt.title(f"Winning {mw.title()}'s Times vs Year For the Olympic 100m Dash")
plt.xlabel("Year")
plt.ylabel(f"Winning {mw.title()}'s Times")
plt.show()

while True:
    given = int(input("What year do you want to predict? "))
    
    h_func = w0 + w1 * given
    h_func = round(h_func, 2)
    
    print(f"Predicted winning time for the {given} {mw}'s 100m dash: {h_func} seconds.")
    
    end = input("Do you want to predict another year? (y/n) ").lower()
    if end != 'y':
        break