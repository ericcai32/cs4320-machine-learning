x = [1]
weights = [
    -8.01817212e+01,
     4.38719851e-01,
     7.60657962e-02,
     1.11130192e+00,
     5.59676444e-02,
    -8.29494295e+00
]
predicted_weight = 0

# Ask for user input
print("Greetings! Use this tool to predict the birth weight of a baby.")
x.append(input("How many days was the baby's gestation? "))
x.append(input("What is the mother's age in years? "))
x.append(input("What is the mother's height in inches? "))
x.append(input("What is the mother's pregnancy weight in pounds? "))
x.append(input("Did the mother smoke while pregnant? (0 = no | 1 = yes) "))

# Calculate predicted weight
for i in range(6):
    predicted_weight += weights[i] * int(x[i])

print(f"\nThe predicted weight for the baby is {predicted_weight} ounces.")