import numpy as np
import matplotlib.pyplot as plt

quadratic_errors = np.array([
    262.757,
    252.619,
    243.589,
    263.696,
    250.158,
])

linear_errors = np.array([
    266.464,
    240.124,
    246.96,
    263.952,
    248.823,
])

error_differences = linear_errors - quadratic_errors
plt.scatter(range(1, 6), error_differences)
plt.xlabel("Test Set")
plt.xticks(range(1, 6))
plt.ylabel("Linear Error - Quadratic Error")
plt.title("Error Difference for Each Test Set")


print(error_differences)