# Numpy array examples
import numpy as np

# x = np.array([1, 2, 3, 4])
# print(x)
# print(x.shape)

# x = np.array([[1, 2, 3, 4]])
# print(x)
# print(x.shape)

A = np.array([[1, 2, 3], [4, 5, 6]])
print(A)
print(A.shape)

b = np.array([[1], [2], [3]])
print(b)
print(b.shape)
print(np.zeros((2,3)))
print('')
print(A + A)
print(3 * A)
print('')
print(b * A)