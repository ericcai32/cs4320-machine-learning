import numpy as np

A = np.array([[1, 2, 3], [4, 5, 6]])
print(A.shape)

B = np.array([[-1, -2], [3, 1], [-3, 5]])
print(B.shape)

C = np.dot(A, B)
print(C)

print(np.identity(2))

C = np.array([[1, 3], [3, 5]])
D = np.linalg.pinv(C)
print("Inverse of C is:", D)

print(np.dot(C, D))

print(C.T)