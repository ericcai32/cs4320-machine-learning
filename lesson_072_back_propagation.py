import numpy as np
np.random.seed(1)

def relu(x):
    return (x > 0) * x

def relu2deriv(output):
    return output > 0

alpha = 0.2
hidden_size = 4

streetlights = np.array([[1, 0, 1],
                         [0, 1, 1],
                         [0, 0, 1],
                         [1, 1, 1]])
walk_vs_stop = np.array([[1, 1, 0, 0]]).T

# Randomly generate weighst.
weights_0_1 = 2 * np.random.random((3, hidden_size)) - 1
weights_1_2 = 2 * np.random.random((hidden_size, 1)) - 1

print(weights_0_1)
print(weights_1_2)

for i in range(60):
    layer_2_error = 0
    for j in range(len(streetlights)):
        layer_0 = streetlights[j:j+1]
        layer_1 = relu(np.dot(layer_0, weights_0_1))
        layer_2 = np.dot(layer_1, weights_1_2)
        
        print(f"\n{layer_0} *** {layer_2}")
        
        layer_2_error += np.sum((layer_2 - walk_vs_stop[j:j+1]) ** 2)
        layer_2_delta = (layer_2 - walk_vs_stop[j:j+1])
        layer_1_delta = layer_2_delta.dot(weights_1_2.T) * relu2deriv(layer_1)
        
        weights_1_2 = weights_1_2 - alpha * layer_1.T.dot(layer_2_delta)
        weights_0_1 = weights_0_1 - alpha * layer_0.T.dot(layer_1_delta)
    if i % 10 == 9:
        print(f"Error: {layer_2_error}")