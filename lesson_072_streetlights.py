import numpy as np

def walk_stop(z):
    if z >= 0.5:
        print(z, "WALK")
    else:
        print(z, "STOP")

weights = np.array([1, 1, 1])
alpha = 0.1

streetlights = np.array([[1, 0, 1],
                        [0, 1, 1],
                        [0, 0, 1],
                        [1, 1, 1],
                        [0, 1, 1],
                        [1, 0, 1]])

walk_vs_stop = np.array([0, 1, 0, 1, 1, 0])

input = streetlights[0]
goal_prediction = walk_vs_stop[0]

for i in range(10):
    error_for_all_lights = 0
    for row_index in range(len(walk_vs_stop)):
        input = streetlights[row_index]
        goal_prediction = walk_vs_stop[row_index]
        
        prediction = input.dot(weights)
        # print(prediction)
        
        error = (goal_prediction - prediction) ** 2
        error_for_all_lights += error
        delta = prediction - goal_prediction
        weights = weights - alpha * input * delta
        walk_stop(prediction)
    print("Total Error:", error_for_all_lights)
print("Final weights are:", weights)