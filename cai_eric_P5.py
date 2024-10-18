import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

def find_closest_centroid(df, centroids):
    cluster_1_x1s = []
    cluster_1_x2s = []
    cluster_2_x1s = []
    cluster_2_x2s = []
    for x1, x2 in zip(df['x1'], df['x2']):
        distances = []
        for centroid in centroids:
            x1_distance = centroid[0] - x1
            x2_distance = centroid[1] - x2
            distance = (x1_distance ** 2 + x2_distance ** 2) ** (1/2)
            distances.append(distance)
        ci = distances.index(min(distances))
        if ci == 0:
            cluster_1_x1s.append(x1)
            cluster_1_x2s.append(x2)
        else:
            cluster_2_x1s.append(x1)
            cluster_2_x2s.append(x2)
    
    cluster_1 = pd.DataFrame({'x1': cluster_1_x1s, 'x2': cluster_1_x2s})
    cluster_2 = pd.DataFrame({'x1': cluster_2_x1s, 'x2': cluster_2_x2s})
    
    # plt.figure()
    # plt.scatter(cluster_1['x1'], cluster_1['x2'], c='red')
    # plt.scatter(cluster_2['x1'], cluster_2['x2'], c='purple')   
    # plt.scatter(centroids[0][0], centroids[0][1], c='black', marker='^')
    # plt.scatter(centroids[1][0], centroids[1][1], c='green', marker='^')
    
    return cluster_1, cluster_2
            
def find_new_centroids(cluster_1, cluster_2):
    cluster_1_new_x1 = np.average(cluster_1['x1'])
    cluster_1_new_x2 = np.average(cluster_1['x2'])
    cluster_2_new_x1 = np.average(cluster_2['x1'])
    cluster_2_new_x2 = np.average(cluster_2['x2'])
    centroid_1 = np.array([cluster_1_new_x1, cluster_1_new_x2])
    centroid_2 = np.array([cluster_2_new_x1, cluster_2_new_x2])
    return centroid_1, centroid_2

def calculate_J(centroids, clusters):
    distances_squared = 0
    m = 0
    for centroid, cluster in zip(centroids, clusters):
        m += len(cluster)
        x1_distances_squared = np.sum((cluster['x1'] - centroid[0]) ** 2)
        x2_distances_squared = np.sum((cluster['x2'] - centroid[1]) ** 2)
        cluster_distances_squared = x1_distances_squared + x2_distances_squared
        distances_squared += cluster_distances_squared
    J = (1/m) * distances_squared
    return J

file = input("Enter the name of the file you want to cluster: ")
df = pd.read_csv(file)

initial_centroids = []
final_centroids = []
final_Js = []

for i in range(5):
    centroid_1 = np.array([random.randint(66, 78), random.randint(167, 267)])
    centroid_2 = np.array([random.randint(66, 78), random.randint(167, 267)])
    initial_centroids.append((centroid_1, centroid_2))
    
    cluster_1, cluster_2 = find_closest_centroid(df, [centroid_1, centroid_2])
    J = calculate_J([centroid_1, centroid_2], [cluster_1, cluster_2])
    while True:
        centroid_1, centroid_2 = find_new_centroids(cluster_1, cluster_2)
        df = pd.concat([cluster_1, cluster_2])
        cluster_1, cluster_2 = find_closest_centroid(df, [centroid_1, centroid_2])
        original_J = J
        J = calculate_J([centroid_1, centroid_2], [cluster_1, cluster_2])
        
        
        if J == original_J:
            break

    # plt.figure()
    # plt.scatter(cluster_1['x1'], cluster_1['x2'], c='red')
    # plt.scatter(cluster_2['x1'], cluster_2['x2'], c='purple')   
    # plt.scatter(centroid_1[0], centroid_1[1], c='black', marker='^')
    # plt.scatter(centroid_2[0], centroid_2[1], c='green', marker='^')
        
    final_centroids.append([centroid_1, centroid_2])
    final_Js.append(J)

min_J = min(final_Js)
min_centroids = final_centroids[final_Js.index(min_J)]

cluster_1, cluster_2 = find_closest_centroid(df, [min_centroids[0], min_centroids[1]])

plt.figure()
plt.scatter(cluster_1['x1'], cluster_1['x2'], c='red')
plt.scatter(cluster_2['x1'], cluster_2['x2'], c='purple')   
plt.scatter(min_centroids[0][0], min_centroids[0][1], c='black', marker='^')
plt.scatter(min_centroids[1][0], min_centroids[1][1], c='green', marker='^')


print("INITIAL CENTROIDS")
for initial_centroid in initial_centroids:
    print(*initial_centroid)
print("\nFINAL CENTROIDS")
for final_centroid in final_centroids:
    print(*final_centroid)
print("\nFINAL J VALUES")
for final_J in final_Js:
    print(final_J)
print(f"\n\nThe best centroids are {min_centroids[0]} and {min_centroids[1]}.")
print(f"They give an error term of {min_J}.")
print("This iteration is plotted.")
