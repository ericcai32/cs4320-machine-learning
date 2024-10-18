import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    
    plt.figure()
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.xticks(range(11))
    plt.yticks(range(11))
    
    plt.scatter(centroids[0][0], centroids[0][1], c='red', marker='^')
    plt.scatter(centroids[1][0], centroids[1][1], c='purple', marker='^')
    plt.scatter(cluster_1['x1'], cluster_1['x2'], c='red')
    plt.scatter(cluster_2['x1'], cluster_2['x2'], c='purple')    
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

df = pd.read_csv('Data/ClassExample.csv')
with open('Data/ClassExCs.txt') as f:
    f.readline()
    centroid_coords = f.readline()
    centroid_1 = np.array(centroid_coords.split()).astype(int)
    centroid_coords = f.readline()
    centroid_2 = np.array(centroid_coords.split()).astype(int)

plt.scatter(df['x1'], df['x2'])
plt.scatter(centroid_1[0], centroid_1[1], c='red', marker='^')
plt.scatter(centroid_2[0], centroid_2[1], c='purple', marker='^')

plt.xlim(0, 10)
plt.ylim(0, 10)
plt.xticks(range(11))
plt.yticks(range(11))

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