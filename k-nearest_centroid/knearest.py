import numpy as np
from collections import Counter

# Sample dataset: (x1, x2, class)
dataset = [
    (2, 4, 'Red'),
    (4, 6, 'Red'),
    (4, 2, 'Blue'),
    (6, 4, 'Blue'),
    (6, 6, 'Red')
]

# New data point
d_new = np.array([5, 5])

# Function to calculate Euclidean distance
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

# k-Nearest Neighbors function
def k_nearest_neighbors(dataset, d_new, k):
    # Calculate distances between d_new and all dataset points
    distances = []
    for data in dataset:
        point = np.array([data[0], data[1]])
        dist = euclidean_distance(point, d_new)
        distances.append((dist, data[2]))  # (distance, class label)
    
    # Sort the distances
    distances.sort(key=lambda x: x[0])
    
    # Select the top k neighbors
    k_neighbors = distances[:k]
    
    # Get the most common class among the k neighbors
    k_labels = [neighbor[1] for neighbor in k_neighbors]
    most_common_class = Counter(k_labels).most_common(1)[0][0]
    
    return most_common_class

# Set k to 3
k = 3
result = k_nearest_neighbors(dataset, d_new, k)

print(f'The new point {d_new} is classified as: {result}')
