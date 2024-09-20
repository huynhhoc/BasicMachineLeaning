import numpy as np

# Sample dataset: (x1, x2, class)
dataset = [
    (2, 4, 'Red'),
    (4, 6, 'Red'),
    (6, 6, 'Red'),
    (4, 2, 'Blue'),
    (6, 4, 'Blue')
]

# New data point
d_new = np.array([5, 5])

# Function to calculate the centroid of each class
def compute_centroids(dataset):
    red_points = [np.array([data[0], data[1]]) for data in dataset if data[2] == 'Red']
    blue_points = [np.array([data[0], data[1]]) for data in dataset if data[2] == 'Blue']
    
    # Centroid = mean of all points in the class
    c_red = np.mean(red_points, axis=0)
    c_blue = np.mean(blue_points, axis=0)
    
    return c_red, c_blue

# Function to calculate Euclidean distance
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

# Centroid-Based Classifier function
def centroid_classifier(dataset, d_new):
    # Compute centroids
    c_red, c_blue = compute_centroids(dataset)
    
    # Compute distances from d_new to both centroids
    distance_to_red = euclidean_distance(d_new, c_red)
    distance_to_blue = euclidean_distance(d_new, c_blue)
    
    # Assign the class based on the closest centroid
    if distance_to_red < distance_to_blue:
        return 'Red'
    else:
        return 'Blue'

# Classify the new data point
result = centroid_classifier(dataset, d_new)

print(f'The new point {d_new} is classified as: {result}')
