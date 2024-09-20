import numpy as np
from sklearn.naive_bayes import GaussianNB

# Dataset: #Wheels, Height (0 = Low, 1 = High)
X = np.array([[4, 1], 
              [4, 1], 
              [4, 1], 
              [2, 0], 
              [2, 0], 
              [2, 0], 
              [4, 0], 
              [2, 1]])

# Class labels: 0 = Car, 1 = Bike
y = np.array([0, 0, 0, 1, 1, 1, 0, 1])

# Initialize the Gaussian Naive Bayes model
gnb = GaussianNB()

# Fit the model on the dataset
gnb.fit(X, y)

# Predictions
y_pred = gnb.predict(X)

# Output the results
print(f"Predicted labels: {y_pred}")
print(f"Actual labels: {y}")
