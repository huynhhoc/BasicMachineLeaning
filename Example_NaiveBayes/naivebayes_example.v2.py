import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Extended Dataset: #Wheels, Height (0 = Low, 1 = High)
X = np.array([[4, 1], [4, 1], [4, 1], [2, 0], [2, 0], [2, 0], [4, 0], [2, 1],
              [4, 1], [4, 0], [2, 1], [2, 1], [4, 1], [2, 0], [4, 0]])

# Class labels: 0 = Car, 1 = Bike
y = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0])

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Gaussian Naive Bayes model
gnb = GaussianNB()

# Train the model on the training data
gnb.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = gnb.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["Car", "Bike"])

# Output the results
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", report)
