# k-Nearest Neighbors (k-NN) & Centroid-Based Classifier

This repository contains two Python scripts implementing basic versions of the **k-Nearest Neighbors (k-NN)** algorithm and the **Centroid-Based Classifier** (Rocchio Classifier). Both are simple classification algorithms used in supervised machine learning tasks.

## Overview

### k-Nearest Neighbors (k-NN)
The k-Nearest Neighbors algorithm classifies new data points based on the class of their **k nearest points** in the dataset. In this implementation, we use **Euclidean distance** to measure the closeness of points. Once the distances are calculated, the majority class among the `k` nearest neighbors is used to classify the new point.

### Centroid-Based Classifier (Rocchio Classifier)
The Centroid-Based Classifier assigns a new point to the class whose **centroid** (mean of points in the class) is closest to the new point. The distance between the new point and each class centroid is calculated, and the new point is assigned the class with the minimum distance.


##  How It Works

1. k-Nearest Neighbors (k-NN)

* Input: A dataset of points with labels, and a new point to classify.

* Process:
- Compute the Euclidean distance between the new point and all points in the dataset.
- Select the k closest neighbors.
- Perform majority voting to determine the class of the new point.
* Output: The class label of the new point.

2. Centroid-Based Classifier

* Input: A dataset of points with labels, and a new point to classify.

* Process:

- Compute the centroid of each class.
- Calculate the Euclidean distance from the new point to each centroid.
- Assign the class label of the nearest centroid to the new point.

* Output: The class label of the new point.