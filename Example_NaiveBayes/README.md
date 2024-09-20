# Naive Bayes Classifier Examples

This repository contains two examples of implementing a Naive Bayes Classifier using the scikit-learn library in Python. The classifier predicts whether a vehicle is a Car or a Bike based on the number of wheels and the height of the vehicle.

## Prerequisites

Make sure you have Python installed and the following packages:

* scikit-learn
* numpy

To install these packages, you can use Conda or pip:

```
conda install scikit-learn numpy

```
Or with pip:

```
pip install scikit-learn numpy

```

## Example 1: Basic Naive Bayes Classifier

This is a simple implementation of the Naive Bayes classifier using a small dataset.

### Dataset

| #Wheels | Height | Class Label |
|---------|--------|-------------|
| 4       | High   | Car         |
| 4       | High   | Car         |
| 4       | High   | Car         |
| 2       | Low    | Bike        |
| 2       | Low    | Bike        |
| 2       | Low    | Bike        |
| 4       | Low    | Car         |
| 2       | High   | Bike        |


### How to Run

Navigate to the directory containing the naivebayes_example.py file. Run the following command in the terminal:

```
python naivebayes_example.py

```
Output:

The program will predict the vehicle class (Car/Bike) based on the number of wheels and height. Example output:

* Predicted labels: [0 0 0 1 1 1 0 1]
* Actual labels: [0 0 0 1 1 1 0 1]

## Example 2: Extended Naive Bayes Classifier with Data Splitting

This example extends the basic classifier by adding more data and splitting it into training and testing sets using train_test_split.

### Dataset:

| #Wheels | Height | Class Label |
|---------|--------|-------------|
| 4       | High   | Car         |
| 4       | High   | Car         |
| 4       | High   | Car         |
| 2       | Low    | Bike        |
| 2       | Low    | Bike        |
| 2       | Low    | Bike        |
| 4       | Low    | Car         |
| 2       | High   | Bike        |

| 4       | High   | Car         |
| 4       | Low    | Car         |
| 2       | High   | Bike        |
| 2       | High   | Bike        |
| 4       | High   | Car         |
| 2       | Low    | Bike        |
| 4       | Low    | Car         |

### How to Run

Navigate to the directory containing the naivebayes_example.v2.py file. Run the following command in the terminal:

```
python naivebayes_extended.py

```
Output:

The program will:

* Split the dataset into training and testing sets.
* Train the Naive Bayes classifier on the training data.
* Evaluate the model's accuracy on the test data.

Example output:

    Accuracy: 100.00%

    Classification Report:
                precision    recall  f1-score   support

            Car       1.00      1.00      1.00         3
            Bike      1.00      1.00      1.00         3

        accuracy                          1.00         6
        macro avg       1.00      1.00      1.00         6
    weighted avg      1.00      1.00      1.00         6