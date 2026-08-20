import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# Load Iris dataset
# Use only 2 classes:
# Setosa = 0
# Versicolor = 1
# Use only 2 features:
# Petal length and Petal width

iris = load_iris()

X = iris.data[:100, [2, 3]]
y = iris.target[:100]


# Split data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Standardize the data

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std


# Create Perceptron class

class Perceptron:

    def __init__(self, n_features, lr=0.01, epochs=50):
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.lr = lr
        self.epochs = epochs

    # Step activation function
    def activation(self, z):
        return np.where(z >= 0, 1, 0)

    # Train the perceptron
    def fit(self, X, y):

        for epoch in range(self.epochs):

            errors = 0

            for xi, target in zip(X, y):

                # Calculate weighted sum
                z = np.dot(xi, self.w) + self.b

                # Predict output
                y_pred = self.activation(z)

                # Calculate update
                update = self.lr * (target - y_pred)

                # Update weights and bias
                self.w += update * xi
                self.b += update

                # Count errors
                errors += int(update != 0.0)

            # Stop if there are no errors
            if errors == 0:
                print(f"Converged at epoch {epoch + 1}")
                break

    # Make predictions
    def predict(self, X):
        z = np.dot(X, self.w) + self.b
        return self.activation(z)


# Create Perceptron model

model = Perceptron(
    n_features=2,
    lr=0.01,
    epochs=50
)


# Train the model

model.fit(X_train, y_train)


# Make predictions

y_pred = model.predict(X_test)


# Calculate accuracy

accuracy = np.mean(y_pred == y_test) * 100

print(f"Test Accuracy: {accuracy:.2f}%")

print("Final weights:", model.w)
print("Bias:", model.b)