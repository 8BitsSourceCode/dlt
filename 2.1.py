import numpy as np

import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split


# Load data - use only 2 classes (Setosa=0, Versicolor=1) and 2 features

iris = load_iris()

X = iris.data[:100, [2, 3]]       # petal length, petal width

y = iris.target[:100]             # classes 0 and 1


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Standardize

mean, std = X_train.mean(axis=0), X_train.std(axis=0)

X_train = (X_train - mean) / std

X_test = (X_test - mean) / std


class Perceptron:

    def __init__(self, n_features, lr=0.01, epochs=50):
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.lr = lr
        self.epochs = epochs

    def activation(self, z):
        return np.where(z >= 0, 1, 0)

    def fit(self, X, y):

        for epoch in range(self.epochs):

            errors = 0

            for xi, target in zip(X, y):

                z = np.dot(xi, self.w) + self.b

                y_pred = self.activation(z)

                update = self.lr * (target - y_pred)

                self.w += update * xi

                self.b += update

                errors += int(update != 0.0)

            if errors == 0:
                print(f"Converged at epoch {epoch + 1}")
                break

    def predict(self, X):

        z = np.dot(X, self.w) + self.b

        return self.activation(z)


model = Perceptron(
    n_features=2,
    lr=0.01,
    epochs=50
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = np.mean(y_pred == y_test) * 100

print(f"Test Accuracy: {accuracy:.2f}%")

print("Final weights:", model.w, "Bias:", model.b)