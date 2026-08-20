import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt


# Load CIFAR-10 dataset

(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()


# Convert labels from 2D to 1D

y_train = y_train.flatten()
y_test = y_test.flatten()


# Normalize pixel values

X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0


# CIFAR-10 class names

class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]


# Build MLP model

model = keras.Sequential([
    keras.Input(shape=(32, 32, 3)),

    keras.layers.Flatten(),

    keras.layers.Dense(
        512,
        activation="relu"
    ),

    keras.layers.Dropout(0.3),

    keras.layers.Dense(
        256,
        activation="relu"
    ),

    keras.layers.Dense(
        10,
        activation="softmax"
    )
])


# Compile the model

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# Display model architecture

model.summary()


# Train the model

history = model.fit(
    X_train,
    y_train,
    epochs=15,
    batch_size=128,
    validation_split=0.2
)


# Evaluate the model

test_loss, test_acc = model.evaluate(
    X_test,
    y_test,
    verbose=0
)


print(f"\nTest Accuracy: {test_acc * 100:.2f}%")


# Plot training and validation accuracy

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Train Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title("Training and Validation Accuracy")

plt.legend()

plt.grid(True)

plt.show()