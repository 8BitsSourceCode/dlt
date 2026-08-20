import tensorflow as tf

from tensorflow import keras

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt


# Load CIFAR-10 dataset

(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

y_train, y_test = y_train.flatten(), y_test.flatten()


# Normalize

X_train = X_train.astype('float32') / 255.0

X_test = X_test.astype('float32') / 255.0


class_names = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]


# Build MLP model

model = keras.Sequential([
    keras.Input(shape=(32, 32, 3)),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])


model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


model.summary()


history = model.fit(
    X_train,
    y_train,
    epochs=15,
    batch_size=128,
    validation_split=0.2
)


test_loss, test_acc = model.evaluate(
    X_test,
    y_test,
    verbose=0
)


print(f"\nTest Accuracy: {test_acc * 100:.2f}%")


# Plot accuracy

plt.plot(
    history.history['accuracy'],
    label='train_acc'
)

plt.plot(
    history.history['val_accuracy'],
    label='val_acc'
)

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.legend()

plt.savefig('accuracy_plot.png')

print("\nAccuracy plot saved as: accuracy_plot.png")