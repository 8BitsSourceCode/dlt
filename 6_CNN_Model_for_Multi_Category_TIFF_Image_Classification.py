import tensorflow as tf

from tensorflow.keras import layers, regularizers, Sequential

import numpy as np

from PIL import Image

import glob, os

import matplotlib.pyplot as plt


def load_tiff_dataset(root_dir, img_size=(64, 64)):

    X, y, class_names = [], [], sorted(os.listdir(root_dir))

    for idx, cls in enumerate(class_names):

        for path in glob.glob(f"{root_dir}/{cls}/*.tif*"):

            img = Image.open(path).convert('RGB').resize(img_size)

            X.append(np.array(img) / 255.0)

            y.append(idx)

    return np.array(X, dtype='float32'), np.array(y), class_names


X, y, class_names = load_tiff_dataset('data/tiff_images')

n = len(X)
split = int(0.8 * n)

X_train, X_test = X[:split], X[split:]

y_train, y_test = y[:split], y[split:]


# Baseline CNN (prone to overfitting on small datasets)

baseline = Sequential([
    layers.Input(shape=(64, 64, 3)),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])

baseline.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

hist_base = baseline.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=30,
    verbose=0
)


# Diagnosis: compare final train vs val accuracy

train_acc = hist_base.history['accuracy'][-1]

val_acc = hist_base.history['val_accuracy'][-1]

print(f"Baseline -> train_acc={train_acc:.3f}, val_acc={val_acc:.3f}")

if train_acc - val_acc > 0.15:

    print("Diagnosis: OVERFITTING")

elif train_acc < 0.6 and val_acc < 0.6:

    print("Diagnosis: UNDERFITTING")

else:

    print("Diagnosis: GOOD FIT")


# Improved CNN with Dropout + L2 regularization to fix overfitting

improved = Sequential([
    layers.Input(shape=(64, 64, 3)),
    layers.Conv2D(
        32,
        3,
        activation='relu',
        kernel_regularizer=regularizers.l2(1e-4)
    ),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),
    layers.Conv2D(
        64,
        3,
        activation='relu',
        kernel_regularizer=regularizers.l2(1e-4)
    ),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(
        128,
        activation='relu',
        kernel_regularizer=regularizers.l2(1e-4)
    ),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation='softmax')
])

improved.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

hist_imp = improved.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=30,
    verbose=0
)

test_loss, test_acc = improved.evaluate(
    X_test,
    y_test,
    verbose=0
)

print(f"Improved model Test Accuracy: {test_acc * 100:.2f}%")