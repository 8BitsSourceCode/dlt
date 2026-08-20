import tensorflow as tf

from tensorflow import keras

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt


# Load and preprocess data

iris = load_iris()

X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


def build_model(activation='relu', optimizer='adam'):

    model = keras.Sequential([
        keras.layers.Dense(
            16,
            activation=activation,
            input_shape=(4,)
        ),

        keras.layers.Dense(
            8,
            activation=activation
        ),

        keras.layers.Dense(
            3,
            activation='softmax'
        )
    ])

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# Compare activation functions

results = {}

for act in ['relu', 'sigmoid', 'tanh']:

    model = build_model(
        activation=act,
        optimizer='adam'
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=50,
        validation_split=0.2,
        verbose=0
    )

    loss, acc = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    results[act] = acc

    print(
        f"Activation={act:8s} -> Test Accuracy = {acc:.4f}"
    )


# Compare optimizers

for opt in ['sgd', 'adam', 'rmsprop']:

    model = build_model(
        activation='relu',
        optimizer=opt
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=50,
        validation_split=0.2,
        verbose=0
    )

    loss, acc = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    print(
        f"Optimizer={opt:8s} -> Test Accuracy = {acc:.4f}"
    )