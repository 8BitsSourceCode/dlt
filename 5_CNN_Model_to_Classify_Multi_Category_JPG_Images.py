import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np


# Image size and batch size

IMG_SIZE = (128, 128)
BATCH_SIZE = 32


# Load training dataset
# Expected directory structure:
#
# data/
# └── train/
#     ├── class1/
#     │   ├── image1.jpg
#     │   ├── image2.jpg
#     │   └── ...
#     ├── class2/
#     │   ├── image1.jpg
#     │   └── ...
#     └── ...


train_ds = keras.utils.image_dataset_from_directory(
    "data/train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="training",
    seed=123
)


# Load validation dataset

val_ds = keras.utils.image_dataset_from_directory(
    "data/train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="validation",
    seed=123
)


# Get class names

class_names = train_ds.class_names

print("Classes:", class_names)


# Normalize pixel values from 0-255 to 0-1

normalization_layer = layers.Rescaling(1.0 / 255)

train_ds = train_ds.map(
    lambda x, y: (normalization_layer(x), y)
)

val_ds = val_ds.map(
    lambda x, y: (normalization_layer(x), y)
)


# Build CNN model

model = keras.Sequential([
    layers.Input(shape=(128, 128, 3)),

    layers.Conv2D(
        32,
        3,
        activation="relu"
    ),

    layers.MaxPooling2D(),

    layers.Conv2D(
        64,
        3,
        activation="relu"
    ),

    layers.MaxPooling2D(),

    layers.Conv2D(
        128,
        3,
        activation="relu"
    ),

    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dense(
        len(class_names),
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


# Train the CNN

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)


# Predict label for a new image

img = keras.utils.load_img(
    "new_image.jpg",
    target_size=IMG_SIZE
)


# Convert image to NumPy array

img_array = keras.utils.img_to_array(img)

img_array = img_array / 255.0


# Add batch dimension

img_array = np.expand_dims(
    img_array,
    axis=0
)


# Make prediction

predictions = model.predict(img_array)


# Get predicted class

predicted_class = class_names[
    np.argmax(predictions)
]


# Get confidence

confidence = np.max(predictions) * 100


print(
    f"Predicted label: {predicted_class} "
    f"(confidence: {confidence:.2f}%)"
)