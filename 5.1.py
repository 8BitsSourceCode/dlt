import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers

import numpy as np

IMG_SIZE = (128, 128)

BATCH_SIZE = 32

# Load dataset from directory structure: data/train/<class_name>/*.jpg

train_ds = keras.utils.image_dataset_from_directory(
    'data/train',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset='training',
    seed=123
)

val_ds = keras.utils.image_dataset_from_directory(
    'data/train',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset='validation',
    seed=123
)

class_names = train_ds.class_names

print("Classes:", class_names)

normalization_layer = layers.Rescaling(1.0 / 255)

train_ds = train_ds.map(
    lambda x, y: (normalization_layer(x), y)
)

val_ds = val_ds.map(
    lambda x, y: (normalization_layer(x), y)
)

# Build CNN

model = keras.Sequential([
    layers.Input(shape=(128, 128, 3)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# Predict label for a new image

img = keras.utils.load_img(
    'new_image.jpg',
    target_size=IMG_SIZE
)

img_array = keras.utils.img_to_array(img) / 255.0

img_array = np.expand_dims(
    img_array,
    axis=0
)

predictions = model.predict(img_array)

predicted_class = class_names[
    np.argmax(predictions)
]

print(
    f"Predicted label: {predicted_class} "
    f"(confidence: {np.max(predictions) * 100:.2f}%)"
)