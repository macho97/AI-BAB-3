"""AlexNet-style image classifier implemented with TensorFlow Keras."""

import tensorflow as tf
from tensorflow.keras import layers


model = tf.keras.Sequential(
    [
        layers.Input(shape=(224, 224, 3)),
        layers.Conv2D(
            filters=96,
            kernel_size=(11, 11),
            strides=(4, 4),
            activation="relu",
            padding="valid",
        ),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"),
        layers.Conv2D(
            filters=256,
            kernel_size=(11, 11),
            activation="relu",
            padding="valid",
        ),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"),
        layers.Conv2D(filters=384, kernel_size=(3, 3), activation="relu"),
        layers.Conv2D(filters=384, kernel_size=(3, 3), activation="relu"),
        layers.Conv2D(filters=256, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"),
        layers.Flatten(),
        layers.Dense(4096, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(4096, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(1000, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(17, activation="softmax"),
    ],
    name="alexnet",
)

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
)

model.summary()
