import tensorflow as tf
from tensorflow.python.keras.api import keras
from tensorflow.python.keras.api.keras import (
    datasets,
    layers,
    models,
    optimizers,
    metrics,
)


def build_model(learning_rate, n_actions, input_dimension):
    model = models.Sequential()
    model.add(layers.Conv2D(16, (3, 3), activation="relu", input_shape=input_dimension))
    model.add(layers.Conv2D(32, (3, 3), activation="relu"))
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dense(n_actions, activation="softmax"))
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )
    return model
