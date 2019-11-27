import os
import logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # FATAL
logging.getLogger("tensorflow").setLevel(logging.FATAL)

import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers, metrics, Model, Input, callbacks
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.losses import mse


def build_model(learning_rate, n_actions, input_dimension):

    model = models.Sequential()
    model.add(
        layers.Conv2D(
            32, (8, 8), strides=8, activation="relu", input_shape=input_dimension
        )
    )
    model.add(layers.Conv2D(64, (4, 4), strides=4, activation="relu"))
    model.add(layers.Conv2D(64, (3, 3), strides=1, activation="relu"))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dense(n_actions, activation="softmax"))
    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate), loss="mse")
    return model
