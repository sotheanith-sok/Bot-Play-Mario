# Code for program goes here
import retro

# import tensorflow as tf
import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)
import tensorflow as tf
from tensorflow.keras import datasets, layers, models


class QLearning(object):
    def __init__(self, num_control=12):
        pass

    def run(self, loop=100):
        """Run the loop to train  AI
        
        Keyword Arguments:
            loop {int} -- [How many loop to run] (default: {100})
        """
        model = self.build_model()

        # Random run to get the first sample
        initial_x, initial_y, score = self.get_training_data()
        initial_x, initial_y = self.regularization(initial_x, initial_y)

        model = self.train_model(model, initial_x, initial_y)

        for _ in range(loop):
            x, y, score = self.get_training_data(model=model)
            x, y = self.regularization(x, y)
            model = self.train_model(model, x, y)

        pass

    def get_training_data(self, model=None, threshold=0, max_iteration=1):
        """Get training data and score randomly or using model
        """

        # Variable to keep tracks of histories
        current_iteration = 0
        score_history = []
        observations_history = []
        actions_history = []

        env = retro.make(game="SuperMarioWorld-Snes", state="Forest1")

        while current_iteration < max_iteration:
            observations = []
            actions = []
            score = 0

            done = False
            previous_observation = env.reset()
            while not done:
                if model:
                    action = self.integers_to_bits(np.array([np.argmax(model.predict(np.array([previous_observation/255.])))]))[0]
                    pass
                else:
                    action = env.action_space.sample()

                observation, reward, done, info = env.step(action)
                print(env.get_action_meaning(action))
                env.render()
                observations.append(previous_observation)
                actions.append(action)

                previous_observation = observation
                score += reward

            score_history.append(score)
            observations_history.append(observations)
            actions_history.append(actions)

            current_iteration += 1
        env.close()

        index = np.argmax(score_history)

        return (
            np.array(observations_history[index]),
            np.array(actions_history[index]),
            score_history[index],
        )

    def regularization(self, x, y):
        """Regularize training data 
        
        Arguments:
            x {[type]} -- [feature set]
            y {[type]} -- [expected output]

        """
        # Scale x by 255 (Max color scaling)
        x = x / 255.0

        # Convert bits array to one hot encoding of 4096 possible classes
        y = self.bits_to_intergers(y)
        y = y.astype(float)

        return x, y

    def bits_to_intergers(self, values):
        """Convert array of bits array to int array
        """
        return values.dot(1 << np.arange(values.shape[-1]))

    def integers_to_bits(self, values, bits_length=12):
        """Covert array of integers to array of 12 bits
        """
        return (((values[:, None] & (1 << np.arange(bits_length)))) > 0).astype(int)

    def build_model(self, input_shape=(224, 256, 3), output_classes=4096):
        """Builing the cnn model
        
        Keyword Arguments:
            input_shape {tuple} -- [Size of our input] (default: {(224, 256, 3)})
            output_classes {int} -- [how many possible combination of input. 2^12 (SNES has 12 botton)] (default: {4096})
        """

        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation="relu"))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation="relu"))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation="relu"))
        model.add(layers.Dense(output_classes, activation="softmax"))
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def train_model(self, model, x, y, epochs=10):
        """Train model with given dataset for a certain amount of epochs
        """
        model.fit(x, y, epochs=epochs)
        return model


QLearning().run()

