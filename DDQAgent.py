import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

from ReplayBuffer import ReplayBuffer
from models import build_model
from tensorflow.keras import models, backend
import gc
from os import path
import numpy as np
import pickle
import tensorflow as tf


class DDQAgent(object):
    def __init__(
        self,
        alpha,
        gamma,
        n_actions,
        epsilon,
        batch_size,
        input_dimension,
        epsilon_dec=99999975,
        epsilon_min=0.1,
        memory_size=20000,
        filename="DDQ_Model",
        replace_target=10000,
    ):
        """Initialize Double Deep Q Agent
        
        Arguments:
            alpha {float} -- [Learning rate of models]
            gamma {float} -- [Discount rate of the agent]
            n_actions {int} -- [Number of possible actions]
            epsilon {float} -- [Exploration factor]
            batch_size {int} -- [Size of memory that should be sample per training]
            input_dimension {tuple} -- [Dimension of inputs]
        
        Keyword Arguments:
            epsilon_dec {float} -- [Epsilon decresion rate] (default: {0.9999})
            epsilon_min {float} -- [Minimum Epsilon] (default: {0.01})
            memory_size {int} -- [Max memory size] (default: {1000})
            filename {str} -- [Name of model] (default: {"DDQ_Model.h5"})
            replace_target {int} -- [Interval in which weight of target model is being updated] (default: {100})
        """
        self.n_actions = n_actions
        self.actions_space = [i for i in range(self.n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_min
        self.memory = ReplayBuffer(memory_size, input_dimension, n_actions, True)
        self.filename = filename
        self.replace_target = replace_target
        self.batch_size = batch_size
        self.q_evaluation = build_model(alpha, n_actions, input_dimension)
        self.q_target = build_model(alpha, n_actions, input_dimension)

    def remember(self, state, action, reward, new_state, done):
        """Remember game information
        
        Arguments:
            state {array} -- [current oberservation]
            action {array} -- [current action]
            reward {float} -- [reward produce by such action]
            new_state {array} -- [new observation]
            done {bool} -- [Is the episode complete?]
        """
        action = self.encode_game_input(action)
        self.memory.store_trainsition(state, action, reward, new_state, done)

    def choose_action(self, state):
        """Choose action based on state
        
        Arguments:
            state {array} -- [Current observation]
        
        Returns:
            [arry] -- [action]
        """
        rand = np.random.random_sample()
        state = state[np.newaxis, :]

        # Exploration
        if rand < self.epsilon:
            action = np.random.choice(self.actions_space)
            
        # Greedy choice
        else:
            actions = self.q_evaluation.predict(state)
            action = np.argmax(actions)

        return self.decode_game_input(action)

    def learn(self):
        """Update weights of q models
        """

        # Only learning if memory capacity is above batch_size
        if self.memory.memory_counter > self.batch_size:

            # Sample memory for datasets
            states, actions, rewards, new_states, dones = self.memory.sample_buffer(
                self.batch_size
            )
            actions_value = np.array(self.actions_space, dtype=np.int8)

            # Convert one-hot-encoding to int
            action_indices = np.dot(actions, actions_value)

            # Find the ideal q for new states
            q_next = self.q_target.predict(new_states)

            # Find the predicted q for the new states
            q_eval = self.q_evaluation.predict(new_states)

            # Find the predicted q for the current states
            q_pred = self.q_evaluation.predict(states)

            # Find the most likely action to be taken by finding the maximum q
            max_actions = np.argmax(q_eval, axis=1)

            q_target = q_pred

            # Expand batch_size to array of 0,1,...,batch_size-1
            batch_index = np.arange(self.batch_size, dtype=np.int8)

            # Update q value of actions that will be taken to its ideal q
            q_target[batch_index, action_indices] = (
                rewards
                + self.gamma * q_next[batch_index, max_actions.astype(int)] * dones
            )

            # Train evaluation model to fit states to q_target
            self.q_evaluation.fit(states, q_target, verbose=0)

            # Update epsilon
            self.epsilon = (
                self.epsilon * self.epsilon_dec
                if self.epsilon > self.epsilon_min
                else self.epsilon_min
            )

            # Update target model after a certain amount of training
            if self.memory.memory_counter % self.replace_target == 0:
                self.update_network_params()

    def update_network_params(self):
        """Update target model with weights of evaluation model
        """
        self.q_target.set_weights(self.q_evaluation.get_weights())

    def save_model(self):
        """Save evaluation model
        """
        self.q_evaluation.save(self.filename+"_eval.h5")
        self.q_target.save(self.filename+"_target.h5")
        backend.clear_session()
        gc.collect()
        if path.exists(self.filename+"_eval.h5"):
            self.q_evaluation = models.load_model(self.filename+"_eval.h5")
            self.q_target = models.load_model(self.filename+"_target.h5")
        self.save_parameters()


    def load_model(self):
        """Load evaluation model. Update target model if epsilon is 0
        """
        if path.exists(self.filename+"_eval.h5"):
            self.q_evaluation = models.load_model(self.filename+"_eval.h5")
            self.q_target = models.load_model(self.filename+"_target.h5")

        self.load_parameters()

    def encode_game_input(self, value):
        """Compress game_input to smaller array to remove unneccessary input
        """
        raw_input = np.array(value, dtype=np.int8)
        if np.array_equal(raw_input, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
            return 0  # Doing nothing
        elif np.array_equal(raw_input, [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]):
            return 1  # Run right
        elif np.array_equal(raw_input, [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]):
            return 2  # Run left
        elif np.array_equal(raw_input, [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]):
            return 3  # down
        elif np.array_equal(raw_input, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
            return 4  # Jump
        elif np.array_equal(raw_input, [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]):
            return 5  # Spin Jump
        elif np.array_equal(raw_input, [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]):
            return 6  # Jump Right
        elif np.array_equal(raw_input, [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]):
            return 7  # Spin Jump Right
        

    def decode_game_input(self, value):
        """Decompress input back to its original form
        """
        if value == 0:
            return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif value == 1:
            return [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        elif value == 2:
            return [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        elif value == 3:
            return [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        elif value == 4:
            return [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif value == 5:
            return [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        elif value == 6:
            return [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        elif value == 7:
            return [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]

    def save_parameters(self):
        with open('hyperparameters.pkl', 'wb') as f:
            pickle.dump(self.epsilon, f)

    def load_parameters(self):
        if path.exists("hyperparameters.pkl"):
            with open('hyperparameters.pkl', "rb") as f:  # Python 3: open(..., 'rb')
                self.epsilon = pickle.load(f)
