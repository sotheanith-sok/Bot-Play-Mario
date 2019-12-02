import numpy as np
import sys


class PrioritizedReplayBuffer(object):
    def __init__(self, max_size, input_dimension, n_actions, discrete=False):
        """Initialize prioritized replay buffer
        
        Arguments:
            max_size {int} -- [buffer capacity]
            input_dimension {numpy.array} -- [shape of observations]
            n_actions {int} -- [number of possible actions]
        
        Keyword Arguments:
            discrete {bool} -- [is action descrete or continue?] (default: {False})
        """

        # Memory counter uses to keep track of how many experience has been stored so far
        self.memory_counter = 0

        # Save buffer memory size
        self.memory_size = max_size

        # Initialize array for storing oberservations
        self.states_memory = np.zeros(
            ((self.memory_size,) + input_dimension), dtype=np.float32
        )
        self.new_states_memory = np.zeros(
            ((self.memory_size,) + input_dimension), dtype=np.float32
        )

        # Initialize array for storing actions, reward, terminals, and priorities
        self.discrete = discrete
        dtype = np.int8 if self.discrete else np.float
        self.actions_memory = np.zeros(
            ((self.memory_size,) + (n_actions,)), dtype=dtype
        )
        self.rewards_memory = np.zeros(self.memory_size, dtype=np.float32)
        self.terminals_memory = np.zeros(self.memory_size, dtype=np.int8)
        self.priorities = np.zeros(self.memory_size, dtype=np.float32)

    def store_trainsition(self, state, action, reward, new_state, done):
        """Store experience into memory
        
        Arguments:
            state {numpy.array} -- [current state]
            action {int or float} -- [action taken given current state]
            reward {float} -- [reward produces from given action]
            new_state {numpy.array} -- [new state]
            done {0 or 1} -- [is it the end of episode]
        """
        # Decide which slot to store experience
        index = self.memory_counter % self.memory_size
        self.states_memory[index] = state
        self.actions_memory[index] = action
        if self.discrete:
            actions = np.zeros(np.shape(self.actions_memory)[1])
            actions[action] = 1.0
            self.actions_memory[index] = actions
        else:
            self.actions_memory[index] = action
        self.new_states_memory[index] = new_state
        self.terminals_memory[index] = 1 - int(done)

        # Calculate priority of experience
        if self.memory_counter == 0:
            self.priorities[index] = 1
        else:
            self.priorities[index] = np.max(self.priorities)

        # Increase memory counter
        self.memory_counter += 1

    def get_probabilities(self, priority_scale):
        """Get scaled priorities
        
        Arguments:
            priority_scale {float} -- [priority scaling factor. 0 is pure randomness and 
                                        1 is select only highest priorities]
        """
        scaled_priority = np.power(self.priorities, priority_scale)
        sample_probabilities = np.divide(scaled_priority, np.sum(scaled_priority))
        return sample_probabilities

    def get_importance(self, probabilities, probabilities_scale):
        """Importance of the probabilities
        
        Arguments:
            probabilities {numpy.array} -- [probabilities of sample experience]
            probabilities_scale {float} -- [probabilities scaling factor. 0 at start and 1 at the
                                             end because this weight is importance in the end as 
                                             graph begin to converage]
        """
        importance = np.power(
            1 / self.memory_size * 1 / probabilities, probabilities_scale
        )
        importance_normalized = importance / max(importance)
        return importance_normalized

    def sample_buffer(self, batch_size, priority_scale=1.0, probabilities_scale=1.0):
        """Sample some experience from buffers
        
        Arguments:
            batch_size {int} -- [number of experiences to sample]
        
        Keyword Arguments:
            priority_scale {float} -- [priority scaling factor] (default: {1.0})
            probabilities_scale {float} -- [probabilities scaling factor] (default: {1.0})
        """

        max_memory = min(self.memory_counter, self.memory_size)
        sample_probs = self.get_probabilities(priority_scale)

        indices = np.random.choice(max_memory, batch_size, p=sample_probs[0:max_memory])
        states = self.states_memory[indices]
        new_states = self.new_states_memory[indices]
        rewards = self.rewards_memory[indices]
        actions = self.actions_memory[indices]
        terminals = self.terminals_memory[indices]
        importances = self.get_importance(sample_probs[indices], probabilities_scale)
        return states, actions, rewards, new_states, terminals, importances, indices

    def set_priorities(self, indices, errors, offset=0.1):
        """Update priorities in buffer
        
        Arguments:
            indices {int} -- [which experience to update]
            errors {float} -- [loss as result of such experiences]
        
        Keyword Arguments:
            offset {float} -- [same offset to ensure that priorities never be 0] (default: {0.1})
        """
        for i, e in zip(indices, errors):
            self.priorities[i] = abs(e) + offset
