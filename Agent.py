import json
import os

import numpy as np
import ray

import tensorflow as tf


class Agent(object):
    """Implements a standard DDDQN agent"""

    def __init__(self,
                 dqn,
                 target_dqn,
                 replay_buffer,
                 n_actions,
                 input_shape=(2,),
                 batch_size=32,
                 eps_initial=1,
                 eps_final=0.1,
                 eps_final_state=0.01,
                 eps_evaluation=0.0,
                 eps_annealing_states=1000000,
                 replay_buffer_start_size=50000,
                 max_states=25000000,
                 use_per=True
                 ):
        """
        Arguments:
            dqn: A DQN (returned by the DQN function) to predict moves
            target_dqn: A DQN (returned by the DQN function) to predict target-q values.  This can be initialized in the same way as the dqn argument
            replay_buffer: A ReplayBuffer object for holding all previous experiences
            n_actions: Number of possible actions for the given environment
            input_shape: Tuple/list describing the shape of the pre-processed environment
            batch_size: Number of samples to draw from the replay memory every updating session
            eps_initial: Initial epsilon value.
            eps_final: The "half-way" epsilon value.  The epsilon value decreases more slowly after this
            eps_final_state: The final epsilon value
            eps_evaluation: The epsilon value used during evaluation
            eps_annealing_states: Number of states during which epsilon will be annealed to eps_final, then eps_final_state
            replay_buffer_start_size: Size of replay buffer before beginning to learn (after this many states, epsilon is decreased more slowly)
            max_states: Number of total states the agent will be trained for
            use_per: Use PER instead of classic experience replay
        """

        self.n_actions = n_actions
        self.input_shape = input_shape

        # Memory information
        self.replay_buffer_start_size = replay_buffer_start_size
        self.max_states = max_states
        self.batch_size = batch_size

        self.replay_buffer = replay_buffer
        self.use_per = use_per

        # Epsilon information
        self.eps_initial = eps_initial
        self.eps_final = eps_final
        self.eps_final_state = eps_final_state
        self.eps_evaluation = eps_evaluation
        self.eps_annealing_states = eps_annealing_states

        # Slopes and intercepts for exploration decrease
        # (Credit to Fabio M. Graetz for this and calculating epsilon based on state number)
        self.slope = -(self.eps_initial - self.eps_final) / self.eps_annealing_states
        self.intercept = self.eps_initial - self.slope * self.replay_buffer_start_size
        self.slope_2 = -(self.eps_final - self.eps_final_state) / (
                self.max_states - self.eps_annealing_states - self.replay_buffer_start_size)
        self.intercept_2 = self.eps_final_state - self.slope_2 * self.max_states

        # DQN
        self.dqn = dqn
        self.target_dqn = target_dqn

    def calc_epsilon(self, state_number, evaluation=False):
        """Get the appropriate epsilon value from a given state number
        Arguments:
            state_number: Global state number (used for epsilon)
            evaluation: True if the model is evaluating, False otherwise (uses eps_evaluation instead of default epsilon value)
        Returns:
            The appropriate epsilon value
        """
        if evaluation:
            return self.eps_evaluation
        elif state_number < self.replay_buffer_start_size:
            return self.eps_initial
        elif self.replay_buffer_start_size <= state_number < self.replay_buffer_start_size + self.eps_annealing_states:
            return self.slope * state_number + self.intercept
        elif state_number >= self.replay_buffer_start_size + self.eps_annealing_states:
            return self.slope_2 * state_number + self.intercept_2

    def get_action(self, state_number, state, evaluation=False):
        """Query the DQN for an action given a state
        Arguments:
            state_number: Global state number (used for epsilon)
            state: State to give an action for
            evaluation: True if the model is evaluating, False otherwise (uses eps_evaluation instead of default epsilon value)
        Returns:
            An integer as the predicted move
        """

        # Calculate epsilon based on the state number
        eps = self.calc_epsilon(state_number, evaluation)

        # With chance epsilon, take a random action
        if np.random.rand(1) < eps:
            return np.random.randint(0, self.n_actions)

        # Otherwise, query the DQN for an action
        q_vals = self.dqn.predict.remote(np.expand_dims(state, axis=0))
        action = q_vals.argmax()
        return action

    def update_target_network(self):
        """Update the target Q network"""
        self.target_dqn.set_weights.remote(self.dqn.get_weights.remote())

    def add_experience(self, action, state, reward, terminal):
        """Wrapper function for adding an experience to the Agent's replay buffer"""
        self.replay_buffer.add_experience(action, state, reward, terminal)

    def learn(self, batch_size, gamma, state_number, priority_scale=1.0):
        """Sample a batch and use it to improve the DQN
        Arguments:
            batch_size: How many samples to draw for an update
            gamma: Reward discount
            state_number: Global state number (used for calculating importance)
            priority_scale: How much to weight priorities when sampling the replay buffer. 0 = completely random, 1 = completely based on priority
        Returns:
            The loss between the predicted and target Q as a float
        """
        if self.use_per:
            (states, actions, rewards, new_states,
             terminal_flags), importance, indices = self.replay_buffer.get_minibatch(batch_size=self.batch_size,
                                                                                     priority_scale=priority_scale)
            importance = importance ** (1 - self.calc_epsilon(state_number))
        else:
            states, actions, rewards, new_states, terminal_flags = self.replay_buffer.get_minibatch(
                batch_size=self.batch_size, priority_scale=priority_scale)

        # Target DQN estimates q-vals for new states
        result_ids = []
        for state in new_states:
            result_ids.append(self.target_dqn.predict.remote(np.expand_dims(state, axis=0)))

        results = ray.get(result_ids)
        target_future_v = np.amax(np.array(results).squeeze(), axis=1)

        # Calculate targets (bellman equation)
        target_q = rewards + (gamma * target_future_v * (1 - terminal_flags))

        # Use targets to calculate loss (and use loss to calculate gradients)
        with tf.GradientTape() as tape:
            trainable_variables = ray.get(self.dqn.trainable_variables.remote())
            tape.watch(trainable_variables)

            predict_ids = []
            for state in states:
                predict_ids.append(self.dqn.call.remote(np.expand_dims(state, axis=0)))

            q_values = tf.squeeze(tf.stack(ray.get(predict_ids)))

            one_hot_actions = tf.keras.utils.to_categorical(actions, self.n_actions,
                                                            dtype=np.float32)  # using tf.one_hot causes strange errors
            Q = tf.reduce_sum(tf.multiply(q_values, one_hot_actions), axis=1)

            error = Q - target_q
            loss = tf.keras.losses.Huber(delta=1.35)(target_q, Q)

            if self.use_per:
                # Multiply the loss by importance, so that the gradient is also scaled.
                # The importance scale reduces bias against situataions that are sampled
                # more frequently.
                loss = tf.reduce_mean(loss * importance)

        model_gradients = tape.gradient(loss, trainable_variables)
        self.dqn.apply_gradients.remote(model_gradients, trainable_variables)

        if self.use_per:
            self.replay_buffer.set_priorities(indices, error)

        return float(loss.numpy()), error

    def save(self, folder_name, **kwargs):
        """Saves the Agent and all corresponding properties into a folder
        Arguments:
            folder_name: Folder in which to save the Agent
            **kwargs: Agent.save will also save any keyword arguments passed.  This is used for saving the state_number
        """

        # Create the folder for saving the agent
        if not os.path.isdir(folder_name):
            os.makedirs(folder_name)

        # Save DQN and target DQN
        self.DQN.save(folder_name + '/dqn.h5')
        self.target_dqn.save(folder_name + '/target_dqn.h5')

        # Save replay buffer
        self.replay_buffer.save(folder_name + '/replay-buffer')

        # Save meta
        with open(folder_name + '/meta.json', 'w+') as f:
            f.write(json.dumps({**{'buff_count': self.replay_buffer.count, 'buff_curr': self.replay_buffer.current},
                                **kwargs}))  # save replay_buffer information and any other information

    def load(self, folder_name, load_replay_buffer=True):
        """Load a previously saved Agent from a folder
        Arguments:
            folder_name: Folder from which to load the Agent
        Returns:
            All other saved attributes, e.g., state number
        """

        if not os.path.isdir(folder_name):
            raise ValueError(f'{folder_name} is not a valid directory')

        # Load DQNs
        self.DQN = tf.keras.models.load_model(folder_name + '/dqn.h5')
        self.target_dqn = tf.keras.models.load_model(folder_name + '/target_dqn.h5')
        self.optimizer = self.DQN.optimizer

        # Load replay buffer
        if load_replay_buffer:
            self.replay_buffer.load(folder_name + '/replay-buffer')

        # Load meta
        with open(folder_name + '/meta.json', 'r') as f:
            meta = json.load(f)

        if load_replay_buffer:
            self.replay_buffer.count = meta['buff_count']
            self.replay_buffer.current = meta['buff_curr']

        del meta['buff_count'], meta['buff_curr']  # we don't want to return this information
        return meta
