import tensorflow as tf
import numpy as np
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import (Add, Conv2D, Dense, Flatten, Input,
                                     Lambda, Subtract)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop

from tensorflow.python import keras

import NmodelDynamics
import ReplayBuffer
import Agent

if __name__ == "__main__":
    input_shape = (2,)
    n_actions = 2
    rb_size = 1000
    batch_size = 8
    eps_initial = 1
    eps_final = 0.1
    eps_final_state = 0.01
    eps_evaluation = 0.0
    eps_annealing_states = 9000
    replay_buffer_start_size = 8
    max_states = 20000
    use_per = True
    max_time_step = 1000
    max_episode = 100
    start_state = np.array([0, 0])
    h = np.array([3, 1])
    gamma = 0.99
    priority_scale = 0.7
    C = 4

    for episode in range(max_episode):
        network = NmodelDynamics.ProcessingNetwork.Nmodel_from_load(0.95)

        current_state = np.copy(start_state)

        dqn = keras.Sequential([
            keras.layers.Dense(10, input_dim=2, activation='relu'),
            keras.layers.Dense(n_actions)
        ])
        dqn.compile(loss=tf.keras.losses.Huber(), optimizer='adam')

        target_dqn = keras.Sequential([
            keras.layers.Dense(10, input_dim=2, activation='relu'),
            keras.layers.Dense(n_actions)
        ])
        target_dqn.compile(loss=tf.keras.losses.Huber(), optimizer='adam')

        replay_buffer = ReplayBuffer.ReplayBuffer(rb_size, input_shape, use_per)

        agent = Agent.Agent(dqn, target_dqn, replay_buffer, n_actions, input_shape, batch_size, eps_initial, eps_final,
                            eps_final_state, eps_evaluation, eps_annealing_states, replay_buffer_start_size, max_states,
                            use_per)

        for t in range(max_time_step):
            if t % C == 0:
                agent.update_target_network()
            action = agent.get_action(t, current_state)
            next_state = network.next_state_N1(current_state, action).T
            reward = -(next_state @ h)
            terminal = (t == max_time_step - 1)
            agent.add_experience(action, current_state, reward, terminal)
            if replay_buffer.count > replay_buffer_start_size:
                agent.learn(batch_size, gamma, t, priority_scale)






