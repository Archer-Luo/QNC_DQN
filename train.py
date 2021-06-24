import math

import tensorflow as tf
import numpy as np
from tensorflow.python import keras

import NmodelDynamics
import ReplayBuffer
import Agent

if __name__ == "__main__":
    state_shape = (2,)
    n_actions = 2

    rho_list = [0.8]
    h = np.array([3, 1])
    start_state = np.array([5, 5])
    gamma = 0.995

    priority_scale = 0.7
    offset = 0.1

    use_per = True
    max_episode = 50
    max_states = 2 ** 14
    batch_size = 2 ** 2
    rb_size = max_states
    eps_initial = 1
    eps_final = 0.1
    eps_final_state = 0.1
    eps_evaluation = 0.0
    C = 256
    replay_buffer_start_size = batch_size
    eps_annealing_states = int(max_states / 5)

    for rho in rho_list:
        network = NmodelDynamics.ProcessingNetwork.Nmodel_from_load(rho)

        current_state = np.copy(start_state)

        dqn = keras.Sequential([
            keras.layers.Dense(10, input_dim=2, activation='tanh'),
            keras.layers.Dense(10, activation='tanh'),
            keras.layers.Dense(10, activation='tanh'),
            keras.layers.Dense(n_actions)
        ])
        dqn.compile(loss=tf.keras.losses.Huber(delta=1.35), optimizer=tf.keras.optimizers.Adam(learning_rate=0.00025, epsilon=0.001))

        target_dqn = keras.Sequential([
            keras.layers.Dense(10, input_dim=2, activation='tanh'),
            keras.layers.Dense(10, activation='tanh'),
            keras.layers.Dense(10, activation='tanh'),
            keras.layers.Dense(n_actions)
        ])
        target_dqn.compile(loss=tf.keras.losses.Huber(delta=1.35), optimizer=tf.keras.optimizers.Adam(learning_rate=0.00025, epsilon=0.001))

        replay_buffer = ReplayBuffer.ReplayBuffer(rb_size, state_shape, use_per, offset)

        agent = Agent.Agent(dqn, target_dqn, replay_buffer, n_actions, state_shape, batch_size, eps_initial, eps_final,
                            eps_final_state, eps_evaluation, eps_annealing_states, replay_buffer_start_size, max_states,
                            use_per)

        for t in range(max_states):
            if t % C == 0:
                agent.update_target_network()
            action = agent.get_action(t, current_state)
            # print('current state is %s, action %d is taken' % (current_state, action,))
            next_state = network.next_state_N1(current_state, action)
            reward = -(current_state @ h)
            terminal = (t == max_states - 1)
            agent.add_experience(action, current_state, reward, terminal)
            if replay_buffer.count > replay_buffer_start_size:
                loss, error = agent.learn(batch_size, gamma, t, priority_scale)
                print(loss)
            current_state = next_state
            print(t)

        action_result = np.empty([50, 50])
        v_result = np.empty([50, 50])
        for a in range(50):
            for b in range(50):
                state = np.array([a, b])
                values = agent.dqn.predict(np.expand_dims(state, axis=0)).squeeze()
                action_result[a][b] = np.argmax(values) + 1
                v_result[a][b] = np.amax(values)

        np.savetxt('rho{0}_gamma{1}_action'.format(rho, gamma), action_result, fmt='%i', delimiter=",")
        np.savetxt('rho{0}_gamma{1}_value'.format(rho, gamma), v_result, fmt='%10.5f', delimiter=",")
