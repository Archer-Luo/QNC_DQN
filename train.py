import numpy as np
import NmodelDynamics
import ReplayBuffer
import Agent
import Network
import ray

ray.init()

if __name__ == "__main__":
    state_shape = (2,)
    n_actions = 2
    rb_size = 4000
    batch_size = 8
    eps_initial = 1
    eps_final = 0.1
    eps_final_state = 0.01
    eps_evaluation = 0.0
    eps_annealing_states = 4000
    replay_buffer_start_size = 1000
    max_states = 10000
    use_per = True
    max_episode = 50
    start_state = np.array([5, 5])
    h = np.array([3, 1])
    gamma = 0.995
    priority_scale = 0.7
    C = 32
    rho_list = [0.85]

    for rho in rho_list:
        network = NmodelDynamics.ProcessingNetwork.Nmodel_from_load(rho)

        current_state = np.copy(start_state)

        dqn = Network.Network.remote(n_actions)

        target_dqn = Network.Network.remote(n_actions)

        replay_buffer = ReplayBuffer.ReplayBuffer(rb_size, state_shape, use_per)

        agent = Agent.Agent(dqn, target_dqn, replay_buffer, n_actions, state_shape, batch_size, eps_initial,
                                   eps_final,
                                   eps_final_state, eps_evaluation, eps_annealing_states, replay_buffer_start_size,
                                   max_states,
                                   use_per)

        for t in range(max_states):
            if t % C == 0:
                agent.update_target_network()
            action = agent.get_action(t, current_state)
            # print('current state is %s, action %d is taken' % (current_state, action,))
            next_state = network.next_state_N1(current_state, action)
            reward = -((next_state - current_state) @ h)
            terminal = (t == max_states - 1)
            agent.add_experience(action, current_state, reward, terminal)
            if replay_buffer.count > replay_buffer_start_size:
                agent.learn(batch_size, gamma, t, priority_scale)
            current_state = next_state
            print(t)

        action_result = np.empty([50, 50])
        v_result = np.empty([50, 50])
        for a in range(50):
            for b in range(50):
                state = np.array([a, b])
                values = agent.dqn.predict(np.expand_dims(state, axis=0))
                action = values.argmax()
                action_result[a][b] = action + 1
                v_result[a][b] = np.amax(values)

        np.savetxt('rho{0}_gamma{1}_action'.format(rho, gamma), action_result, fmt='%i', delimiter=",")
        np.savetxt('rho{0}_gamma{1}_value'.format(rho, gamma), v_result, fmt='%10.5f', delimiter=",")
