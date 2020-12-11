import numpy as np
import time
from PID import PID
from states_drone import Actions, StateDiscrete
from game_of_drones import GameOfDrone
from collections import defaultdict
import sys
import dill
import matplotlib.pyplot as plt
from world_settings import *
from time import perf_counter
import pandas as pd


def _get_number_of_actions():
    """Return total number of possible actions."""
    actions = [a.value for a in Actions]
    return max(actions)


class QLearning:
    """Class implementing Q-learning agent."""

    def __init__(self):
        """Defining instance variables for Q-learning agent."""
        self.epsilon = 0.9
        self.pid_parameter = {'kp': 0.5, 'ki': 0.5, 'kd': 0.5}
        self.episodes = 1000
        self.take_off_procedure = 5 * 3
        self.alpha = 0.01
        self.gamma = 0.99
        self.number_of_actions = _get_number_of_actions()
        self.Q = None
        self.policy = np.zeros(STATE_VALUE_BUCKETS, dtype=Actions)
        self.total_reward = []
        self.render_every = 100

        self.env = GameOfDrone()
        self.pid = PID(self.pid_parameter['kp'], self.pid_parameter['ki'], self.pid_parameter['kd'])

    def epsilon_greedy(self, states, number_of_actions):
        """Epsilon-greedy policy policy to ensure exploration."""
        if np.random.rand() < self.epsilon:
            action = np.argmax(np.array([self.Q[action][states] for action in Actions]))
            return Actions(action)
        else:
            action = np.random.randint(0, number_of_actions)
            return Actions(action)

    def apply_q_learning(self):
        """Apply Q-learning to the environment."""
        # Initialize PID controller, environment and states

        states = self.env.reset()

        # Discretize the state space
        get_discrete_states = StateDiscrete(states.dist_to_target,
                                            states.distance_vector.optimal_heading - states.angle,
                                            states.total_velocity)
        discrete_states = get_discrete_states.discrete()

        # Initialize the Q table
        if self.Q is None:
            self.Q = dict()
            for action in Actions:
                self.Q[action] = np.zeros(get_discrete_states.state_buckets)

        # Start the training
        for episode in range(self.episodes):
            states = self.env.reset()

            # Set a flag for when environment to be rendered
            if episode % self.render_every == 0:
                print(episode)
                do_rendering = True
            else:
                do_rendering = False
            current_step = 0
            episode_reward = 0
            while True:
                # Setting up usage of PID controller

                self.pid.update_targets(states)
                if current_step % 3 == 0 and current_step <= self.take_off_procedure:
                    action = self.epsilon_greedy(discrete_states, self.number_of_actions)
                    self.pid.set_target_position(states.drone_position)
                elif current_step <= self.take_off_procedure:
                    action = self.pid.pid_control(states)
                elif current_step % 3 == 0 and current_step >= self.take_off_procedure:
                    action = self.pid.pid_control(states)
                else:
                    action = self.epsilon_greedy(discrete_states, self.number_of_actions)

                # Taking the action in the environment
                next_states, reward, episode_done = self.env.step(action)
                episode_reward += reward
                get_next_discrete_states = StateDiscrete(next_states.dist_to_target,
                                                         next_states.distance_vector.optimal_heading - next_states.angle,
                                                         next_states.total_velocity)
                next_discrete_states = get_next_discrete_states.discrete()

                # Q table update

                self.Q[action][discrete_states] += self.alpha * (
                        reward + (self.gamma * np.argmax(
                    np.array([self.Q[action][next_discrete_states] for action in Actions])) -
                                  self.Q[action][discrete_states]))

                if do_rendering:
                    self.env.render(mode='rgb_array')

                if episode_done:
                    self.env.close()
                    break
                current_step += 1

                states = next_states
                discrete_states = next_discrete_states

            self.total_reward.append(episode_reward)

        self.backup_Q_table()
        return self.Q, self.policy, self.total_reward

    def test_agent(self):
        for _ in range(20):
            state_continuous = self.env.reset()

            step = 0
            while True:
                discrete = StateDiscrete(state_continuous.dist_to_target,
                                         state_continuous.distance_vector.optimal_heading - state_continuous.angle,
                                         state_continuous.total_velocity)
                state = discrete.discrete()
                if step % 4 == 0 and step <= self.take_off_procedure:

                    self.pid.set_target_position(state_continuous.drone_position)
                    action = self.policy[state]

                elif step <= self.take_off_procedure:
                    action = self.pid.pid_control(state_continuous)
                elif step % 3 == 0 and step >= self.take_off_procedure:
                    action = self.pid.pid_control(state_continuous)
                else:
                    self.pid.update_targets(state_continuous)
                    action = self.policy[state]
                print("Applying: ", action)
                ns_continuous, reward, done = self.env.step(action)
                # episode.append((state, action.value, reward))

                # time.sleep(0.01)
                self.env.render(mode='rgb_array')

                if done:
                    self.env.close()
                    break
                state_continuous = ns_continuous
                step += 1

    def backup_Q_table(self):
        # write python dict to a file
        output = open('Q_table_backup/QLearn_1.pkl', 'wb')
        dill.dump(self.Q, output)
        output.close()
        output = open('policy_backup/QLearn_1.pkl', 'wb')
        dill.dump(self.policy, output)
        output.close()
        print('Backup successful')

    def restore_Q_table(self):
        # read python dict back from the file
        pkl_file = open('Q_table_backup/QLearn_1.pkl', 'rb')
        self.Q = dill.load(pkl_file)
        pkl_file.close()

        print('Restoring successful')

    def restore_policy(self):
        """
        Restoring policy from Q_Table.
        :return:
        """
        for i in range(STATE_VALUE_BUCKETS[0]):
            for j in range(STATE_VALUE_BUCKETS[1]):
                for k in range(STATE_VALUE_BUCKETS[2]):
                    self.policy[i, j, k] = Actions(np.argmax(np.array([self.Q[action][i, j, k] for action in Actions])))


class SARSA:
    """Class implementing an SARSA agent."""

    def __init__(self):
        self.epsilon = 0.9
        self.pid_parameter = {'kp': 0.5, 'ki': 0.5, 'kd': 0.5}
        self.episodes = 500
        self.take_off_procedure = 50 * 4
        self.env_max_steps = 1000  # to be checked
        self.alpha = 0.01
        self.gamma = 0.9
        self.number_of_actions = _get_number_of_actions()
        self.render_every = 100

        # Initialize PID controller, environment and states
        self.pid = PID(self.pid_parameter['kp'], self.pid_parameter['ki'], self.pid_parameter['kd'])
        self.env = GameOfDrone()
        states = self.env.reset()

        # Discretize the state space
        get_discrete_states = StateDiscrete(states.dist_to_target,
                                            states.distance_vector.optimal_heading - states.angle,
                                            states.total_velocity)
        self.discrete_states = get_discrete_states.discretize()

        # Initialize the Q table
        self.Q = dict()
        for a in Actions:
            self.Q[a] = np.zeros(get_discrete_states.state_buckets)
        self.policy = np.zeros(get_discrete_states.state_buckets, dtype=Actions)
        for i in range(get_discrete_states.state_buckets[0]):
            for j in range(get_discrete_states.state_buckets[1]):
                for k in range(get_discrete_states.state_buckets[2]):
                    self.policy[i, j, k] = Actions.ENGINES_OFF

    def epsilon_greedy(self, states, number_of_actions):
        if np.random.rand() < self.epsilon:
            # action = np.argmax(np.array([self.Q[a][states] for a in Actions]))
            print(states)
            action = self.policy[states]
            return action
        else:
            action = np.random.randint(0, number_of_actions)
            return Actions(action)

    def train_sarsa(self):

        # Start the training
        for episode in range(self.episodes):
            states = self.env.reset()
            print(states.angle, states.optimal_heading_to_target)
            # Set a flag for when environment to be rendered
            if episode % self.render_every == 0:
                print(episode)
                do_rendering = True
            else:
                do_rendering = False
            episode_done = False
            current_step = 0
            while not episode_done and current_step < self.env_max_steps:
                # Setting when/how to use the PID controller
                if current_step % 4 == 0 and current_step <= self.take_off_procedure:
                    action = self.epsilon_greedy(self.discrete_states, self.number_of_actions)
                    self.pid.set_target_position(states.drone_position)
                elif current_step <= self.take_off_procedure:
                    action = self.pid.pid_control(states)
                elif current_step % 4 == 0 and current_step >= self.take_off_procedure:
                    action = self.pid.pid_control(states)
                else:
                    action = self.epsilon_greedy(self.discrete_states, self.number_of_actions)
                    # self.pid.update_targets(states)
                if current_step % 3:
                    self.pid.update_targets(states)
                print("Applying following action: ", action)
                # Taking the action in the environment
                next_states, reward, episode_done = self.env.step(action)
                get_next_discrete_states = StateDiscrete(next_states.dist_to_target,
                                                         next_states.distance_vector.optimal_heading - states.angle,
                                                         next_states.total_velocity)
                next_discrete_states = get_next_discrete_states.discretize()

                # Q table update print(next_discrete_states, self.policy[next_discrete_states], self.discrete_states,
                # states.angle, states.optimal_heading_to_target - states.angle) print("current angle: ", np.rad2deg(
                # states.angle * 2 * np.pi), "optimal heading", np.rad2deg(states.optimal_heading_to_target * 2 *
                # np.pi))
                self.Q[action][self.discrete_states] += self.alpha * (
                        reward + (self.gamma * self.Q[self.policy[next_discrete_states]][next_discrete_states] -
                                  self.Q[action][self.discrete_states]))
                # Update policy
                self.policy[self.discrete_states] = action
                temp_max_Q = np.abs(self.Q[action][self.discrete_states])

                for action_pol in Actions:
                    if np.abs(self.Q[action_pol][self.discrete_states]) > temp_max_Q:
                        self.policy[self.discrete_states] = action
                        temp_max_Q = np.abs(self.Q[action_pol][self.discrete_states])

                # print("New best policy for state: ",self.discrete_states, self.policy[self.discrete_states],
                # episode_done, self.take_off_procedure - current_step)
                time.sleep(0.01)
                if do_rendering:
                    self.env.render(mode='rgb_array')

                if episode_done:
                    self.env.close()
                    break
                current_step += 1

                states = next_states
                self.discrete_states = next_discrete_states
            if episode % 20:
                self.backup_Q_table()

        self.backup_Q_table()

    def test_agent(self):
        for _ in range(20):
            state_continuous = self.env.reset()
            for step in range(self.env_max_steps):
                discrete = StateDiscrete(state_continuous.dist_to_target,
                                         state_continuous.distance_vector.optimal_heading - state_continuous.angle,
                                         state_continuous.total_velocity)
                state = discrete.discretize()
                if step % 4 == 0 and step <= self.take_off_procedure:

                    self.pid.set_target_position(state_continuous.drone_position)
                    action = self.policy[state]

                elif step <= self.take_off_procedure:
                    action = self.pid.pid_control(state_continuous)
                elif step % 3 == 0 and step >= self.take_off_procedure:
                    action = self.pid.pid_control(state_continuous)
                else:
                    self.pid.update_targets(state_continuous)
                    action = self.policy[state]

                print("Applying sarsa: ", action)
                ns_continuous, reward, done = self.env.step(action)
                # episode.append((state, action.value, reward))

                self.env.render(mode='rgb_array')

                if done:
                    self.env.close()
                    break
                state_continuous = ns_continuous

    def backup_Q_table(self):
        # write python dict to a file
        output = open('Q_table_backup/sarsa_1.pkl', 'wb')
        dill.dump(self.Q, output)
        output.close()
        output = open('policy_backup/sarsa_1.pkl', 'wb')
        dill.dump(self.policy, output)
        output.close()
        print('Backup successful')

    def restore_Q_table(self):
        # read python dict back from the file
        pkl_file = open('Q_table_backup/sarsa_1.pkl', 'rb')
        self.Q = dill.load(pkl_file)
        pkl_file.close()
        policy_file = open('policy_backup/sarsa_1.pkl', 'rb')
        self.policy = dill.load(policy_file)
        policy_file.close()
        print('Restoring successful')


class MonteCarlo:
    """
    Class to implement Monte Carlo Control Algorithm with epsilon-greedy policy
    """

    def __init__(self):
        # self.env = gym.make('GameOfDrones-v0').unwrapped
        self.env = GameOfDrone()
        self.train_episodes = 100000
        self.nA = self.env.action_space.n
        self.Q = defaultdict(lambda: np.full(self.nA, -10000.0, dtype=float))
        self.discount_factor = 0.9
        self.epsilon = 0.1
        self.env_max_steps = 500
        self.render = False
        self.test_episodes = 1000
        self.rewards = []
        self.target_reach_count = 0
        self.output_file_q_table = 'Q_table_backup/200000/Q_wind_' + str(self.env.wind_disturbance) + '_controller_' \
                                   + str(self.env.controller_disturbance) + '_df_' + str(self.discount_factor) \
                                   + '_ep_' + str(self.epsilon)
        self.output_file_rewards = 'Q_table_backup/200000/rew_wind_' + str(self.env.wind_disturbance) + '_controller_' \
                                   + str(self.env.controller_disturbance) + '_df_' + str(self.discount_factor) \
                                   + '_ep_' + str(self.epsilon)
        self.restore_file_q_table = 'Q_table_backup/200000/Q_wind_' + str(self.env.wind_disturbance) + '_controller_' \
                                    + str(self.env.controller_disturbance) + '_df_' + str(self.discount_factor) \
                                    + '_ep_' + str(self.epsilon)
        self.restore_file_rewards = 'Q_table_backup/200000/rew_wind_' + str(self.env.wind_disturbance) + '_controller_' \
                                    + str(self.env.controller_disturbance) + '_df_' + str(self.discount_factor) \
                                    + '_ep_' + str(self.epsilon)

    def make_epsilon_greedy_policy(self):
        """
        Creates an epsilon-greedy policy based on a given Q-function and epsilon.
        Returns:
            A function that takes the observation (state) as an argument and returns
            the probabilities for each action in the form of a numpy array of length nA.
        """

        def policy_fn(observation):
            prob = np.ones(self.nA, dtype=float) * self.epsilon / self.nA
            optimal_action = np.argmax(self.Q[tuple(observation)])
            prob[optimal_action] += (1.0 - self.epsilon)
            return prob

        return policy_fn

    def mc_control_train(self):
        """
        Monte Carlo Control using Epsilon-Greedy policies.
        Finds an optimal epsilon-greedy policy.
        Returns:
            A tuple (Q, policy).
            Q is a dictionary mapping state -> action values.
            policy is a function that takes an observation as an argument and returns
            action probabilities
        """

        # Keeps track of sum and count of returns for each state
        returns_sum = defaultdict(float)
        returns_count = defaultdict(float)

        # A nested dictionary that maps state -> (action -> action-value).
        policy_fn = self.make_epsilon_greedy_policy()
        counter = perf_counter()

        for i_episode in range(1, self.train_episodes + 1):
            episode = []
            self.env.reset()
            current_state_space = self.env.state_space
            state = StateDiscrete(current_state_space.dist_to_target,
                                  current_state_space.optimal_heading_to_target - current_state_space.angle,
                                  current_state_space.total_velocity).discretize()

            if i_episode % 5 == 0:
                print("\rEpisode {}/{}.".format(i_episode, self.train_episodes), end="")
                print('no of states explored:', len(self.Q.keys()))
                print('Time taken:', (perf_counter() - counter) / 60, 'minutes')
                counter = perf_counter()
                print('Target reached count:', self.target_reach_count)
                # self.render = True
                sys.stdout.flush()
            else:
                self.render = False

            reward_episode = 0.0
            for step in range(self.env_max_steps):
                if state in self.Q:
                    prob_values = policy_fn(state)
                    action = np.random.choice(np.arange(self.nA), p=prob_values)
                else:
                    action = self.env.action_space.sample()

                _, reward, done, ns_state_space = self.env.step(action)
                next_state = StateDiscrete(ns_state_space.dist_to_target,
                                           ns_state_space.distance_vector.optimal_heading - ns_state_space.angle,
                                           ns_state_space.total_velocity).discretize()
                episode.append((next_state, action, reward))
                reward_episode += reward

                if self.render:
                    self.env.render(mode='rgb_array')

                if done:
                    if self.env.state_space.target_reached:
                        self.target_reach_count += 1
                    self.env.close()
                    break
                state = next_state

            self.rewards.append(reward_episode)
            sa_in_episode = set([(x[0], x[1]) for x in episode])
            for state, action in sa_in_episode:
                sa_pair = (state, action)
                # Find the first occurrence of the (state, action) pair in the episode
                first_occurrence_idx = next(i for i, x in enumerate(episode)
                                            if x[0] == state and x[1] == action)
                # Sum up all rewards since the first occurrence
                G = sum([x[2] * (self.discount_factor ** i) for i, x in enumerate(episode[first_occurrence_idx:])])
                # Calculate average return for this state over all sampled episodes
                returns_sum[sa_pair] += G
                returns_count[sa_pair] += 1.0
                self.Q[state][action] = returns_sum[sa_pair] / returns_count[sa_pair]

            if i_episode % 500 == 0:
                self.backup_Q_table()

        return self.Q, self.rewards

    def test_agent(self):
        for ep in range(self.test_episodes):
            reward_episode = 0.0
            self.env.reset()
            state = StateDiscrete(self.env.state_space.dist_to_target,
                                  self.env.state_space.distance_vector.optimal_heading - self.env.state_space.angle,
                                  self.env.state_space.total_velocity).discretize()

            if ep % 10 == 0:
                print("\rEpisode {}/{}.".format(ep, self.test_episodes), end="")
                self.render = True
                sys.stdout.flush()
            else:
                self.render = False

            for step in range(1000):
                if state in self.Q:
                    action = Actions(np.argmax(self.Q[tuple(state)]))
                else:
                    action = Actions(self.env.action_space.sample())
                    print('State not found in Q')

                _, reward, done, state_space = self.env.step(action)
                reward_episode += reward

                if self.render:
                    time.sleep(0.01)
                    self.env.render(mode='rgb_array')

                if done:
                    if state_space.target_reached:
                        self.target_reach_count += 1
                    self.env.close()
                    break
                next_state = StateDiscrete(state_space.dist_to_target,
                                           state_space.distance_vector.optimal_heading - state_space.angle,
                                           state_space.total_velocity).discretize()
                state = next_state

            self.rewards.append(reward_episode)
        return self.rewards, self.target_reach_count / self.test_episodes

    def backup_Q_table(self):
        # write python dict to a file
        output_q_table = open(self.output_file_q_table, 'wb')
        dill.dump(self.Q, output_q_table)
        output_q_table.close()
        print('Backup Q-table successful')
        output_reward = open(self.output_file_rewards, 'wb')
        dill.dump(self.rewards, output_reward)
        output_reward.close()
        print('Backup rewards successful')

    def restore_Q_table(self):
        # read python dict back from the file
        pkl_file = open(self.restore_file_q_table, 'rb')
        self.Q = dill.load(pkl_file)
        pkl_file.close()

        rew_file = open(self.restore_file_rewards, 'rb')
        self.rewards = dill.load(rew_file)
        rew_file.close()
        print('Restoration successful')

    def plot_analysis(self):
        rew_df = pd.DataFrame(self.rewards, columns=['rewards'])
        # Rolling mean
        close_px = rew_df['rewards']
        moving_average = close_px.rolling(window=1000).mean()
        plt.figure(figsize=(12, 6))
        close_px.plot(label='Episodic rewards')
        moving_average.plot(label='moving average reward')
        plt.xlabel('Episodes', fontsize=15)
        plt.ylabel('Reward/Episode', fontsize=15)
        plt.title('Rewards over episodes - Monte Carlo Control')
        plt.legend()
        plt.savefig('flappy_bird_mc.pdf')
        plt.show()


if __name__ == "__main__":
    train = True
    train_new = True
    agent = 'MC'  # Choose agent between ['QLearning','SARSA', 'MC']

    if agent in 'SARSA':
        print("Applying SARSA !!")
        sarsa = SARSA()
        if train:
            sarsa.train_sarsa()
        else:
            sarsa.restore_Q_table()
            sarsa.test_agent()
    elif agent in 'MC':
        print("Applying Monte Carlo !!")
        monte_carlo_agent = MonteCarlo()
        if train:
            if train_new:
                monte_carlo_agent.mc_control_train()
            else:
                monte_carlo_agent.restore_Q_table()
                monte_carlo_agent.mc_control_train()
        else:
            monte_carlo_agent.restore_Q_table()
            monte_carlo_agent.plot_analysis()
    elif agent in 'QLearning':
        print("Applying QLearning")
        ql_agent = QLearning()
        if train:
            Q, policy, total_reward = ql_agent.apply_q_learning()
            # print(Q)
            plt.figure(figsize=(15, 10))
            plt.plot(total_reward, 'g')
            plt.xlabel('Reward')
            plt.ylabel('Episode')
            plt.show()
        else:
            ql_agent.restore_Q_table()
            ql_agent.restore_policy()
            ql_agent.test_agent()

    else:
        print("Agent does not exists !!")
