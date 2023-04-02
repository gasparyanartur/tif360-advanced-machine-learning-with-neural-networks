import numpy as np
import random
import math
import h5py
import gmpy2
import json
import matplotlib.pyplot as plt

from src.gameboardClass import TGameBoard


def pack_q_table(q_table, actions):
    packed_data = {
        "q_table": {k: list(v) for k, v in q_table.items()},
        "actions": {k: v for k, v in actions.items()}
    }

    return packed_data


def unpack_q_table(strategy_file):
    with open(strategy_file, 'r') as f:
        packed_data = json.load(f)

    q_table = {int(k): np.array(v) for k, v in packed_data['q_table'].items()}    
    actions = {int(k): v for k, v in packed_data['actions'].items()}    

    return q_table, actions


# This file provides the skeleton structure for the classes TQAgent and TDQNAgent to be completed by you, the student.
# Locations starting with # TO BE COMPLETED BY STUDENT indicates missing code that should be written by you.


class TQAgent:
    # Agent for learning to play tetris using Q-learning
    def __init__(self, alpha, epsilon, episode_count, moving_avg_size=100):
        # Initialize training parameters
        self.alpha = alpha
        self.epsilon = epsilon
        self.episode = 0
        self.episode_count = episode_count

        self.rewards = []
        self.avg_rewards = []
        self.moving_avg_size = moving_avg_size

    def encode_state(self):
        enc_types = np.zeros(len(self.gameboard.tiles), dtype=int)
        enc_types[self.gameboard.cur_tile_type] = 1
        enc_board = (self.gameboard.board==1).flatten()
        occ = np.hstack((enc_types, enc_board))
        code = int(gmpy2.pack(occ.tolist(), 1))

        return code

    def fn_init(self, gameboard: TGameBoard, task_name='', load_strategy=False, save_strategy=True, save_plot=True, show_plot=True):
        self.episode_reward = 0
        self.gameboard = gameboard
        self.q_table = dict()
        self.actions = dict()

        self.enc_prev_state = None
        self.fn_read_state()

        self.task_name = task_name

        self.load_strategy = load_strategy
        self.save_strategy = save_strategy
        self.save_plot = save_plot
        self.show_plot = show_plot

        if load_strategy:
            self.fn_load_strategy(f'./src/params/params-{task_name}.json')

    def fn_load_strategy(self, strategy_file):
        self.q_table, self.actions = unpack_q_table(strategy_file)

    def fn_read_state(self):
        self.enc_state = self.encode_state()

    def fn_select_action(self):
        enc_state = self.enc_state

        if enc_state in self.q_table:
            q_values = self.q_table[enc_state]
            actions = self.actions[enc_state]

        else:
            actions = []
            curr_tile = self.gameboard.tiles[self.gameboard.cur_tile_type]
            n_orientations = len(curr_tile)

            for tile_orientation in range(n_orientations):
                width = len(curr_tile[tile_orientation])
                for tile_x in range(self.gameboard.N_col - width + 1):
                    actions.append((tile_x, tile_orientation))

            q_values = np.zeros(len(actions))
            self.q_table[enc_state] = q_values
            self.actions[enc_state] = actions

        self.i_prev_action = i_action = np.random.randint(len(actions)) \
            if (np.random.rand() < self.epsilon) \
            else np.argmax(q_values)
        tile_x, tile_orientation = actions[i_action]
        self.gameboard.fn_move(tile_x, tile_orientation)

    def fn_reinforce(self, enc_prev_state, reward):
        prev_q = self.q_table[enc_prev_state][self.i_prev_action]
        max_q = self.q_table[self.enc_state].max() if (self.enc_state in self.q_table) else 0

        next_q = prev_q + self.alpha * (reward + max_q - prev_q)
        self.q_table[enc_prev_state][self.i_prev_action] = next_q


    def fn_turn(self):
        if self.gameboard.gameover:
            self.episode += 1
            self.rewards.append(self.episode_reward)
            self.episode_reward = 0

            if self.episode % self.moving_avg_size == 0:
                avg_reward = np.mean(self.rewards[self.episode-self.moving_avg_size:self.episode])
                self.avg_rewards.append(avg_reward)
                print(f'Episode: {self.episode}/{self.episode_count}, Mean reward [T={self.moving_avg_size}]: {avg_reward}')

            if self.episode >= self.episode_count:
                self.fn_save_metrics()
                raise SystemExit(0)

            else:
                self.gameboard.fn_restart()

        else:
            self.fn_select_action()
            enc_prev_state = self.enc_state
            reward = self.gameboard.fn_drop()
            self.episode_reward += reward
            self.fn_reinforce(enc_prev_state, reward)

    def fn_save_metrics(self):
        if self.save_strategy:
            packed_data = pack_q_table(self.q_table, self.actions)

            with open(f'./src/params/params-{self.task_name}.json', 'w') as f:
                json.dump(packed_data, f)

        plt.plot(np.arange(len(self.rewards)), self.rewards)
        plt.plot(self.moving_avg_size*np.arange(len(self.avg_rewards)), self.avg_rewards)
        plt.legend(['Rewards', f'Moving average [T={self.moving_avg_size}]'])
        plt.title(f'Rewards and moving average rewards over episode, task {self.task_name}')
        plt.xlabel(r'Episode $E$')
        plt.ylabel(r'Reward $R$')
        
        if self.save_plot:
            plt.savefig(f'./src/plots/fig-{self.task_name}.png')

        if self.show_plot:
            plt.show()


class TDQNAgent:
    # Agent for learning to play tetris using Q-learning
    def __init__(self, alpha, epsilon, epsilon_scale, replay_buffer_size, batch_size, sync_target_episode_count, episode_count):
        # Initialize training parameters
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_scale = epsilon_scale
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        self.sync_target_episode_count = sync_target_episode_count
        self.episode = 0
        self.episode_count = episode_count

    def fn_init(self, gameboard):
        self.gameboard = gameboard
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # In this function you could set up and initialize the states, actions, the Q-networks (one for calculating actions and one target network), experience replay buffer and storage for the rewards
        # You can use any framework for constructing the networks, for example pytorch or tensorflow
        # This function should not return a value, store Q network etc as attributes of self

        # Useful variables:
        # 'gameboard.N_row' number of rows in gameboard
        # 'gameboard.N_col' number of columns in gameboard
        # 'len(gameboard.tiles)' number of different tiles
        # 'self.alpha' the learning rate for stochastic gradient descent
        # 'self.episode_count' the total number of episodes in the training
        # 'self.replay_buffer_size' the number of quadruplets stored in the experience replay buffer

    def fn_load_strategy(self, strategy_file):
        pass
        # TO BE COMPLETED BY STUDENT
        # Here you can load the Q-network (to Q-network of self) from the strategy_file

    def fn_read_state(self):
        pass
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # In this function you could calculate the current state of the gane board
        # You can for example represent the state as a copy of the game board and the identifier of the current tile
        # This function should not return a value, store the state as an attribute of self

        # Useful variables:
        # 'self.gameboard.N_row' number of rows in gameboard
        # 'self.gameboard.N_col' number of columns in gameboard
        # 'self.gameboard.board[index_row,index_col]' table indicating if row 'index_row' and column 'index_col' is occupied (+1) or free (-1)
        # 'self.gameboard.cur_tile_type' identifier of the current tile that should be placed on the game board (integer between 0 and len(self.gameboard.tiles))

    def fn_select_action(self):
        pass
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # Choose and execute an action, based on the output of the Q-network for the current state, or random if epsilon greedy
        # This function should not return a value, store the action as an attribute of self and exectute the action by moving the tile to the desired position and orientation

        # Useful variables:
        # 'self.epsilon' parameter epsilon in epsilon-greedy policy
        # 'self.epsilon_scale' parameter for the scale of the episode number where epsilon_N changes from unity to epsilon

        # Useful functions
        # 'self.gameboard.fn_move(tile_x,tile_orientation)' use this function to execute the selected action
        # The input argument 'tile_x' contains the column of the tile (0 <= tile_x < self.gameboard.N_col)
        # The input argument 'tile_orientation' contains the number of 90 degree rotations of the tile (0 < tile_orientation < # of non-degenerate rotations)
        # The function returns 1 if the action is not valid and 0 otherwise
        # You can use this function to map out which actions are valid or not

    def fn_reinforce(self, batch):
        pass
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # Update the Q network using a batch of quadruplets (old state, last action, last reward, new state)
        # Calculate the loss function by first, for each old state, use the Q-network to calculate the values Q(s_old,a), i.e. the estimate of the future reward for all actions a
        # Then repeat for the target network to calculate the value \hat Q(s_new,a) of the new state (use \hat Q=0 if the new state is terminal)
        # This function should not return a value, the Q table is stored as an attribute of self

        # Useful variables:
        # The input argument 'batch' contains a sample of quadruplets used to update the Q-network

    def fn_turn(self):
        if self.gameboard.gameover:
            self.episode += 1
            if self.episode % 100 == 0:
                print('episode '+str(self.episode)+'/'+str(self.episode_count)+' (reward: ',
                      str(np.sum(self.reward_tots[range(self.episode-100, self.episode)])), ')')
            if self.episode % 1000 == 0:
                saveEpisodes = [1000, 2000, 5000, 10000, 20000,
                                50000, 100000, 200000, 500000, 1000000]
                if self.episode in saveEpisodes:
                    pass
                    # TO BE COMPLETED BY STUDENT
                    # Here you can save the rewards and the Q-network to data files
            if self.episode >= self.episode_count:
                raise SystemExit(0)
            else:
                if (len(self.exp_buffer) >= self.replay_buffer_size) and ((self.episode % self.sync_target_episode_count) == 0):
                    pass
                    # TO BE COMPLETED BY STUDENT
                    # Here you should write line(s) to copy the current network to the target network
                self.gameboard.fn_restart()
        else:
            # Select and execute action (move the tile to the desired column and orientation)
            self.fn_select_action()
            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to copy the old state into the variable 'old_state' which is later stored in the ecperience replay buffer

            # Drop the tile on the game board
            reward = self.gameboard.fn_drop()

            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to add the current reward to the total reward for the current episode, so you can save it to disk later

            # Read the new state
            self.fn_read_state()

            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to store the state in the experience replay buffer

            if len(self.exp_buffer) >= self.replay_buffer_size:
                # TO BE COMPLETED BY STUDENT
                # Here you should write line(s) to create a variable 'batch' containing 'self.batch_size' quadruplets
                self.fn_reinforce(batch)


class THumanAgent:
    def fn_init(self, gameboard):
        self.episode = 0
        self.reward_tots = [0]
        self.gameboard = gameboard

    def fn_read_state(self):
        pass

    def fn_turn(self, pygame):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit(0)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.reward_tots = [0]
                    self.gameboard.fn_restart()
                if not self.gameboard.gameover:
                    if event.key == pygame.K_UP:
                        self.gameboard.fn_move(self.gameboard.tile_x, (self.gameboard.tile_orientation+1) % len(
                            self.gameboard.tiles[self.gameboard.cur_tile_type]))
                    if event.key == pygame.K_LEFT:
                        self.gameboard.fn_move(
                            self.gameboard.tile_x-1, self.gameboard.tile_orientation)
                    if event.key == pygame.K_RIGHT:
                        self.gameboard.fn_move(
                            self.gameboard.tile_x+1, self.gameboard.tile_orientation)
                    if (event.key == pygame.K_DOWN) or (event.key == pygame.K_SPACE):
                        self.reward_tots[self.episode] += self.gameboard.fn_drop()
