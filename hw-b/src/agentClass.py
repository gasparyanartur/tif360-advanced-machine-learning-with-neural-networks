import numpy as np
import random
import math
import h5py
import gmpy2
import json
import matplotlib.pyplot as plt

from src.gameboardClass import TGameBoard

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
rng = np.random.default_rng(69420)

class SmallStateAutoEncoder(nn.Module):
    def __init__(self, d_L, d_H) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 2, padding=1),
            nn.ReLU(),
            nn.Flatten(1),   # dim 
            nn.Linear(400, d_H),
            nn.ReLU(),
            nn.Linear(d_H, d_L),
            nn.ReLU(),
        )        

        self.decoder = nn.Sequential(
            nn.Linear(d_L, d_H),
            nn.ReLU(),
            nn.Linear(d_H, 256),
            nn.ReLU(),
            nn.Unflatten(1, (16, 4, 4)),
            nn.ConvTranspose2d(16, 1, 1, stride=1, padding=0),
        )


    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return torch.tanh(x)


class LargeStateAutoEncoder(nn.Module):
    def __init__(self, d_L, d_H) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 2, padding=1),
            nn.ReLU(),
            nn.Flatten(1),
            nn.Linear(5184, d_H),
            nn.ReLU(),
            nn.Linear(d_H, d_H),
            nn.ReLU(),
            nn.Linear(d_H, d_L),
            nn.ReLU(),
        )        

        self.decoder = nn.Sequential(
            nn.Linear(d_L, d_H),
            nn.ReLU(),
            nn.Linear(d_H, d_H),
            nn.ReLU(),
            nn.Linear(d_H, 5184),
            nn.ReLU(),
            nn.Unflatten(1, (64, 9, 9)),
            nn.ConvTranspose2d(64, 32, 1, stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 2, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return torch.tanh(x)
        

class SmallTileEncoder:
    def __init__(self):
        self.binary = -1 + 2*torch.tensor([
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 1, 1],
            [0, 1, 0, 0],
            [0, 1, 0, 1],
            [0, 1, 1, 0],
            [0, 1, 1, 1],
            [1, 0, 0, 0],
            [1, 0, 0, 1],
            [1, 0, 1, 0],
            [1, 0, 1, 1],
            [1, 1, 0, 0],
            [1, 1, 0, 1],
            [1, 1, 1, 0],
            [1, 1, 1, 1],
        ])
        self.encoding = {
            (0, 0): 0,
            (0, 1): 1,
            (1, 0): 2,
            (1, 1): 3,
            (2, 0): 4,
            (2, 1): 5,
            (2, 2): 6,
            (2, 3): 7,
            (3, 0): 8
        }

        self._keys = list(self.encoding.keys())

    def encode_tile(self, tile_type, tile_orient, to_binary=True):
        enc = self.encoding[tile_type, tile_orient]

        if to_binary:
            enc = self.binary[enc]

        return enc

    def keys(self):
        return self._keys

class LargeTileEncoder:
    def __init__(self):
        self.binary = -1 + 2*torch.tensor([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 1],
            [0, 0, 1, 1, 0],
            [0, 0, 1, 1, 1],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 1],
            [0, 1, 0, 1, 0],
            [0, 1, 0, 1, 1],
            [0, 1, 1, 0, 0],
            [0, 1, 1, 0, 1],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 1],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 1, 0],
            [1, 0, 0, 1, 1],
            [1, 0, 1, 0, 0],
            [1, 0, 1, 0, 1],
            [1, 0, 1, 1, 0],
            [1, 0, 1, 1, 1],
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 1],
            [1, 1, 0, 1, 0],
            [1, 1, 0, 1, 1],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 1],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
        ])
        self.encoding = {
            (0, 0): 0,
            (0, 1): 1,
            (1, 0): 2,
            (1, 1): 3,
            (2, 0): 4,
            (2, 1): 5,
            (3, 0): 6,
            (3, 1): 7,
            (3, 2): 8,
            (3, 3): 9,
            (4, 0): 10,
            (4, 1): 11,
            (4, 2): 12,
            (4, 3): 13,
            (5, 0): 14,
            (5, 1): 15,
            (5, 2): 16,
            (5, 3): 17,
            (6, 0): 18
        } 

        self._keys = list(self.encoding.keys())
        
    def encode_tile(self, tile_type, tile_orient, to_binary=True):
        enc = self.encoding[tile_type, tile_orient]

        if to_binary:
            enc = self.binary[enc]

        return enc

    def keys(self):
        return self._keys



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


class QNetworkSmall(nn.Module):
    def __init__(self, n_valid) -> None:
        super().__init__()

        self.linear = nn.Sequential(
            nn.Linear(36, 32),
            nn.ReLU(),
            nn.Linear(32, n_valid),
            nn.Softmax()
        )

    def forward(self, state_enc):
        return self.linear(state_enc)


class QNetworkLarge(nn.Module):
    def __init__(self, n_valid) -> None:
        super().__init__()

        self.linear = nn.Sequential(
            nn.Linear(37, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_valid),
            nn.Softmax(),
        )

    def forward(self, state_enc):
        return self.linear(state_enc)

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

        self.i_prev_action = i_action = rng.integers(len(actions)) \
            if (rng.random() < self.epsilon) \
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
    def __init__(self, alpha, epsilon, epsilon_scale, replay_buffer_size, batch_size, sync_target_episode_count, episode_count, moving_avg_size=100):
        # Initialize training parameters
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_scale = epsilon_scale
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        self.sync_target_episode_count = sync_target_episode_count
        self.episode = 0
        self.episode_count = episode_count

        self.rewards = []
        self.avg_rewards = []
        self.moving_avg_size = moving_avg_size

    def get_valid_actions(self, n_tile_types):
        actions = []
        n_valid_actions = []

        for tile_type in range(n_tile_types):
            curr_tile = self.gameboard.tiles[tile_type]
            n_orientations = len(curr_tile)
            tile_actions = []
            for tile_orientation in range(n_orientations):
                width = len(curr_tile[tile_orientation])
                for tile_x in range(self.gameboard.N_col - width + 1):
                    tile_actions.append((tile_x, tile_orientation))

            n_valid_actions.append(len(tile_actions))
            actions.append(tile_actions)

        return actions, n_valid_actions


    def fn_init(self, gameboard, task_name=None, load_strategy=False, save_strategy=False, save_plot=False, show_plot=False):
        self.gameboard = gameboard
        self.exp_buffer = []
        self.reward_tots = np.zeros(self.episode_count)
        self.n_exp = 0

        self.task_name = task_name

        self.load_strategy = load_strategy
        self.save_strategy = save_strategy
        self.save_plot = save_plot
        self.show_plot = show_plot

        self.is_large = self.gameboard.tile_size > 2

        if not self.is_large:
            n_tile_types = 4
            valid_actions, n_valid_actions = self.get_valid_actions(n_tile_types)
            d_state = 32
            d_tile = 5
            ae = SmallStateAutoEncoder(d_state, 64)
            ae.load_state_dict(torch.load('./src/models/ae_small.pt'))
            tile_enc = SmallTileEncoder()
            qnn = QNetworkSmall(max(n_valid_actions))
        else:
            n_tile_types = 7
            valid_actions = self.get_valid_actions(n_tile_types)
            n_valid_actions = len(valid_actions)
            d_state = 32
            d_tile = 4
            ae = LargeStateAutoEncoder(d_state, 128)
            ae.load_state_dict(torch.load('./src/models/ae_large.pt'))
            tile_enc = LargeTileEncoder()
            qnn = QNetworkLarge(max(n_valid_actions))

        ae.eval()

        self.d_state = d_state
        self.d_tile = d_tile

        self.valid_actions = valid_actions
        self.n_valid_actions = n_valid_actions

        self.board_encoder = ae.encoder
        self.board_encoder.eval()

        self.tile_encoder = tile_enc
        self.qnn = qnn.to(device)
        self.qnn_target = copy.deepcopy(qnn).to(device)

        self.qnn.eval()
        self.qnn_target.eval()

        self.optim = torch.optim.Adam(self.qnn.parameters(), lr=self.alpha)
        self.loss_fn = nn.MSELoss()

        #print(self.gameboard.board)
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

    def encode_board(self, board):
        board_enc = self.board_encoder(board)
        return board_enc

    def encode_tile(self, tile_type, tile_orient):
        tile_enc = self.tile_encoder.encode_tile(tile_type, tile_orient, to_binary=True)
        return tile_enc[None, :]
    
    def get_action(self, idx):
        return self.valid_actions[self.gameboard.cur_tile_type][idx]

    def get_random_action(self):
        return rng.choice(self.valid_actions[self.gameboard.cur_tile_type])

    def fn_read_state(self):
        board = torch.tensor(self.gameboard.board)[None, None, ...]
        tile_type = self.gameboard.cur_tile_type
        tile_orient = self.gameboard.tile_orientation

        self.board_enc = self.encode_board(board)
        self.tile_enc = self.encode_tile(tile_type, tile_orient)

        self.state_enc = torch.concat((self.board_enc, self.tile_enc), axis=1)

        # Useful variables:
        # 'self.gameboard.N_row' number of rows in gameboard
        # 'self.gameboard.N_col' number of columns in gameboard
        # 'self.gameboard.board[index_row,index_col]' table indicating if row 'index_row' and column 'index_col' is occupied (+1) or free (-1)
        # 'self.gameboard.cur_tile_type' identifier of the current tile that should be placed on the game board (integer between 0 and len(self.gameboard.tiles))

    def fn_select_action(self):
        self.qnn.eval()
        epsilon = max(self.epsilon, 1 - self.episode / self.epsilon_scale)

        if rng.random() < epsilon:
            i_action = rng.integers(self.n_valid_actions[self.gameboard.curr_tile_type])
        else:
            state = self.state_enc.to(device)[None, :]
            qs = self.qnn(state)
            i_action = torch.argmax(qs)
            # TODO: Handle case if greater than length of valid actions

        self.action = self.get_action(i_action)
        self.gameboard.fn_move(*self.action)


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

        self.qnn.train()
        self.optim.zero_grad()

        q = torch.zeros(len(batch))
        y = torch.zeros(len(batch))

        # TODO: vectorize
        for i, (state_prev, action_prev, reward, state) in enumerate(batch):
            q[i] = self.qnn(state_prev)[action_prev]
            y[i] = reward + self.qnn_target(state).max()

            
        loss = self.loss_fn(q.to(device), y.to(device))
        loss.backward()

        self.optim.step()


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
                    self.qnn.eval()
                    self.qnn_target = copy.deepcopy(self.qnn).to(device)
                    pass
                    # TO BE COMPLETED BY STUDENT
                    # Here you should write line(s) to copy the current network to the target network
                self.gameboard.fn_restart()
        else:
            # Select and execute action (move the tile to the desired column and orientation)
            self.fn_select_action()
            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to copy the old state into the variable 'old_state' which is later stored in the ecperience replay buffer
            self.state_prev = self.state_enc

            # Drop the tile on the game board
            reward = self.gameboard.fn_drop()

            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to add the current reward to the total reward for the current episode, so you can save it to disk later
            self.reward_tots[self.episode] = reward

            # Read the new state
            self.fn_read_state()

            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to store the state in the experience replay buffer
            self.exp_buffer.append((self.state_prev, self.action, reward, self.state_enc))


            if len(self.exp_buffer) >= self.replay_buffer_size:
                # TO BE COMPLETED BY STUDENT
                # Here you should write line(s) to create a variable 'batch' containing 'self.batch_size' quadruplets
                batch = rng.choice(self.exp_buffer, self.batch_size)
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

