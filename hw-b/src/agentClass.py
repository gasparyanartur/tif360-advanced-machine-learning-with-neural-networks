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

rng = np.random.default_rng(69420)

class SmallStateAutoEncoder(nn.Module):
    def __init__(self, d_L, d_H) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 2, padding=0),
            nn.ReLU(),
            nn.Flatten(1),   # dim 
            nn.Linear(512, d_H),
            nn.ReLU(),
            nn.Linear(d_H, d_L),
            nn.Tanh(),
        )        

        self.decoder = nn.Sequential(
            nn.Linear(d_L, d_H),
            nn.ReLU(),
            nn.Linear(d_H, 512),
            nn.ReLU(),
            nn.Unflatten(1, (32, 4, 4)),
            nn.ConvTranspose2d(32, 16, 1, stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 1, stride=1, padding=0),
        )

        self.load_state_dict(torch.load('./src/models/ae_small.pt'))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return torch.tanh(x)


class LargeStateAutoEncoder(nn.Module):
    def __init__(self, d_L, d_H) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 6, padding=1),
            nn.ReLU(),
            nn.Flatten(1),
            nn.Linear(2048, d_H),
            nn.ReLU(),
            nn.Linear(d_H, d_H),
            nn.ReLU(),
            nn.Linear(d_H, d_L),
            nn.Tanh()
        )        

        self.decoder = nn.Sequential(
            nn.Linear(d_L, d_H),
            nn.ReLU(),
            nn.Linear(d_H, d_H),
            nn.ReLU(),
            nn.Linear(d_H, 2048),
            nn.ReLU(),
            nn.Unflatten(1, (128, 4, 4)),
            nn.ConvTranspose2d(128, 64, 2, stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 2, stride=1, padding=0),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return torch.tanh(x)
        

class TileEncoder:
    def __init__(self, d_enc):
        self.d_enc = d_enc
        
        self.values = -torch.ones(d_enc, d_enc, dtype=torch.float32)
        for i in range(d_enc):
            self.values[i, i] = 1

    def encode_tile(self, tile_type):
        return self.values[tile_type][None, ...]


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
    def __init__(self, n_valid, d_state=16, d_hidden=64) -> None:
        super().__init__()

        self.linear = nn.Sequential(
            nn.Linear(d_state, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, n_valid),
        )


    def forward(self, state_enc, tile_enc):
        state_enc = state_enc.squeeze()
        tile_enc = tile_enc.squeeze()
        if len(state_enc.shape) == 1:
            x = torch.concat((state_enc, tile_enc), dim=0)
        else:
            x = torch.concat((state_enc, tile_enc), dim=1)
        x = self.linear(x)
        return x


class QNetworkLarge(nn.Module):
    def __init__(self, n_valid, d_state=64, d_hidden=128) -> None:
        super().__init__()

        self.linear = nn.Sequential(
            nn.Linear(d_state, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, n_valid),
        )

    def forward(self, state_enc):
        return self.linear(state_enc)

class QNetworkCNN(nn.Module):
    def __init__(self, n_valid, d_state=64, d_hidden=128, d_tile=7) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 6, padding=1),
            nn.ReLU(),
            nn.Flatten(1),
            nn.Linear(2048, d_hidden),
            nn.Tanh(),
        )
        self.linear = nn.Sequential(
            nn.Linear(d_hidden+d_tile, d_state)
        )

    def forward(self, state_enc, tile_enc):
        state_enc = state_enc.squeeze()
        tile_enc = tile_enc.squeeze()

        if len(state_enc.shape) == 2:
            state_enc = state_enc[None, ...]

        b, h, w = state_enc.shape
        state_enc = state_enc.reshape(b, 1, h, w)

        if len(tile_enc.shape) == 1:
            tile_enc = tile_enc[None, ...]

        x = self.encoder(state_enc)
        
        x = torch.concat((x, tile_enc), dim=1)
        x = self.linear(x)

        return x




class StateFlatEncoder(nn.Module):
    def forward(self, x):
        return nn.Flatten(1)(x)

class StateIdentiyEncoder(nn.Module):
    def forward(self, x):
        return x


class TQAgent:
    def __init__(self, alpha, epsilon, episode_count, moving_avg_size=100):
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
    def __init__(self, alpha, epsilon, epsilon_scale, replay_buffer_size, batch_size, sync_target_episode_count, episode_count, moving_avg_size=100):
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

    def get_valid_actions(self, n_tile_types, check_bounds=True):
        actions = []
        n_valid_actions = []

        for tile_type in range(n_tile_types):
            curr_tile = self.gameboard.tiles[tile_type]
            n_orientations = len(curr_tile)
            tile_actions = []
            for tile_orientation in range(n_orientations):
                width = len(curr_tile[tile_orientation])
                for tile_x in range(self.gameboard.N_col):
                    if (not check_bounds) or (tile_x <= self.gameboard.N_col - width):
                        tile_actions.append((tile_x, tile_orientation))

            n_valid_actions.append(len(tile_actions))
            actions.append(tile_actions)

        return actions, n_valid_actions


    def fn_init(self, gameboard, task_name=None, load_strategy=False, save_strategy=False, save_plot=False, show_plot=False):
        self.task_name = task_name
        self.load_strategy = load_strategy
        self.save_strategy = save_strategy
        self.save_plot = save_plot
        self.show_plot = show_plot        

        self.gameboard = gameboard
        self.exp_buffer = []
        self.episode_reward = 0
        self.rewards = np.zeros(self.episode_count)
        self.n_exp = 0
        self.is_large = self.gameboard.tile_size > 2

        self.buf_prev_boards = []
        self.buf_actions = []
        self.buf_rewards = []
        self.buf_boards = []
        self.buf_terminals = []
        self.buf_prev_tiles = []
        self.buf_tiles = []

        if not self.is_large:
            self.d_state = self.gameboard.N_col*self.gameboard.N_row
            self.d_tile = len(self.gameboard.tiles)
            self.valid_actions, self.n_valid_actions = self.get_valid_actions(self.d_tile, check_bounds=False)
            self.n_actions = max(self.n_valid_actions)

            self.board_encoder = StateFlatEncoder() 
            self.qnn = QNetworkSmall(self.n_actions, d_state=self.d_state+self.d_tile, d_hidden=128)
            self.checkpoint_path = './src/checkpoints/cp-small.pt'

        else:
            self.d_state = self.gameboard.N_col * self.gameboard.N_row
            self.d_tile = len(self.gameboard.tiles)
            self.valid_actions, self.n_valid_actions = self.get_valid_actions(self.d_tile, check_bounds=False)
            self.n_actions = max(self.n_valid_actions)

            #self.board_encoder = StateFlatEncoder()
            #self.qnn = QNetworkLarge(self.n_actions, d_state=self.d_state+self.d_tile, d_hidden=256)

            self.board_encoder = StateIdentiyEncoder()
            self.qnn = QNetworkCNN(self.n_actions, d_tile=self.d_tile, d_state=self.d_state, d_hidden=256)

            #self.board_encoder = LargeStateAutoEncoder(32, 512).encoder
            #self.board_encoder.requires_grad_(False)
            #self.d_state = 32

            #self.qnn = QNetworkLarge(self.n_actions, d_state=self.d_state+self.d_tile, d_hidden=128)
            self.checkpoint_path = './src/checkpoints/cp-large.pt'

        self.tile_encoder = TileEncoder(self.d_tile)

        self.qnn.eval()
        self.tnn = copy.deepcopy(self.qnn)
        self.tnn.requires_grad_(False)

        self.optim = torch.optim.Adam(self.qnn.parameters(), lr=self.alpha)
        self.loss_fn = nn.MSELoss()

        if load_strategy:
            self.fn_load_strategy(self.checkpoint_path)

    def fn_load_strategy(self, strategy_file):
        cp = torch.load(strategy_file)
        self.qnn.load_state_dict(cp['qnn'])
        self.tnn.load_state_dict(cp['tnn'])
        self.optim.load_state_dict(cp['optim'])
        self.rewards = cp['rewards_tot']
        self.avg_rewards = cp['rewards_avg']
        self.buf_prev_boards = cp['boards_old']
        self.buf_actions = cp['actions']
        self.buf_rewards = cp['rewards']
        self.buf_boards = cp['boards_new']
        self.buf_terminals = cp['terminals']
        self.buf_prev_tiles = cp['prev_tiles']
        self.buf_tiles = cp['tiles']
        self.n_exp = cp['n_exp']
        self.episode = cp['episode']

    def encode_board(self, board):
        board_enc = self.board_encoder(board)
        return board_enc

    def encode_tile(self, tile_type):
        tile_enc = self.tile_encoder.encode_tile(tile_type)
        return tile_enc
    
    def get_action(self, idx):
        if idx >= self.n_valid_actions[self.gameboard.cur_tile_type]:
            return -1, -1

        return self.valid_actions[self.gameboard.cur_tile_type][idx]

    def fn_read_state(self):
        board = torch.tensor(self.gameboard.board)[None, None, ...]
        tile_type = self.gameboard.cur_tile_type

        self.board_enc = self.encode_board(board)
        self.tile_enc = self.encode_tile(tile_type)

    def fn_select_action(self):
        self.qnn.eval()
        epsilon = max(self.epsilon, 1 - self.episode / self.epsilon_scale)

        if rng.random() < epsilon:
            i_action = rng.integers(self.n_actions)
        else:
            board = self.board_enc
            tile = self.tile_enc
            qs = self.qnn(board, tile).flatten()
            i_action = torch.argmax(qs)


        self.i_action = i_action
        self.action = self.get_action(i_action)
        self.gameboard.fn_move(*self.action)

    def fn_reinforce(self, batch):
        self.qnn.train()
        self.optim.zero_grad()

        prev_states, actions, rewards, states, terminals, prev_tile, tile = batch

        qp = self.qnn(prev_states, prev_tile)
        qt = self.tnn(states, tile)
        q = qp[range(self.batch_size), actions]
        y = rewards + (terminals==0) * (torch.max(qt, dim=-1).values)

        loss = self.loss_fn(q, y)
        loss.backward()
        self.optim.step()


    def fn_turn(self):
        if self.gameboard.gameover or (self.episode >= self.episode_count):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.episode += 1

            if self.episode % 100 == 0:
                avg_reward = np.mean(self.rewards[self.episode-self.moving_avg_size:self.episode])
                self.avg_rewards.append(avg_reward)
                print(f'Episode: {self.episode}/{self.episode_count}, Mean reward [T={self.moving_avg_size}]: {avg_reward}')

            if self.episode % 1000 == 0:
                saveEpisodes = [1000, 2000, 5000, 10000, 20000,
                                50000, 100000, 200000, 500000, 1000000]
                if self.episode in saveEpisodes:
                    self.save_state()
                    
            if self.episode >= self.episode_count:
                self.save_metrics()
                raise SystemExit(0)

            else:
                if (self.n_exp >= self.replay_buffer_size) and ((self.episode % self.sync_target_episode_count) == 0):
                    self.tnn = copy.deepcopy(self.qnn)
                    self.tnn.requires_grad_(False)

                self.gameboard.fn_restart()
        else:
            self.fn_select_action()
            self.board_prev = self.board_enc
            self.tile_prev = self.tile_enc

            reward = self.gameboard.fn_drop()
            self.rewards[self.episode] += reward

            self.fn_read_state()

            t_action = torch.tensor([self.i_action])
            t_reward = torch.tensor([reward])
            t_terminal = torch.tensor([self.gameboard.gameover])

            self.add_buffer(self.board_prev, t_action, t_reward, self.board_enc, t_terminal, self.tile_prev, self.tile_enc)

            self.n_exp += 1

            if self.n_exp >= self.replay_buffer_size:

                prev_boards, actions, rewards, boards, terminals, prev_tiles, tiles = [],[],[],[],[],[],[]

                for _ in range(min(self.batch_size, len(self.buf_prev_boards))):
                    idx = rng.integers(0, len(self.buf_prev_boards))

                    prev_boards.append(self.buf_prev_boards[idx])
                    actions.append(self.buf_actions[idx])
                    rewards.append(self.buf_rewards[idx])
                    boards.append(self.buf_boards[idx])
                    terminals.append(self.buf_terminals[idx])
                    prev_tiles.append(self.buf_prev_tiles[idx])
                    tiles.append(self.buf_tiles[idx])
                    
                self.fn_reinforce((
                    torch.stack(prev_boards), 
                    torch.stack(actions).flatten(), 
                    torch.stack(rewards).flatten(), 
                    torch.stack(boards), 
                    torch.stack(terminals).flatten(),
                    torch.stack(prev_tiles),
                    torch.stack(tiles),
                ))

                self.pop_buffer()

    def save_metrics(self):
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

    def add_buffer(self, state_prev, action, reward, state_new, terminal, prev_tile, tile):
        self.buf_prev_boards.append(state_prev)
        self.buf_actions.append(action)
        self.buf_rewards.append(reward)
        self.buf_boards.append(state_new)
        self.buf_terminals.append(terminal)
        self.buf_prev_tiles.append(prev_tile)
        self.buf_tiles.append(tile)

    def pop_buffer(self):
        self.buf_prev_boards.pop(0)
        self.buf_actions.pop(0)
        self.buf_rewards.pop(0)
        self.buf_boards.pop(0)
        self.buf_terminals.pop(0)
        self.buf_prev_tiles.pop(0)
        self.buf_tiles.pop(0)

    def save_state(self):
        self.qnn.eval()                    
        torch.save({
            'qnn': self.qnn.state_dict(),
            'tnn': self.tnn.state_dict(),
            'optim': self.optim.state_dict(),
            'rewards_tot': self.rewards,
            'rewards_avg': self.avg_rewards,
            'boards_old': self.buf_prev_boards,
            'boards_new': self.buf_boards,
            'actions': self.buf_actions,
            'rewards': self.buf_rewards,
            'terminals': self.buf_terminals,
            'prev_tiles': self.buf_prev_tiles,
            'tiles': self.buf_tiles,
            'episode': self.episode,
            'n_exp': self.n_exp
        }, self.checkpoint_path)


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

