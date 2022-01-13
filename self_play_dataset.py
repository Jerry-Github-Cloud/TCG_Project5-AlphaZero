import numpy as np
import torch
import glob
import os
import sgf
import json
import ast
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# from _board_pb import Board
# from module import Board
from _board import Board
from collections import deque
BOARD_SIZE = 9

class SelfPlayDataset(Dataset):
    __slots__ = ['buffer']
    def __init__(self, capacity, device='cuda') -> None:
        super().__init__()
        self.sgf_paths = []
        self.device = device
        # self.buffer = []
        self.buffer = deque(maxlen=capacity)
    
    def sgfmove2coor(self, sgfmove):
        x = ord(sgfmove[0]) - ord('a')
        y = BOARD_SIZE - (ord(sgfmove[1]) - ord('a')) - 1
        return x, y

    def _dihedral(self, f, r):
        f = f if r < 4 else f.transpose(1, 2)
        return torch.rot90(f, r % 4, (1, 2))
        
    def _add_isomorphic_data(self, data):
        feature, policy_label, value_label = data
        for r in range(7):
            feature = self._dihedral(feature, r)
            policy_label = self._dihedral(policy_label, r)
            # policy_label = torch.reshape(policy_label, (-1,))
            self.buffer.append((
                feature.float().to(self.device), 
                policy_label.reshape(BOARD_SIZE**2).float().to(self.device), 
                torch.tensor(value_label, dtype=torch.float).to(self.device) ))
    def _index2coor(self, index):
        return index // BOARD_SIZE, index % BOARD_SIZE 

    def dict2labels(self, d):
        sim_count = sum(d.values())
        labels = torch.zeros(BOARD_SIZE, BOARD_SIZE, dtype=torch.float)
        for i, value in d.items():
            labels[self._index2coor(i)] = value / sim_count
        return labels
    def _process_one_game(self, game, board):
        winner = game.root.properties['RE']
        for i, node in enumerate(game):
            prop = node.properties
            # print(prop)
            color = None
            sgf_move = ''
            comment = ''
            if ('B' in prop) or ('W' in prop):
                if 'B' in prop:
                    color = 'B'
                    sgf_move = prop['B'][0]
                elif 'W' in prop:
                    color = 'W'
                    sgf_move = prop['W'][0]
                if 'C' in prop:
                    common_str = prop['C'][0]
                    comment = ast.literal_eval(common_str)
                # print('common_str:', common_str)
                # print('type(common_str):' ,type(common_str))
                # print('common:', comment)
                # print('type(comment):', type(comment))
                # feature map
                feature = torch.tensor(board.observation_tensor()).reshape(10, BOARD_SIZE, BOARD_SIZE)

                # policy label
                # policy_label = torch.zeros(BOARD_SIZE, BOARD_SIZE, dtype=torch.float)
                policy_label = self.dict2labels(comment)
                # print(policy_label)
                policy_label = policy_label.reshape(1, BOARD_SIZE, BOARD_SIZE)

                # value label
                value_label = 1 if color == winner[0][0] else -1
                print('feature:', feature)
                print('policy_label:', policy_label)
                print('value_label:', value_label)
                self._add_isomorphic_data((feature, policy_label, value_label))
                if color == 'B':
                    who = 1
                if color == 'W':
                    who = 2
                x, y = self.sgfmove2coor(sgf_move)
                board.place(x, y, who)
                # print(board)
    def load(self, sgf_dir):
        for f in os.listdir(sgf_dir):
            path = os.path.join(sgf_dir, f)
            _, ext = os.path.splitext(path)
            if ext != '.sgf':
                continue
            print(path)
            self.sgf_paths.append(path)
        for sgf_path in self.sgf_paths:
            # print(sgf_path)
            f = open(sgf_path)
            collection = sgf.parse(f.read())
            game = collection.children[0]
            board = Board()
            self._process_one_game(game, board)

    def __getitem__(self, index):
        return self.buffer[index]
    def __len__(self):
        return len(self.buffer)

if __name__ == '__main__':
    dataset = SelfPlayDataset(capacity=2000)
    dataset.load('./sgf/iteration_001')
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True)
    # for features, batch_p, batch_v in dataloader:
    #     print(batch_v)
    # feature, policy_label, value_label = dataset[1113]
    # print(feature.shape)
    # print(policy_label.shape)
    # print(value_label)

    ### test pybind Board
    # b = Board()
    # print(b)
    # print(b.observation_tensor())
    # print(b.check_liberty(3, 2, 1))
    # b.place(2, 1, 1)
    # b.place(0, 1, 2)
    # print(b)
    # print(b)
    # print(help(Board))
    # b.rotate(1)
    # print(b)

        