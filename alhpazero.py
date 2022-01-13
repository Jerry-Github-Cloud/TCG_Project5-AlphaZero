from _agent import AlphaZeroPlayer
from _board import Board
from _action import action
from _episode import episode
from self_play_dataset import SelfPlayDataset
from network import AlphaZeroNet

import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader

import os

class AlphaZero:
    def __init__(self, args):
        self.net = AlphaZeroNet().to(args.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.lr, weight_decay=1e-4)
        self.batch_size = args.batch_size
        self.device = args.device
        self.dataset = SelfPlayDataset(args.capacity, device=args.device)
        self.dataset.load(args.sgf_path)
    def self_play(self, sgf_path, model_path):
        if not os.path.exists(sgf_path):
            os.mkdir(sgf_path)
        black_player = AlphaZeroPlayer("name=black mcts N=10 role=black")
        white_player = AlphaZeroPlayer("name=white mcts N=10 role=white")
        black_player.load_model(model_path)
        white_player.load_model(model_path)
        num_game = 10
        for i in range(num_game):    
            game = episode()
            board = Board()
            turn = 0
            while True:
                if turn % 2 == 0:
                    player = black_player
                else:
                    player = white_player
                move, policy_labels = player.take_action(board)
                move.apply(board)
                if game.apply_action(move, policy_labels) != True:
                    break
                turn += 1
            with open(os.path.join(sgf_path,'/{0:05d}.sgf'.format(i)), 'w') as f:
                f.write(game.__repr__())

    def network_optimization(self, sgf_path, model_path):
        self.dataset.load(sgf_path)
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        policy_criterion = nn.CrossEntropyLoss()
        value_criterion = nn.MSELoss()
        for features, policy_labels, value_labels in dataloader:
            policy_logits, value_logits = self.net(features)
            # print(policy_logits.shape, policy_labels.shape)
            policy_loss = policy_criterion(policy_logits, policy_labels)
            # print(value_logits.shape, value_labels.shape)
            value_loss = value_criterion(value_logits, value_labels)
            print('\tpolicy_loss:', policy_loss.item())
            print('\tvalue_loss:', value_loss.item())
            loss = policy_loss + value_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        # torch.save(self.net.state_dict(), 'model/test.pt')
        self.save_model(model_path)
    def save_model(self, model_path, checkpoint=False):
        # inputs = torch.rand(1, 10, 9, 9).to(self.device)
        # traced_net = torch.jit.trace(self.net, inputs)
        # traced_net.save(model_path)
        script_model = torch.jit.script(self.net)
        script_model.save(model_path)
        

def main(args):
    os.system("./self_play  --total=10 --name=\"Hollow-NoGo\" --black=\"mcts N=10\" --white=\"mcts N=10\" --model_path=\"model/test2.pt\" --sgf_path=\"sgf\"")
    # alphazero = AlphaZero(args)
    # last_model_path = os.path.join(args.sgf_dir, 'Iteration_000')
    # alphazero.save_model(last_model_path)
    # for i in range(1, args.num_iteration):
    #     sgf_path = os.path.join(args.sgf_dir, 'Iteration_{0:03d}'.format(i))
    #     alphazero.self_play(sgf_path, last_model_path)
    #     model_path = os.path.join(args.model_dir, 'Iteration_{0:03d}.pt'.format(i))
    #     alphazero.network_optimization(model_path)
    #     last_model_path = model_path



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-I', '--num_iteration', default=2)
    # replay buffer
    parser.add_argument('-c', '--capacity', default=1000)
    parser.add_argument('--sgf_dir', default='./sgf')
    # network
    parser.add_argument('-d', '--device', default='gpu')
    parser.add_argument('-bs', '--batch_size', default=16, type=int)
    parser.add_argument('-lr', '--lr', default=.01, type=float)
    parser.add_argument('-m', '--model_dir', default='model')
    args = parser.parse_args()
    main(args)

    # black_player = AlphaZeroPlayer("name=black mcts N=10 role=black")
    # white_player = AlphaZeroPlayer("name=white mcts N=10 role=white")
    # black_player.load_model("/mnt/nfs/work/oo12374/Course/TCG/TCG_Project5/code/model/test.pt")
    # white_player.load_model("/mnt/nfs/work/oo12374/Course/TCG/TCG_Project5/code/model/test.pt")
    # num_game = 10
    # for i in range(num_game):    
    #     game = episode()
    #     board = Board()
    #     turn = 0
    #     while True:
    #         if turn % 2 == 0:
    #             player = black_player
    #         else:
    #             player = white_player
    #         move, policy_labels = player.take_action(board)
    #         move.apply(board)
    #         if game.apply_action(move, policy_labels) != True:
    #             break
    #         turn += 1
    #     with open('sgf/{0:05d}.sgf'.format(i), 'w') as f:
    #         f.write(game.__repr__())
    #     print(board)
    #     print(game)

    # for i in range(50):
    #     if i % 2 == 0:
    #         player = black_player
    #     else:
    #         player = white_player
    #     action = player.take_action(board)
    #     action.apply(board)
    #     print(board)

    
