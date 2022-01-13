# from _agent import AlphaZeroPlayer
# from _board import Board
# from _action import action
# from _episode import episode
from self_play_dataset import SelfPlayDataset
from network import AlphaZeroNet

import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


import os

class AlphaZero:
    def __init__(self, args):
        self.net = AlphaZeroNet().to(args.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, [10, 20, 30], gamma=0.1, last_epoch=-1)
        self.batch_size = args.batch_size
        self.device = args.device
        self.dataset = SelfPlayDataset(args.capacity, device=args.device)
        self.replay_buffer = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=False)
        self.writer = writer = SummaryWriter(args.logdir)
        self.total_steps = 0
        # self.dataset.load(args.sgf_dir)
    def self_play(self, sim_count, sgf_path, model_path):
        if not os.path.exists(sgf_path):
            os.makedirs(sgf_path)
        os.system("./self_play --total=10 --name=\"Hollow-NoGo\" --black=\"mcts N={} T=1000\" --white=\"mcts N={} T=1000\" --sgf_path={} --model_path={}". format(
            sim_count, sim_count, sgf_path, model_path
        ))
    def network_optimization(self, sgf_path, model_path):
        self.dataset.load(sgf_path)
        print('len(self.dataset):', len(self.dataset))
        replay_buffer = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=False)
        # policy_criterion = nn.CrossEntropyLoss()
        policy_criterion = lambda p_logits, p_labels: (
            (-p_labels * torch.log_softmax(p_logits, dim=1)).sum(dim=1).mean())
        value_criterion = nn.MSELoss()
        for _ in range(3):
            total_loss = 0
            for features, policy_labels, value_labels in replay_buffer:
                self.writer.add_scalar('Train/Learning Rate', self.optimizer.param_groups[0]['lr'], self.total_steps)
                self.total_steps += 1
                policy_logits, value_logits = self.net(features)
                # print(policy_logits.shape, policy_labels.shape)
                # print(policy_logits.dtype, policy_labels.dtype)
                # print('policy_labels[:, (30,31,32,39,40,41,48,49,50)]:', policy_labels[:, (30,31,32,39,40,41,48,49,50)])
                policy_labels[:, (30,31,32,39,40,41,48,49,50)] = 0
                policy_loss = policy_criterion(policy_logits, policy_labels)
                self.writer.add_scalar('Train/Policy Loss', policy_loss, self.total_steps)
                # print(value_logits.shape, value_labels.shape)
                # print('value_logits:', value_logits)
                # print('value_labels:', value_labels)
                value_loss = value_criterion(value_logits.squeeze(), value_labels)
                self.writer.add_scalar('Train/Value Loss', value_loss, self.total_steps)
                loss = policy_loss + value_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
        self.scheduler.step()
        self.save_model(model_path)
    def save_model(self, model_path):
        inputs = torch.rand(1, 10, 9, 9).to(self.device)
        traced_model = torch.jit.trace(self.net, inputs)
        traced_model.save(model_path)
    def load_model(self, model_path):
        self.net = torch.jit.load(model_path)
        

def train(args):
    alphazero = AlphaZero(args)
    # alphazero.self_play(sim_count=args.sim_count, sgf_path='./sgf/iteration_038', model_path='./model/iteration_038.pt')
    for i in range(1, args.num_iteration):
        sgf_path = os.path.join(args.sgf_dir, 'iteration_{0:03d}'.format(i))
        model_path = os.path.join(args.model_dir, 'iteration_{0:03d}.pt'.format(i))
        print('sgf_path:', sgf_path)
        print('model_path:', model_path)
        if not os.path.exists(sgf_path):
            os.makedirs(sgf_path)
        alphazero.network_optimization(sgf_path, model_path)
        alphazero.self_play(sim_count=args.sim_count, sgf_path=sgf_path, model_path=model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-I', '--num_iteration', default=1000, type=int)
    parser.add_argument('--logdir', default='log/')
    # replay buffer
    parser.add_argument('-c', '--capacity', default=10000, type=int)
    parser.add_argument('--sgf_dir', default='./sgf')
    # self play
    parser.add_argument('--sim_count', default=1000, type=int)
    parser.add_argument('--time_limit', default=1000, type=int)
    # network
    parser.add_argument('-d', '--device', default='cuda')
    parser.add_argument('-bs', '--batch_size', default=512, type=int)
    parser.add_argument('-lr', '--lr', default=.01, type=float)
    parser.add_argument('-m', '--model_dir', default='./model')
    args = parser.parse_args()
    train(args)


    
