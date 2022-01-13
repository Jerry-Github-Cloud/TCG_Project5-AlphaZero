import torch
from torch import nn
from torch.utils.data import DataLoader

from self_play_dataset import SelfPlayDataset
BOARD_SIZE = 9

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        nn.init.kaiming_normal_(self.conv1.weight,
                                mode="fan_out",
                                nonlinearity="relu")
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        nn.init.kaiming_normal_(self.conv2.weight,
                                mode="fan_out",
                                nonlinearity="relu")
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu1(y)
        y = self.conv2(y)
        y = self.bn2(y)
        x = x + y
        x = self.relu2(x)
        return x


class ResNet(nn.Module):
    def __init__(self, in_channels, blocks, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.convs = nn.ModuleList([
            BasicBlock(in_channels=out_channels,
                       out_channels=out_channels) for _ in range(blocks)
        ])

    def forward(self, x):
        x = self.conv1(x)
        for conv in self.convs:
            x = conv(x)
        return x


class AlphaZeroNet(nn.Module):
    def __init__(self,):
        super().__init__()
        # channels, height, width
        self.observation_tensor_shape = (10, BOARD_SIZE, BOARD_SIZE)
        in_channels, height, width = self.observation_tensor_shape
        channels = 32
        blocks = 10
        self.backbone = ResNet(in_channels, blocks, channels)

        # policy head
        self.policy_head_front = nn.Sequential(
            nn.Conv2d(in_channels=channels,
                      out_channels=2,
                      kernel_size=1),
            nn.BatchNorm2d(num_features=2),
            nn.ReLU(),
        )

        self.policy_head_end = nn.Sequential(
            nn.Linear(in_features=2 * height * width,
                      out_features=BOARD_SIZE**2),
            nn.Softmax(dim=1)
        )

        # value head
        self.value_head_front = nn.Sequential(
            nn.Conv2d(in_channels=channels,
                      out_channels=1,
                      kernel_size=1),
            nn.BatchNorm2d(num_features=1),
            nn.ReLU(),
        )

        self.value_head_end = nn.Sequential(
            nn.Linear(in_features=height * width,
                      out_features=channels),
            nn.ReLU(),
            nn.Linear(in_features=channels,
                      out_features=1),
            nn.Tanh()
        )

    def forward(self, x):
        _, height, width = self.observation_tensor_shape
        x = self.backbone(x)
        # policy head
        p = self.policy_head_front(x)
        p = p.view(-1, 2 * height * width)
        p = self.policy_head_end(p)        
        # value head
        v = self.value_head_front(x)
        v = v.view(-1, height * width)
        v = self.value_head_end(v)
        return p, v

if __name__ == '__main__':
    net = AlphaZeroNet()
    dataset = SelfPlayDataset(capacity=1000, device='cpu')
    dataset.load('./sgf/iteration_001')
    dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=False)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=1e-4)

    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()
    for features, policy_labels, value_labels in dataloader:
        # features = features.to(device)
        # policy_labels = policy_labels.to(device)
        # value_labels = value_labels.to(device)
        policy_logits, value_logits = net(features)
        # print(policy_logits.shape, policy_labels.shape)
        policy_loss = policy_criterion(policy_logits, policy_labels)
        # print(value_logits.shape, value_labels.shape)
        value_loss = value_criterion(value_logits.squeeze(), value_labels)
        print('\tpolicy_loss:', policy_loss.item())
        print('\tvalue_loss:', value_loss.item())
        loss = policy_loss + value_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.save(net, 'test3.pt')
