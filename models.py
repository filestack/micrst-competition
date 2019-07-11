import torch.nn as nn
import torch.nn.functional as F


class MicrClassifier(nn.Module):
    def __init__(self):
        super(MicrClassifier, self).__init__()
        # In channels, out channels, kernel size, stride
        self.out_channels_1 = 10
        self.conv1 = nn.Conv2d(3, self.out_channels_1, 5, 1)

        self.out_channels_2 = 20
        self.conv2 = nn.Conv2d(self.out_channels_1, self.out_channels_2, 5, 1)

        # 4 * 4 is the image size after 2 pools and 2 convs
        # it's apparently not super trivial to calculate that shape
        # (long fractions and hidden +/- 1)
        self.fully_connected_input = 4 * 4 * self.out_channels_2
        self.fc1 = nn.Linear(self.fully_connected_input, 200)
        self.fc2 = nn.Linear(200, 14)

    def forward(self, x):
        print('Input:', x.shape)
        x = F.relu(self.conv1(x))
        print('Crelu1', x.shape)
        x = F.max_pool2d(x, 2, 2)
        # print('Pool2d', x.shape)
        x = F.relu(self.conv2(x))
        # print('Crelu2', x.shape)
        # print('Fancy transpose:', x.permute(3, 0, 2, 1).shape)
        # print('Fancy squeeze  :', x.permute(3, 0, 2, 1).squeeze(1).shape)
        x = F.max_pool2d(x, 2, 2)
        # print(x.shape)
        x = x.view(-1, self.fully_connected_input)
        # print(x.shape)
        x = F.relu(self.fc1(x))
        # print(x.shape)
        x = self.fc2(x)
        # print(x.shape)
        return F.log_softmax(x, dim=1)
