import torch.nn as nn
import torch.nn.functional as F


class MicrClassifier(nn.Module):
    def __init__(self, verbose=False):
        super(MicrClassifier, self).__init__()
        # Print some diagnostics?
        self.verbose = verbose

        # In channels, out channels, kernel size, stride
        self.out_channels_1 = 16
        self.conv1 = nn.Conv2d(3, self.out_channels_1, 5, 1)

        self.bn1 = nn.BatchNorm2d(self.out_channels_1)

        self.out_channels_2 = 32
        self.conv2 = nn.Conv2d(self.out_channels_1, self.out_channels_2, 5, 1)

        self.bn2 = nn.BatchNorm2d(self.out_channels_2)

        # 4 * 4 is the image size after 2 pools and 2 convs
        # it's apparently not super trivial to calculate that shape
        # (long fractions and hidden +/- 1)
        self.fully_connected_input = 4 * 4 * self.out_channels_2
        self.fc1 = nn.Linear(self.fully_connected_input, 200)
        self.fc2 = nn.Linear(200, 14)

    def forward(self, x):
        self.debug('Input:', x.shape)

        x = F.leaky_relu(self.conv1(x))
        x = self.bn1(x)
        x = F.avg_pool2d(x, 2, 2)

        self.debug('Conv block 1:', x.shape)

        x = F.leaky_relu(self.conv2(x))
        x = self.bn2(x)
        x = F.avg_pool2d(x, 2, 2)

        self.debug('Conv block 2:', x.shape)

        x = x.view(-1, self.fully_connected_input)

        x = F.leaky_relu(self.fc1(x))

        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

    def debug(self, title, value):
        if self.verbose:
            print(title, value)


class DeeperMicrClassifier(nn.Module):
    def __init__(self, verbose=False):
        super(DeeperMicrClassifier, self).__init__()
        # Print some diagnostics?
        self.verbose = verbose

        # In channels, out channels, kernel size, stride
        self.out_channels_1 = 16
        self.conv1 = nn.Conv2d(3, self.out_channels_1, 5, 1)

        self.bn1 = nn.BatchNorm2d(self.out_channels_1)

        self.out_channels_2 = 32
        self.conv2 = nn.Conv2d(self.out_channels_1, self.out_channels_2, 5, 1)

        self.bn2 = nn.BatchNorm2d(self.out_channels_2)

        self.out_channels_3 = 64
        self.conv3 = nn.Conv2d(self.out_channels_2, self.out_channels_3, 5, 1)

        self.bn3 = nn.BatchNorm2d(self.out_channels_3)

        # 4 * 4 is the image size after 2 pools and 2 convs
        # it's apparently not super trivial to calculate that shape
        # (long fractions and hidden +/- 1)
        self.fully_connected_input = 4 * 4 * self.out_channels_3
        self.fc1 = nn.Linear(self.fully_connected_input, 200)
        self.fc2 = nn.Linear(200, 14)

    def forward(self, x):
        self.debug('Input:', x.shape)

        x = F.leaky_relu(self.conv1(x))
        x = self.bn1(x)
        x = F.avg_pool2d(x, 2, 2)

        self.debug('Conv block 1:', x.shape)

        x = F.leaky_relu(self.conv2(x))
        x = self.bn2(x)

        self.debug('Conv block 2:', x.shape)

        x = F.leaky_relu(self.conv3(x))
        x = self.bn3(x)
        self.debug('Conv block 3:', x.shape)

        x = x.view(-1, self.fully_connected_input)

        x = F.leaky_relu(self.fc1(x))

        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

    def debug(self, title, value):
        if self.verbose:
            print(title, value)
