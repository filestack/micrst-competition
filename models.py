import torch.nn as nn
import torch.nn.functional as F


class MicrClassifier(nn.Module):
    def __init__(self):
        super(MicrClassifier, self).__init__()
        # In channels, out channels, kernel size, stride
        self.out_channels_1 = 16
        self.conv1 = nn.Conv2d(3, self.out_channels_1, 5, 1)

        self.out_channels_2 = 32
        self.conv2 = nn.Conv2d(self.out_channels_1, self.out_channels_2, 5, 1)

        # 4 * 4 is the image size after 2 pools and 2 convs
        # it's apparently not super trivial to calculate that shape
        # (long fractions and hidden +/- 1)
        self.fully_connected_input = 4 * 4 * self.out_channels_2
        self.fc1 = nn.Linear(self.fully_connected_input, 200)
        self.fc2 = nn.Linear(200, 14)

    def forward(self, x):
        x = F.relu(self.conv1(x))

        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.conv2(x))

        x = F.max_pool2d(x, 2, 2)

        x = x.view(-1, self.fully_connected_input)

        x = F.relu(self.fc1(x))

        x = self.fc2(x)

        return F.log_softmax(x, dim=1)
