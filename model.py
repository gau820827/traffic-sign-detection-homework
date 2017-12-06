import torch
import torch.nn as nn
import torch.nn.functional as F

nclasses = 43   # GTSRB as 43 classes


# This is the baseline model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 500)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


# 3x3 Convolution
def conv3x3(in_channels, out_channels, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


# Residual Network
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        # First CNN
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Second CNN
        out = self.conv2(out)
        out = self.bn2(out)

        # Down sampling
        if self.downsample:
            residual = self.downsample(x)

        # Residual learning
        out += residual
        out = self.relu(out)
        return out


# ResNet Module
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=nclasses):
        super(ResNet, self).__init__()
        self.in_channels = 32

        self.conv = conv3x3(3, 32)
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self.make_layer(block, 32, layers[0])
        self.layer2 = self.make_layer(block, 64, layers[1])
        self.layer3 = self.make_layer(block, 128, layers[2], stride=2)
        self.layer4 = self.make_layer(block, 256, layers[3], stride=2)
        self.avg_pool = nn.AvgPool2d(4)

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def make_layer(self, block, out_channels, num_blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, num_blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # First CNN
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)

        # Residual Block
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)

        # Flatten
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        x = F.dropout(out, training=self.training)
        out = self.fc2(out)
        return F.log_softmax(out)


class LocalNet(nn.Module):
    def __init__(self):
        super(LocalNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 100, kernel_size=5),
            nn.BatchNorm2d(100),
            nn.ReLU(True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(100, 200, kernel_size=5),
            nn.BatchNorm2d(200),
            nn.ReLU(True)
        )

    def forward(self, x):
        out1 = F.max_pool2d(self.conv1(x), 2)
        out2 = F.max_pool2d(self.conv2(out1), 2)
        out1 = F.max_pool2d(out1, 4)
        out2 = F.max_pool2d(out2, 2)

        # Concatenate out1 and out2
        batch_size = out1.size()[0]
        out = torch.cat((out1.view(batch_size, -1), out2.view(batch_size, -1)), dim=1)
        return out

# Spatial Network
class SpaNet(nn.Module):
    def __init__(self):
        super(SpaNet, self).__init__()
        # Color transformer
        self.colorization = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=1),
            nn.BatchNorm2d(10),
            nn.ReLU(True),
            nn.Conv2d(10, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(True)
        )

        self.conv1 = nn.Conv2d(3, 100, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm2d(100)
        self.conv2 = nn.Conv2d(100, 150, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(150)
        self.conv3 = nn.Conv2d(150, 200, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm2d(200)
        self.conv4 = nn.Conv2d(200, 250, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(250)
        self.fc1 = nn.Linear(1000, 500)
        self.fc2 = nn.Linear(500, nclasses)

        self.localization = LocalNet()
        # # Spatial transformer localization-network
        # self.localization = nn.Sequential(
        #     nn.Conv2d(3, 100, kernel_size=5),
        #     nn.BatchNorm2d(100),
        #     nn.ReLU(True),

        #     nn.Conv2d(100, 200, kernel_size=5),
        #     nn.BatchNorm2d(200),
        #     nn.ReLU(True),

        #     nn.MaxPool2d(2, stride=4),
        #     nn.MaxPool2d(2, stride=2)
        # )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(1700, 100),
            nn.ReLU(True),
            nn.Linear(100, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.fill_(0)
        self.fc_loc[2].bias.data = torch.FloatTensor([1, 0, 0, 0, 1, 0])

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # colorization
        x = self.colorization(x)

        # transform the input
        x = self.stn(x)

        # Perform the usual froward pass
        x = F.relu(F.max_pool2d(self.bn1(self.conv1(x)), 2))
        x = F.relu(F.max_pool2d(F.dropout2d(self.bn2(self.conv2(x))), 2))
        x = F.relu(F.max_pool2d(F.dropout2d(self.bn3(self.conv3(x))), 2))
        x = F.relu(F.max_pool2d(F.dropout2d(self.bn4(self.conv4(x))), 2))
        x = x.view(-1, 1000)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


if __name__ == '__main__':
    resnet = ResNet(BasicBlock, [3, 3, 3, 3])
    resnet.cuda()
