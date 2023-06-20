# Imports
import torch.nn as nn
import torch.nn.functional as F


class SimpleNeuralNet(nn.Module):
    """
    SimpleNeuralNet és una xarxa totalment connectada amb 1 sola capa oculta.
    """

    def __init__(self, in_size, hidden_size=50):
        super(SimpleNeuralNet, self).__init__()
        self.fc1 = nn.Linear(in_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Transformem el tensor de cada imatge en un vector unidimensional
        x = x.reshape(x.shape[0], -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        x = self.sigmoid(x)

        return x


class SimpleConvNet(nn.Module):
    """
    SimpleConvNet és una xarxa convolucional amb 1 capa de convolució i 1 capa totalment connectada.
    """

    def __init__(self, in_channels=3):
        super(SimpleConvNet, self).__init__()
        KERNEL_SIZE = 3
        CONV_1_OUT = 8

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=CONV_1_OUT, kernel_size=KERNEL_SIZE)

        self.fc1 = nn.Linear(CONV_1_OUT*24*24, 1)

        self.pool = nn.MaxPool2d(2, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.conv1(x)))

        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.sigmoid(x)

        return x


class ConvNet(nn.Module):
    """
    ConvNet és una xarxa convolucional amb 2 capes de convolució i 2 capes totalment connectades.
    """

    def __init__(self, in_channels=3):
        super(ConvNet, self).__init__()
        KERNEL_SIZE = 3
        CONV_1_OUT = 16
        CONV_2_OUT = 16
        FC_1_OUT = 64

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=CONV_1_OUT, kernel_size=KERNEL_SIZE)
        self.bn1 = nn.BatchNorm2d(CONV_1_OUT)

        self.conv2 = nn.Conv2d(in_channels=CONV_1_OUT,
                               out_channels=CONV_2_OUT, kernel_size=KERNEL_SIZE)
        self.bn2 = nn.BatchNorm2d(CONV_2_OUT)

        self.fc1 = nn.Linear(CONV_2_OUT*11*11, FC_1_OUT)
        self.fc2 = nn.Linear(FC_1_OUT, 1)

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.bn1(self.pool(F.leaky_relu(self.conv1(x))))
        x = self.bn2(self.pool(F.leaky_relu(self.conv2(x))))

        x = x.view(x.shape[0], -1)

        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)

        x = F.sigmoid(x)

        return x


class LargeConvNet(nn.Module):
    """
    LargeConvNet és una xarxa convolucional amb 3 capes de convolució i 3 capes totalment connectades.
    """

    def __init__(self, in_channels=3):
        super(LargeConvNet, self).__init__()
        KERNEL_SIZE = 3
        CONV_1_OUT = 16
        CONV_2_OUT = 16
        CONV_3_OUT = 16
        FC_1_OUT = 128
        FC_2_OUT = 64

        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=CONV_1_OUT, kernel_size=KERNEL_SIZE, )
        self.bn1 = nn.BatchNorm2d(CONV_1_OUT)

        self.conv2 = nn.Conv2d(in_channels=CONV_1_OUT,
                               out_channels=CONV_2_OUT, kernel_size=KERNEL_SIZE)
        self.bn2 = nn.BatchNorm2d(CONV_2_OUT)

        self.conv3 = nn.Conv2d(in_channels=CONV_2_OUT,
                               out_channels=CONV_3_OUT, kernel_size=KERNEL_SIZE)
        self.bn3 = nn.BatchNorm2d(CONV_3_OUT)

        self.fc1 = nn.Linear(CONV_3_OUT*4*4, FC_1_OUT)
        self.fc2 = nn.Linear(FC_1_OUT, FC_2_OUT)
        self.fc3 = nn.Linear(FC_2_OUT, 1)

        self.pool = nn.MaxPool2d(2, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.bn1(self.pool(F.leaky_relu(self.conv1(x))))
        x = self.bn2(self.pool(F.leaky_relu(self.conv2(x))))
        x = self.bn3(self.pool(F.leaky_relu(self.conv3(x))))

        x = x.view(x.size(0), -1)

        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)

        x = self.sigmoid(x)

        return x
