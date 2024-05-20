import torch
from torch import nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_operation = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv_operation(x)

class DownSampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.doubleconv = DoubleConv(in_channels=in_channels, out_channels=out_channels)
        # maxpooling membagi ukuran image jadi 2 kali lebih kecil atau setengahnya
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        conv_result = self.doubleconv(x)
        pool_result = self.pool(conv_result)

        return conv_result, pool_result

class UpSampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.doubleconv = DoubleConv(in_channels=in_channels, out_channels=out_channels)
        # transpose convolution
        self.up = nn.ConvTranspose2d()