import torch.nn as nn
import torch

import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, pad=1, stride=1, bias=True, depthwise=False):
        super().__init__()
        if depthwise:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=pad, groups=in_channels, bias=bias),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias),
                nn.ReLU(),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=bias),
                nn.ReLU(),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        return self.block(x)

class BallTrackerNet(nn.Module):
    def __init__(self, out_channels=14):
        super().__init__()
        self.out_channels = out_channels

        self.conv1 = ConvBlock(in_channels=1, out_channels=32, depthwise=True)
        self.conv2 = ConvBlock(in_channels=32, out_channels=32, depthwise=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = ConvBlock(in_channels=32, out_channels=64, depthwise=True)
        self.conv4 = ConvBlock(in_channels=64, out_channels=64, depthwise=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = ConvBlock(in_channels=64, out_channels=128, depthwise=True)
        self.conv6 = ConvBlock(in_channels=128, out_channels=128, depthwise=True)
        self.conv7 = ConvBlock(in_channels=128, out_channels=128, depthwise=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv8 = ConvBlock(in_channels=128, out_channels=256, depthwise=True)
        self.conv9 = ConvBlock(in_channels=256, out_channels=256, depthwise=True)
        self.conv10 = ConvBlock(in_channels=256, out_channels=256, depthwise=True)
        self.ups1 = nn.Upsample(scale_factor=2)
        self.conv11 = ConvBlock(in_channels=256, out_channels=128, depthwise=True)
        self.conv12 = ConvBlock(in_channels=128, out_channels=128, depthwise=True)
        self.conv13 = ConvBlock(in_channels=128, out_channels=128, depthwise=True)
        self.ups2 = nn.Upsample(scale_factor=2)
        self.conv14 = ConvBlock(in_channels=128, out_channels=64, depthwise=True)
        self.conv15 = ConvBlock(in_channels=64, out_channels=64, depthwise=True)
        self.ups3 = nn.Upsample(scale_factor=2)
        self.conv16 = ConvBlock(in_channels=64, out_channels=32, depthwise=True)
        self.conv17 = ConvBlock(in_channels=32, out_channels=32, depthwise=True)
        self.conv18 = ConvBlock(in_channels=32, out_channels=self.out_channels, depthwise=True)

        self._init_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.ups1(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.ups2(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.ups3(x)
        x = self.conv16(x)
        x = self.conv17(x)
        x = self.conv18(x)
        return x

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.uniform_(module.weight, -0.05, 0.05)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
                
if __name__ == '__main__':
    device = 'cpu'
    model = BallTrackerNet().to(device)
    inp = torch.rand(1, 1, 360, 640)
    out = model(inp)
    print('out = {}'.format(out.shape))
    
    