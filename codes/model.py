
import pvtv2
import resnet
from resnet import resnet34
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
import scipy.stats as st
torch.autograd.set_detect_anomaly(True)


class DilatedConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.c1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=3, dilation=3),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

        self.c3 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=6, dilation=6),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

        self.c4 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=9, dilation=9),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

        self.c5 = nn.Sequential(
            nn.Conv2d(out_c*4, out_c, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

    def forward(self, inputs):
        x1 = self.c1(inputs)
        x2 = self.c2(inputs)
        x3 = self.c3(inputs)
        x4 = self.c4(inputs)
        x = torch.cat([x1, x2, x3, x4], axis=1)
        x = self.c5(x)
        return x


class SpatialAttention(nn.Module):  #SA
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)

class ChannelAttention(nn.Module):   #CA
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)
class CSAB(nn.Module):
    def __init__(self, in_channel):
        super(CSAB, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(in_channel)

    def forward(self, x):
        x1_ca = x.mul(self.ca(x))
        x1_sa = x1_ca.mul(self.sa(x1_ca))
        x = x + x1_sa
        return x

class  ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.relu = nn.ReLU()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c)
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_c)
        )

    def forward(self, inputs):
        x1 = self.conv(inputs)
        x2 = self.shortcut(inputs)
        x = self.relu(x1 + x2)
        return x

class Conv2D(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, bias=True, act=True):
        super().__init__()

        self.act = act
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act == True:
            x = self.relu(x)
        return x
class SAB(nn.Module):
    def __init__(self, channels, padding=0, groups=1, matmul_norm=True):
        super(SAB, self).__init__()
        self.channels = channels
        self.padding = padding
        self.groups = groups
        self.matmul_norm = matmul_norm
        self._channels = channels//8

        self.conv_query = nn.Conv2d(in_channels=channels, out_channels=self._channels, kernel_size=1, groups=groups)
        self.conv_key = nn.Conv2d(in_channels=channels, out_channels=self._channels, kernel_size=1, groups=groups)
        self.conv_value = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, groups=groups)

        self.conv_output = Conv2D(in_c=channels, out_c=channels, kernel_size=3, padding=1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # Get query, key, value tensors
        query = self.conv_query(x).view(batch_size, -1, height*width)
        key = self.conv_key(x).view(batch_size, -1, height*width)
        value = self.conv_value(x).view(batch_size, -1, height*width)

        # Apply transpose to swap dimensions for matrix multiplication
        query = query.permute(0, 2, 1).contiguous()  # (batch_size, height*width, channels//8)
        value = value.permute(0, 2, 1).contiguous()  # (batch_size, height*width, channels)

        # Compute attention map
        attention_map = torch.matmul(query, key)
        if self.matmul_norm:
            attention_map = (self._channels**-.5) * attention_map
        attention_map = torch.softmax(attention_map, dim=-1)

        # Apply attention
        out = torch.matmul(attention_map, value)
        out = out.permute(0, 2, 1).contiguous().view(batch_size, channels, height, width)

        # Apply output convolution
        out = self.conv_output(out)
        out = out + x

        return out
class DilatedConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.c1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=3, dilation=3),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

        self.c3 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=6, dilation=6),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

        self.c4 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=9, dilation=9),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

        self.c5 = nn.Sequential(
            nn.Conv2d(out_c*4, out_c, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

    def forward(self, inputs):
        x1 = self.c1(inputs)
        x2 = self.c2(inputs)
        x3 = self.c3(inputs)
        x4 = self.c4(inputs)
        x = torch.cat([x1, x2, x3, x4], axis=1)
        x = self.c5(x)
        return x
class DDANet(nn.Module):
    def __init__(self):
        super().__init__()

        """ Encoder """
        backbone = resnet34()
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.layer1 = nn.Sequential(backbone.maxpool, backbone.layer1)
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.a1 = DilatedConv(64, 64)
        self.a2 = DilatedConv(128, 128)
        self.a3 = DilatedConv(256, 256)
        self.sab = SAB(512)

        self.decoder4 = ResidualBlock(512, 256)
        self.decoder3 = ResidualBlock(256, 128)
        self.decoder2 = ResidualBlock(128, 64)
        self.decoder1 = ResidualBlock(64, 64)

        self.csab1 = CSAB(256)
        self.csab2 = CSAB(128)
        self.csab3 = CSAB(64)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.finaldeconv1 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.finalrelu1 = nn.ReLU()
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nn.ReLU()
        self.finalconv3 = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, inputs):

        """ Encoder """
        e0 = inputs
        e1 = self.layer0(e0)  ## [-1, 64, h/2, w/2]
        e2 = self.layer1(e1)  ## [-1, 64, h/4, w/4]
        e3 = self.layer2(e2)  ## [-1, 128, h/8, w/8]
        e4 = self.layer3(e3)  ## [-1, 256, h/16, w/16]
        e5 = self.layer4(e4)  ## [-1, 512, h/16, w/16]

        e2 = self.a1(e2)
        e3 = self.a2(e3)
        e4 = self.a3(e4)

        e41 = self.sab(e5)

        e42 = self.up1(e41)
        d43 = self.decoder4(e42) + e4
        d44 =self.csab1(d43)

        d3 = self.up1(d44)
        d31 = self.decoder3(d3) + e3
        d32 =self.csab2(d31)

        d2 = self.up1(d32)
        d21 = self.decoder2(d2) + e2
        d22 =self.csab3(d21)

        d1 = self.up1(d22)
        d11 = self.decoder1(d1) + e1


        out = self.finaldeconv1(d11)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        y = self.finalconv3(out)
        return y
if __name__ == "__main__":
    inputs = torch.randn((4, 3, 512, 512))
    model = DDANet()
    y = model(inputs)
    print(y.shape)
