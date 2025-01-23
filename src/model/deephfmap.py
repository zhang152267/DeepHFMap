import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["DeepHFMap"]


class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()

        self.bn_acti = bn_acti

        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        if self.bn_acti:
            self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv(input)

        if self.bn_acti:
            output = self.bn_prelu(output)

        return output


class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)

        return output


class DABModule(nn.Module):
    def __init__(self, nIn, d=1, kSize=3, dkSize=3):
        super().__init__()

        self.bn_relu_1 = BNPReLU(nIn)
        self.conv3x3 = Conv(nIn, nIn // 2, kSize, 1, padding=1, bn_acti=True)

        self.dconv3x1 = Conv(nIn // 2, nIn // 2, (dkSize, 1), 1,
                             padding=(1, 0), groups=nIn // 2, bn_acti=True)
        self.eca1 = ecamodule(nIn//2)
        self.dconv1x3 = Conv(nIn // 2, nIn // 2, (1, dkSize), 1,
                             padding=(0, 1), groups=nIn // 2, bn_acti=True)
        self.ddconv3x1 = Conv(nIn // 2, nIn // 2, (dkSize, 1), 1,
                              padding=(1 * d, 0), dilation=(d, 1), groups=nIn // 2, bn_acti=True)
        self.eca2 = ecamodule(nIn//2)
        self.ddconv1x3 = Conv(nIn // 2, nIn // 2, (1, dkSize), 1,
                              padding=(0, 1 * d), dilation=(1, d), groups=nIn // 2, bn_acti=True)

        self.bn_relu_2 = BNPReLU(nIn)
        self.eca = ecamodule(nIn)
        self.conv1x1 = Conv(nIn, nIn, 1, 1, padding=0, bn_acti=False)

    def forward(self, input):
        output = self.bn_relu_1(input)
        output = self.conv3x3(output)
        br1 = self.dconv3x1(output)
        br1 = self.eca1(br1)
        br1 = self.dconv1x3(br1)
        br2 = self.ddconv3x1(output)
        br2 = self.eca2(br2)
        br2 = self.ddconv1x3(br2)

        output = torch.cat([br1,br2],1)
        output = self.bn_relu_2(output)
        output = self.eca(output)
        output = self.conv1x1(output)

        return output + input


class DownSamplingBlock1(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        self.nIn = nIn
        self.nOut = nOut
        self.con = nn.Conv2d(nIn, nIn-29*2, kernel_size=1, stride=1, bias=True)
        self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output0 = F.interpolate(input, [32,32], mode='bilinear', align_corners=False)
        output1 = self.con(output0)
        output = torch.cat([output0, output1], 1)
        output = self.bn_prelu(output)
        return output


class DownSamplingBlock2(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        self.nIn = nIn
        self.nOut = nOut
        self.con = nn.Conv2d(nIn, nOut, kernel_size=1, stride=1, bias=True)
        self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = F.interpolate(input, [16,16], mode='bilinear', align_corners=False)
        output = self.con(output)
        output = self.bn_prelu(output)
        return output


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
        self.res = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, bias=True)
    def forward(self, x):
        x1  = self.res(x)
        #print(x1.shape)
        x = self.up(x)
        x1 = F.interpolate(x1, x.size()[2:], mode='bilinear', align_corners=False)
        return x+x1
    

class ecamodule(nn.Module):

    def __init__(self, channel, k_size=3):
        super(ecamodule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class DeepHFMap(nn.Module):
    def __init__(self, classes=1, block_1=3, block_2=6):
        super().__init__()
        self.init_conv = nn.Sequential(
            Conv(29, 32, 3, 2, padding=1, bn_acti=True),
            Conv(32, 32, 3, 1, padding=1, bn_acti=True),
            Conv(32, 32, 3, 1, padding=1, bn_acti=True),
        )
        self.initeca = ecamodule(32)

        self.bn_prelu_1 = BNPReLU(32 + 29)

        # DAB Block 1
        self.downsample_1 = DownSamplingBlock1(32 + 29, 64)
        self.DAB_Block_1 = nn.Sequential()
        for i in range(0, block_1):
            self.DAB_Block_1.add_module("DAB_Module_1_" + str(i), DABModule(64, d=2))
        self.bn_prelu_2 = BNPReLU(128 + 29)

        # DAB Block 2
        dilation_block_2 = [4, 4, 8, 8, 16, 16]
        self.downsample_2 = DownSamplingBlock2(128 + 29, 128)
        self.DAB_Block_2 = nn.Sequential()
        for i in range(0, block_2):
            self.DAB_Block_2.add_module("DAB_Module_2_" + str(i),
                                        DABModule(128, d=dilation_block_2[i]))
        self.bn_prelu_3 = BNPReLU(256 + 29)
        self.con = nn.Sequential(Conv(256+29, 256, 1, 1, padding=0))
        self.upcon1 = up_conv(ch_in=256, ch_out=128)
        self.upcon2 = up_conv(ch_in=128 , ch_out=64)
        self.finaleca = ecamodule(64)
        self.Conv_1x1 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
    def forward(self, input):
        b,c,h,w = input.shape
        output0 = self.init_conv(input)
        output0 = self.initeca(output0)
        output0 = F.interpolate(output0, [int(0.75*h),int(0.75*w)], mode='bilinear', align_corners=False)
        down_1 = F.interpolate(input, [int(0.75*h),int(0.75*w)], mode='bilinear', align_corners=False)
        down_2 = F.interpolate(input, [int(0.5*h),int(0.5*w)], mode='bilinear', align_corners=False)
        down_3 = F.interpolate(input, [int(0.25*h),int(0.25*w)], mode='bilinear', align_corners=False)
        output0_cat = self.bn_prelu_1(torch.cat([output0, down_1], 1))

        # DAB Block 1
        output1_0 = self.downsample_1(output0_cat)
        output1 = self.DAB_Block_1(output1_0)
        output1_cat = self.bn_prelu_2(torch.cat([output1, output1_0, down_2], 1))

        # DAB Block 2
        output2_0 = self.downsample_2(output1_cat)
        output2 = self.DAB_Block_2(output2_0)
        output2_cat = self.bn_prelu_3(torch.cat([output2, output2_0, down_3], 1))

        out = self.con(output2_cat)
        out = self.upcon1(out)
        out = self.upcon2(out)
        out = self.finaleca(out)
        out = self.Conv_1x1(out)
        return out



def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

if __name__ == '__main__':
    input = torch.rand(4, 12, 64, 64)
    model = DeepHFMap()
    output= model(input)
    #print_network(model)