#-*-coding:utf-8-*-
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from util.binary_modules import HardBinaryConv,channel_w
# from torchsummary import summary


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.binary_conv = HardBinaryConv(inplanes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.hardtanh = nn.Hardtanh(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.binary_conv(x)
        out = self.bn1(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.hardtanh(out)

        return out


class Cell2_block(nn.Module):
    def __init__(self, block, in_ch, out_ch,stride=1,downsample=None):
        super(Cell2_block, self).__init__()
        self.block1 = block(in_ch, out_ch,stride=stride,downsample=downsample)
        self.block2 = block(in_ch, out_ch,stride=stride,downsample=downsample)
        self.dropout1 = channel_w(out_ch)
        self.dropout2 = channel_w(out_ch)

    def forward(self,x):
        x1, x2 = x

        out1 = self.block1(x1)
        out2 = self.block2(x2)

        resx1 = self.dropout1(out1)
        resx2 = self.dropout2(out2)

        out1 = out1 + resx2
        out2 = out2 + resx1

        return out1, out2

class Cell3_block(nn.Module):
    def __init__(self, block, in_ch, out_ch,stride=1,downsample=None):
        super(Cell3_block, self).__init__()
        self.block1 = block(in_ch, out_ch,stride=stride,downsample=downsample)
        self.block2 = block(in_ch, out_ch,stride=stride,downsample=downsample)
        self.block3 = block(in_ch, out_ch, stride=stride, downsample=downsample)
        self.dropout1 = channel_w(out_ch)
        self.dropout2 = channel_w(out_ch)
        self.dropout3 = channel_w(out_ch)
        self.dropout4 = channel_w(out_ch)
        self.dropout5 = channel_w(out_ch)
        self.dropout6 = channel_w(out_ch)

    def forward(self,x):
        x1, x2, x3 = x

        out1 = self.block1(x1)
        out2 = self.block2(x2)
        out3 = self.block3(x3)

        resx1_1 = self.dropout1(out1)
        resx1_2 = self.dropout2(out1)
        resx2_1 = self.dropout3(out2)
        resx2_2 = self.dropout4(out2)
        resx3_1 = self.dropout5(out3)
        resx3_2 = self.dropout6(out3)

        out1 = out1 + resx2_1 + resx3_2
        out2 = out2 + resx1_1 + resx3_1
        out3 = out3 + resx1_2 + resx2_2

        return out1, out2, out3

class Cell4_block(nn.Module):
    def __init__(self, block, in_ch, out_ch,stride=1,downsample=None):
        super(Cell4_block, self).__init__()
        self.block1 = block(in_ch, out_ch,stride=stride,downsample=downsample)
        self.block2 = block(in_ch, out_ch,stride=stride,downsample=downsample)
        self.block3 = block(in_ch, out_ch, stride=stride, downsample=downsample)
        self.block4 = block(in_ch, out_ch, stride=stride, downsample=downsample)
        self.dropout1 = channel_w(out_ch)
        self.dropout2 = channel_w(out_ch)
        self.dropout3 = channel_w(out_ch)
        self.dropout4 = channel_w(out_ch)
        self.dropout5 = channel_w(out_ch)
        self.dropout6 = channel_w(out_ch)
        self.dropout7 = channel_w(out_ch)
        self.dropout8 = channel_w(out_ch)
        self.dropout9 = channel_w(out_ch)
        self.dropout10 = channel_w(out_ch)
        self.dropout11 = channel_w(out_ch)
        self.dropout12 = channel_w(out_ch)

    def forward(self,x):
        x1, x2, x3,x4 = x

        out1 = self.block1(x1)
        out2 = self.block2(x2)
        out3 = self.block3(x3)
        out4 = self.block4(x3)

        resx1_2 = self.dropout1(out1)
        resx1_3 = self.dropout2(out1)
        resx1_4 = self.dropout3(out1)

        resx2_1 = self.dropout4(out2)
        resx2_3 = self.dropout5(out2)
        resx2_4 = self.dropout6(out2)

        resx3_1 = self.dropout7(out3)
        resx3_2 = self.dropout8(out3)
        resx3_4 = self.dropout9(out3)

        resx4_1 = self.dropout10(out4)
        resx4_2 = self.dropout11(out4)
        resx4_3 = self.dropout12(out4)


        out1 = out1 + resx2_1 + resx3_1 + resx4_1
        out2 = out2 + resx1_2 + resx3_2 + resx4_2
        out3 = out3 + resx1_3 + resx2_3 + resx4_3
        out4 = out4 + resx1_4 + resx2_4 + resx3_4

        return out1, out2, out3, out4
class Cell5_block(nn.Module):
    def __init__(self, block, in_ch, out_ch,stride=1,downsample=None):
        super(Cell5_block, self).__init__()
        self.block1 = block(in_ch, out_ch,stride=stride,downsample=downsample)
        self.block2 = block(in_ch, out_ch,stride=stride,downsample=downsample)
        self.block3 = block(in_ch, out_ch, stride=stride, downsample=downsample)
        self.block4 = block(in_ch, out_ch, stride=stride, downsample=downsample)
        self.block5 = block(in_ch, out_ch, stride=stride, downsample=downsample)
        self.dropout1 = channel_w(out_ch)
        self.dropout2 = channel_w(out_ch)
        self.dropout3 = channel_w(out_ch)
        self.dropout4 = channel_w(out_ch)
        self.dropout5 = channel_w(out_ch)
        self.dropout6 = channel_w(out_ch)
        self.dropout7 = channel_w(out_ch)
        self.dropout8 = channel_w(out_ch)
        self.dropout9 = channel_w(out_ch)
        self.dropout10 = channel_w(out_ch)
        self.dropout11 = channel_w(out_ch)
        self.dropout12 = channel_w(out_ch)
        self.dropout13 = channel_w(out_ch)
        self.dropout14 = channel_w(out_ch)
        self.dropout15 = channel_w(out_ch)
        self.dropout16 = channel_w(out_ch)
        self.dropout17 = channel_w(out_ch)
        self.dropout18 = channel_w(out_ch)
        self.dropout19 = channel_w(out_ch)
        self.dropout20 = channel_w(out_ch)

    def forward(self,x):
        x1, x2, x3,x4,x5 = x

        out1 = self.block1(x1)
        out2 = self.block2(x2)
        out3 = self.block3(x3)
        out4 = self.block4(x4)
        out5 = self.block5(x5)

        resx1_2 = self.dropout1(out1)
        resx1_3 = self.dropout2(out1)
        resx1_4 = self.dropout3(out1)
        resx1_5 = self.dropout4(out1)

        resx2_1 = self.dropout5(out2)
        resx2_3 = self.dropout6(out2)
        resx2_4 = self.dropout7(out2)
        resx2_5 = self.dropout8(out2)

        resx3_1 = self.dropout9(out3)
        resx3_2 = self.dropout10(out3)
        resx3_4 = self.dropout11(out3)
        resx3_5 = self.dropout12(out3)

        resx4_1 = self.dropout13(out4)
        resx4_2 = self.dropout14(out4)
        resx4_3 = self.dropout15(out4)
        resx4_5 = self.dropout16(out4)

        resx5_1 = self.dropout17(out5)
        resx5_2 = self.dropout18(out5)
        resx5_3 = self.dropout19(out5)
        resx5_4 = self.dropout20(out5)


        out1 = out1 + resx2_1 + resx3_1 + resx4_1 + resx5_1
        out2 = out2 + resx1_2 + resx3_2 + resx4_2 + resx5_2
        out3 = out3 + resx1_3 + resx2_3 + resx4_3 + resx5_3
        out4 = out4 + resx1_4 + resx2_4 + resx3_4 + resx5_4
        out5 = out5 + resx1_5 + resx2_5 + resx3_5 + resx4_5

        return out1, out2, out3, out4,out5
class Cell6_block(nn.Module):
    def __init__(self, block, in_ch, out_ch,stride=1,downsample=None):
        super(Cell6_block, self).__init__()
        self.block1 = block(in_ch, out_ch,stride=stride,downsample=downsample)
        self.block2 = block(in_ch, out_ch,stride=stride,downsample=downsample)
        self.block3 = block(in_ch, out_ch, stride=stride, downsample=downsample)
        self.block4 = block(in_ch, out_ch, stride=stride, downsample=downsample)
        self.block5 = block(in_ch, out_ch, stride=stride, downsample=downsample)
        self.block5 = block(in_ch, out_ch, stride=stride, downsample=downsample)
        self.dropout1 = channel_w(out_ch)
        self.dropout2 = channel_w(out_ch)
        self.dropout3 = channel_w(out_ch)
        self.dropout4 = channel_w(out_ch)
        self.dropout5 = channel_w(out_ch)
        self.dropout6 = channel_w(out_ch)
        self.dropout7 = channel_w(out_ch)
        self.dropout8 = channel_w(out_ch)
        self.dropout9 = channel_w(out_ch)
        self.dropout10 = channel_w(out_ch)
        self.dropout11 = channel_w(out_ch)
        self.dropout12 = channel_w(out_ch)
        self.dropout13 = channel_w(out_ch)
        self.dropout14 = channel_w(out_ch)
        self.dropout15 = channel_w(out_ch)
        self.dropout16 = channel_w(out_ch)
        self.dropout17 = channel_w(out_ch)
        self.dropout18 = channel_w(out_ch)
        self.dropout19 = channel_w(out_ch)
        self.dropout20 = channel_w(out_ch)
        self.dropout21 = channel_w(out_ch)
        self.dropout22 = channel_w(out_ch)
        self.dropout23 = channel_w(out_ch)
        self.dropout24 = channel_w(out_ch)
        self.dropout25 = channel_w(out_ch)
        self.dropout26 = channel_w(out_ch)
        self.dropout27 = channel_w(out_ch)
        self.dropout28 = channel_w(out_ch)
        self.dropout29 = channel_w(out_ch)
        self.dropout30 = channel_w(out_ch)

    def forward(self,x):
        x1, x2, x3,x4,x5,x6 = x

        out1 = self.block1(x1)
        out2 = self.block2(x2)
        out3 = self.block3(x3)
        out4 = self.block4(x4)
        out5 = self.block5(x5)
        out6 = self.block5(x6)

        resx1_2 = self.dropout1(out1)
        resx1_3 = self.dropout2(out1)
        resx1_4 = self.dropout3(out1)
        resx1_5 = self.dropout4(out1)
        resx1_6 = self.dropout5(out1)

        resx2_1 = self.dropout6(out2)
        resx2_3 = self.dropout7(out2)
        resx2_4 = self.dropout8(out2)
        resx2_5 = self.dropout9(out2)
        resx2_6 = self.dropout10(out2)

        resx3_1 = self.dropout11(out3)
        resx3_2 = self.dropout12(out3)
        resx3_4 = self.dropout13(out3)
        resx3_5 = self.dropout14(out3)
        resx3_6 = self.dropout15(out3)

        resx4_1 = self.dropout16(out4)
        resx4_2 = self.dropout17(out4)
        resx4_3 = self.dropout18(out4)
        resx4_5 = self.dropout19(out4)
        resx4_6 = self.dropout20(out4)

        resx5_1 = self.dropout21(out5)
        resx5_2 = self.dropout22(out5)
        resx5_3 = self.dropout23(out5)
        resx5_4 = self.dropout24(out5)
        resx5_6 = self.dropout25(out5)

        resx6_1 = self.dropout26(out6)
        resx6_2 = self.dropout27(out6)
        resx6_3 = self.dropout28(out6)
        resx6_4 = self.dropout29(out6)
        resx6_5 = self.dropout30(out6)


        out1 = out1 + resx2_1 + resx3_1 + resx4_1 + resx5_1 + resx6_1
        out2 = out2 + resx1_2 + resx3_2 + resx4_2 + resx5_2 + resx6_2
        out3 = out3 + resx1_3 + resx2_3 + resx4_3 + resx5_3 + resx6_3
        out4 = out4 + resx1_4 + resx2_4 + resx3_4 + resx5_4 + resx6_4
        out5 = out5 + resx1_5 + resx2_5 + resx3_5 + resx4_5 + resx6_5
        out6 = out6 + resx1_6 + resx2_6 + resx3_6 + resx4_6 + resx5_6

        return out1, out2, out3, out4,out5,out6

class BiRealNet(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        super(BiRealNet, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hardtanh = nn.Hardtanh(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn = nn.BatchNorm1d(64)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.hardtanh(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.bn(x)
        x = self.fc(x)

        return x

class CBN2_BiRealNet(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        super(CBN2_BiRealNet, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hardtanh = nn.Hardtanh()
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn = nn.BatchNorm1d(64*2)
        self.fc = nn.Linear(64*2, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = LambdaLayer(lambda x:
                                       F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0))

        layers = []
        layers.append(Cell2_block(block,self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(Cell2_block(block,self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x):
        x1 = self.hardtanh(self.bn1(self.conv1(x)))

        x1,x2 = self.layer1([x1,x1])
        x1,x2 = self.layer2([x1,x2])
        x1,x2 = self.layer3([x1,x2])

        out = torch.cat((x1, x2), dim=1)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.bn(out)
        out = self.fc(out)

        return out

class CBN3_BiRealNet(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        super(CBN3_BiRealNet, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hardtanh = nn.Hardtanh(inplace=True)
        # self.conv2 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        # self.bn2 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn = nn.BatchNorm1d(64*3)
        self.fc = nn.Linear(64*3, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = LambdaLayer(lambda x:
                                       F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0))

        layers = []
        layers.append(Cell3_block(block,self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(Cell3_block(block,self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.hardtanh(self.bn1(self.conv1(x)))

        x1,x2,x3 = self.layer1([x1,x1,x1])
        x1,x2,x3 = self.layer2([x1,x2,x3])
        x1,x2,x3 = self.layer3([x1,x2,x3])
        out = torch.cat((x1, x2,x3), dim=1)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.bn(out)
        out = self.fc(out)

        return out

class CBN4_BiRealNet(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        super(CBN4_BiRealNet, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hardtanh = nn.Hardtanh(inplace=True)

        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn = nn.BatchNorm1d(64*4)
        self.fc = nn.Linear(64*4, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = LambdaLayer(lambda x:
                                       F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0))

        layers = []
        layers.append(Cell4_block(block,self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(Cell4_block(block,self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.hardtanh(self.bn1(self.conv1(x)))

        x1,x2,x3,x4 = self.layer1([x1,x1,x1,x1])
        x1,x2,x3,x4 = self.layer2([x1,x2,x3,x4])
        x1,x2,x3,x4 = self.layer3([x1,x2,x3,x4])
        out = torch.cat((x1, x2,x3,x4), dim=1)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.bn(out)
        out = self.fc(out)

        return out
class CBN5_BiRealNet(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        super(CBN5_BiRealNet, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hardtanh = nn.Hardtanh(inplace=True)

        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn = nn.BatchNorm1d(64*5)
        self.fc = nn.Linear(64*5, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = LambdaLayer(lambda x:
                                       F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0))

        layers = []
        layers.append(Cell5_block(block,self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(Cell5_block(block,self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.hardtanh(self.bn1(self.conv1(x)))

        x1,x2,x3,x4,x5 = self.layer1([x1,x1,x1,x1,x1])
        x1,x2,x3,x4,x5 = self.layer2([x1,x2,x3,x4,x5])
        x1,x2,x3,x4,x5 = self.layer3([x1,x2,x3,x4,x5])
        out = torch.cat((x1, x2,x3,x4,x5), dim=1)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.bn(out)
        out = self.fc(out)

        return out
class CBN6_BiRealNet(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        super(CBN6_BiRealNet, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hardtanh = nn.Hardtanh(inplace=True)

        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn = nn.BatchNorm1d(64*6)
        self.fc = nn.Linear(64*6, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = LambdaLayer(lambda x:
                                       F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0))

        layers = []
        layers.append(Cell6_block(block,self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(Cell6_block(block,self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.hardtanh(self.bn1(self.conv1(x)))

        x1,x2,x3,x4,x5,x6 = self.layer1([x1,x1,x1,x1,x1,x1])
        x1,x2,x3,x4,x5,x6 = self.layer2([x1,x2,x3,x4,x5,x6])
        x1,x2,x3,x4,x5,x6 = self.layer3([x1,x2,x3,x4,x5,x6])
        out = torch.cat((x1, x2,x3,x4,x5,x6), dim=1)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.bn(out)
        out = self.fc(out)

        return out




def BirealNet20(pretrained=False, **kwargs):
    model = BiRealNet(BasicBlock, [6,6,6], **kwargs)
    return model

def CBN20_K2(pretrained=False, **kwargs):
    """Constructs a BiRealNet-18 model. """
    model = CBN2_BiRealNet(BasicBlock, [6,6,6], **kwargs)
    return model

def CBN20_K3(pretrained=False, **kwargs):
    """Constructs a BiRealNet-18 model. """
    model = CBN3_BiRealNet(BasicBlock, [6,6,6], **kwargs)
    return model

def CBN20_K4(pretrained=False, **kwargs):
    """Constructs a BiRealNet-18 model. """
    model = CBN4_BiRealNet(BasicBlock, [6,6,6], **kwargs)
    return model

def CBN20_K5(pretrained=False, **kwargs):
    """Constructs a BiRealNet-18 model. """
    model = CBN5_BiRealNet(BasicBlock, [6,6,6], **kwargs)
    return model
def CBN20_K6(pretrained=False, **kwargs):
    """Constructs a BiRealNet-18 model. """
    model = CBN6_BiRealNet(BasicBlock, [6,6,6], **kwargs)
    return model

