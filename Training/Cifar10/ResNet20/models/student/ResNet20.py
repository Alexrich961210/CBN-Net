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

# class linear_a(nn.Module):
#     def __init__(self,p):
#         super(linear_a, self).__init__()
#         self.w1 = torch.nn.Parameter(torch.rand(1)*0.01, requires_grad=True)
#         self.dropout = nn.Dropout(p)
#
#     def forward(self,x):
#         output = self.w1 * x
#         output = self.dropout(output)
#         return output

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class conv1x1_a(nn.Module):
    def __init__(self,in_planes, out_planes, p):
        super(conv1x1_a, self).__init__()
        self.conv = nn.Parameter(torch.rand(1)*0.001, requires_grad=True)
        self.dropout = nn.Dropout2d(p)

    def forward(self,x):
        out = self.conv*x
        out = self.dropout(out)
        return out

class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        # self.move0 = LearnableBias(inplanes)
        # self.binary_activation = BinaryActivation()
        self.binary_conv = HardBinaryConv(inplanes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)

        # self.move1 = LearnableBias(planes)
        # self.prelu = nn.PReLU()
        # self.move2 = LearnableBias(planes)
        self.hardtanh = nn.Hardtanh(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        # out = self.move0(x)
        # out = self.binary_activation(out)
        out = self.binary_conv(x)
        out = self.bn1(out)
        # print("the shape of out",out.shape)
        if self.downsample is not None:
            residual = self.downsample(x)
            # print("the shape of residual",residual.shape)
        out += residual
        out = self.hardtanh(out)
        # out = self.move1(out)
        # out = self.prelu(out)
        # out = self.move2(out)

        return out


class Mesh2_block(nn.Module):
    def __init__(self, block, in_ch, out_ch,stride=1,downsample=None):
        super(Mesh2_block, self).__init__()

        self.block1 = block(in_ch, out_ch,stride=stride,downsample=downsample)
        self.block2 = block(in_ch, out_ch,stride=stride,downsample=downsample)
        self.dropout1 = channel_w(out_ch)
        self.dropout2 = channel_w(out_ch)
        self.hardtanh = nn.Hardtanh()

    def forward(self,x):
        x1, x2 = x

        out1 = self.block1(x1)
        out2 = self.block2(x2)

        resx1 = self.dropout1(out1)
        resx2 = self.dropout2(out2)

        out1 = out1 + resx2
        out2 = out2 + resx1

        return out1, out2

class Mesh3_block(nn.Module):
    def __init__(self, block, in_ch, out_ch,stride=1,downsample=None):
        super(Mesh3_block, self).__init__()
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

class En2_block(nn.Module):
    def __init__(self, block, in_ch, out_ch,stride=1,downsample=None):
        super(En2_block, self).__init__()
        self.block1 = block(in_ch, out_ch,stride=stride,downsample=downsample)
        self.block2 = block(in_ch, out_ch,stride=stride,downsample=downsample)

    def forward(self,x):
        x1, x2 = x

        out1 = self.block1(x1)
        out2 = self.block2(x2)

        return out1, out2

class En3_block(nn.Module):
    def __init__(self, block, in_ch, out_ch,stride=1,downsample=None):
        super(En3_block, self).__init__()
        self.block1 = block(in_ch, out_ch,stride=stride,downsample=downsample)
        self.block2 = block(in_ch, out_ch,stride=stride,downsample=downsample)
        self.block3 = block(in_ch, out_ch, stride=stride, downsample=downsample)

    def forward(self,x):
        x1, x2, x3 = x

        out1 = self.block1(x1)
        out2 = self.block2(x2)
        out3 = self.block2(x3)

        return out1, out2, out3

class BiRealNet(nn.Module):

    def __init__(self, block, layers, num_classes=10, zero_init_residual=False):
        super(BiRealNet, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(16)
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
        x = self.bn1(self.conv1(x))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.bn(x)
        x = self.fc(x)

        return x

class Mesh3_BiRealNet(nn.Module):

    def __init__(self, block, layers, num_classes=10, zero_init_residual=False):
        super(Mesh3_BiRealNet, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(16)
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
        layers.append(Mesh3_block(block,self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(Mesh3_block(block,self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.bn1(self.conv1(x))

        x1,x2,x3 = self.layer1([x1,x1,x1])
        x1,x2,x3 = self.layer2([x1,x2,x3])
        x1,x2,x3 = self.layer3([x1,x2,x3])
        out = torch.cat((x1, x2,x3), dim=1)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.bn(out)
        out = self.fc(out)

        return out
class En3_BiRealNet(nn.Module):

    def __init__(self, block, layers, num_classes=10, zero_init_residual=False):
        super(En3_BiRealNet, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(16)
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
        layers.append(En3_block(block,self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(En3_block(block,self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.bn1(self.conv1(x))

        x1,x2,x3 = self.layer1([x1,x1,x1])
        x1,x2,x3 = self.layer2([x1,x2,x3])
        x1,x2,x3 = self.layer3([x1,x2,x3])
        out = torch.cat((x1, x2,x3), dim=1)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.bn(out)
        out = self.fc(out)

        return out

class Mesh2_BiRealNet(nn.Module):

    def __init__(self, block, layers, num_classes=10, zero_init_residual=False):
        super(Mesh2_BiRealNet, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hardtanh = nn.Hardtanh()
        # self.conv2 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        # self.bn2 = nn.BatchNorm2d(64)
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
        layers.append(Mesh2_block(block,self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(Mesh2_block(block,self.inplanes, planes))

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

class En2_BiRealNet(nn.Module):

    def __init__(self, block, layers, num_classes=10, zero_init_residual=False):
        super(En2_BiRealNet, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        # self.conv2 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        # self.bn2 = nn.BatchNorm2d(64)
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
        layers.append(En2_block(block,self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(En2_block(block,self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.bn1(self.conv1(x))

        x1,x2 = self.layer1([x1,x1])
        x1,x2 = self.layer2([x1,x2])
        x1,x2 = self.layer3([x1,x2])
        out = torch.cat((x1, x2), dim=1)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.bn(out)
        out = self.fc(out)
        return out

def BirealNet20(pretrained=False, **kwargs):
    model = BiRealNet(BasicBlock, [6,6,6], **kwargs)
    return model

def MeshNet20_K2(pretrained=False, **kwargs):
    """Constructs a BiRealNet-18 model. """
    model = Mesh2_BiRealNet(BasicBlock, [6,6,6], **kwargs)
    return model

def MeshNet20_K3(pretrained=False, **kwargs):
    """Constructs a BiRealNet-18 model. """
    model = Mesh3_BiRealNet(BasicBlock, [6,6,6], **kwargs)
    return model

def En2Res20(pretrained=False, **kwargs):
    """Constructs a BiRealNet-18 model. """
    model = En2_BiRealNet(BasicBlock, [6,6,6], **kwargs)
    return model

def En3Res20(pretrained=False, **kwargs):
    """Constructs a BiRealNet-18 model. """
    model = En3_BiRealNet(BasicBlock, [6,6,6], **kwargs)
    return model
