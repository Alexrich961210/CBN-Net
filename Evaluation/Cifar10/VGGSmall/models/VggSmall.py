import torch
import torch.nn as nn
from util.binary_modules import channel_w,HardBinaryConv
import math
import torch.nn.init as init

class VGG_SMALL(nn.Module):

    def __init__(self, num_classes=10):
        super(VGG_SMALL, self).__init__()
        self.conv0 = nn.Conv2d(3, 128, kernel_size=3, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(128)
        self.conv1 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(128)
        # self.nonlinear = nn.ReLU(inplace=True)
        self.nonlinear = nn.Hardtanh(inplace=True)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(512)
        self.fc = nn.Linear(512*4*4, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, HardBinaryConv):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.nonlinear(x)

        x = self.conv1(x)
        x = self.pooling(x)
        x = self.bn1(x)
        x = self.nonlinear(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.nonlinear(x)

        x = self.conv3(x)
        x = self.pooling(x)
        x = self.bn3(x)
        x = self.nonlinear(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.nonlinear(x)

        x = self.conv5(x)
        x = self.pooling(x)
        x = self.bn5(x)
        x = self.nonlinear(x)
        # x = self.pooling(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class VGG_SMALL_1W1A(nn.Module):

    def __init__(self, num_classes=10):
        super(VGG_SMALL_1W1A, self).__init__()
        self.conv0 = nn.Conv2d(3, 128, kernel_size=3, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(128)
        self.conv1 = HardBinaryConv(128, 128, kernel_size=3, padding=1)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(128)
        # self.binary = BinaryActivation()
        # self.nonlinear = nn.ReLU(inplace=True)
        self.nonlinear = nn.Hardtanh(inplace=True)
        self.conv2 = HardBinaryConv(128, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = HardBinaryConv(256, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = HardBinaryConv(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = HardBinaryConv(512, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.fc = nn.Linear(512*4*4, num_classes)
    #     self._initialize_weights()
    #
    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #             m.weight.data.normal_(0, math.sqrt(2. / n))
    #             if m.bias is not None:
    #                 m.bias.data.zero_()
    #         elif isinstance(m, HardBinaryConv):
    #             n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #             m.weight.data.normal_(0, math.sqrt(2. / n))
    #             if m.bias is not None:
    #                 m.bias.data.zero_()
    #         elif isinstance(m, nn.BatchNorm2d):
    #             m.weight.data.fill_(1)
    #             m.bias.data.zero_()
    #         elif isinstance(m, nn.Linear):
    #             n = m.weight.size(1)
    #             m.weight.data.normal_(0, 0.01)
    #             m.bias.data.zero_()


    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)

        x = self.conv1(x)
        x = self.pooling(x)
        x = self.bn1(x)
        x = self.nonlinear(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.nonlinear(x)

        x = self.conv3(x)
        x = self.pooling(x)
        x = self.bn3(x)
        x = self.nonlinear(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.nonlinear(x)

        x = self.conv5(x)
        x = self.pooling(x)
        x = self.bn5(x)
        x = self.nonlinear(x)
        # x = self.pooling(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class CBN2_VggSmall_1w1a(nn.Module):

    def __init__(self, num_classes=10):
        super(CBN2_VggSmall_1w1a, self).__init__()
        self.conv0 = nn.Conv2d(3, 128, kernel_size=3, padding=1,bias=False)
        self.bn0 = nn.BatchNorm2d(128)

        self.conv1_1 = HardBinaryConv(128, 128, kernel_size=3, padding=1)
        self.conv1_2 = HardBinaryConv(128, 128, kernel_size=3, padding=1)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1_1 = nn.BatchNorm2d(128)
        self.bn1_2 = nn.BatchNorm2d(128)
        self.dropout1 = channel_w(128)
        self.dropout2 = channel_w(128)

        # self.nonlinear = nn.ReLU(inplace=True)
        self.nonlinear = nn.Hardtanh(inplace=True)
        self.conv2_1 = HardBinaryConv(128, 256, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(256)
        self.conv2_2 = HardBinaryConv(128, 256, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(256)
        self.dropout3 = channel_w(256)
        self.dropout4 = channel_w(256)

        self.conv3_1 = HardBinaryConv(256, 256, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = HardBinaryConv(256, 256, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.dropout5 = channel_w(256)
        self.dropout6 = channel_w(256)

        self.conv4_1 = HardBinaryConv(256, 512, kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.conv4_2 = HardBinaryConv(256, 512, kernel_size=3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.dropout7 = channel_w(512)
        self.dropout8 = channel_w(512)

        self.conv5_1 = HardBinaryConv(512, 512, kernel_size=3, padding=1)
        self.bn5_1 = nn.BatchNorm2d(512)
        self.conv5_2 = HardBinaryConv(512, 512, kernel_size=3, padding=1)
        self.bn5_2 = nn.BatchNorm2d(512)
        self.dropout9 = channel_w(512)
        self.dropout10 = channel_w(512)

        self.bn = nn.BatchNorm1d(512*4*4*2)
        self.fc = nn.Linear(512*4*4*2, num_classes)



    def forward(self, x):
        x = self.bn0(self.conv0(x))

        x1 = self.nonlinear(self.bn1_1(self.pooling(self.conv1_1(x))))

        x2 = self.nonlinear(self.bn1_2(self.pooling(self.conv1_2(x))))

        resx1_1 = self.dropout1(x1)
        resx2_1 = self.dropout2(x2)

        x1 = self.nonlinear(self.bn2_1(self.conv2_1(x1+resx2_1)))
        x2 = self.nonlinear(self.bn2_2(self.conv2_2(x2 + resx1_1)))

        resx1_2 = self.dropout3(x1)
        resx2_2 = self.dropout4(x2)

        x1 = self.nonlinear(self.bn3_1(self.pooling(self.conv3_1(x1+resx2_2))))
        x2 = self.nonlinear(self.bn3_2(self.pooling(self.conv3_2(x2+resx1_2))))

        resx1_3 = self.dropout5(x1)
        resx2_3 = self.dropout6(x2)

        x1 = self.nonlinear(self.bn4_1(self.conv4_1(x1 + resx2_3)))
        x2 = self.nonlinear(self.bn4_2(self.conv4_2(x2 + resx1_3)))

        resx1_4 = self.dropout7(x1)
        resx2_4 = self.dropout8(x2)


        x1 = self.nonlinear(self.bn5_1(self.pooling(self.conv5_1(x1 + resx2_4))))
        x2 = self.nonlinear(self.bn5_2(self.pooling(self.conv5_2(x2 + resx1_4))))

        resx1_6 = self.dropout9(x1)
        resx2_6 = self.dropout10(x2)
        #
        x1 = x1 + resx2_6
        x2 = x2 + resx1_6

        out = torch.cat((x1, x2), dim=1)
        out = out.view(out.size(0), -1)
        out = self.bn(out)
        out = self.fc(out)
        return out

def Vgg_small_1w1a(**kwargs):
    model = VGG_SMALL_1W1A(**kwargs)
    return model

def CBN2_vgg_small_1w1a(**kwargs):
    model = CBN2_VggSmall_1w1a(**kwargs)
    return model