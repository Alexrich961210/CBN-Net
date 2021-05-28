#-*-coding:utf-8-*-
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import math
class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out

class BinActive(torch.autograd.Function):
    '''
    Binarize the input activations for ***** BNN and XNOR *****.
    '''
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        input = input.sign()
        return input
    @staticmethod
    def backward(ctx, grad_output, ):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input

class XnorLinear(nn.Linear):

    def __init__(self, in_channels, out_channels, bias=False):
        super(XnorLinear, self).__init__(in_channels, out_channels, bias)
        self.weight = nn.Parameter(torch.rand((out_channels, in_channels)) * 0.001,requires_grad=True)

    def forward(self, x):
        w = self.weight
        alpha = torch.mean(torch.mean(abs(w), dim=-2, keepdim=True),dim=1, keepdim=True)
        bw = BinActive().apply(w)
        bw = bw * alpha
        output = F.linear(x, bw, self.bias)

        return output

class XnorConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, bias=False):
        super(XnorConv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, bias)

        self.weight = nn.Parameter(torch.rand((out_channels, in_channels, kernel_size, kernel_size)) * 0.001,
                                   requires_grad=True)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):

        w = self.weight
        alpha = torch.mean(
            torch.mean(torch.mean(abs(w), dim=3, keepdim=True), dim=2, keepdim=True), dim=1,
            keepdim=True).detach()
        bw = BinActive().apply(w)
        bw = bw * alpha
        output = F.conv2d(x, bw, self.bias, self.stride, self.padding)

        return output

class BinaryActivation(nn.Module):
    def __init__(self):
        super(BinaryActivation, self).__init__()

    def forward(self, x):
        out_forward = torch.sign(x)
        #out_e1 = (x^2 + 2*x)
        #out_e2 = (-x^2 + 2*x)
        out_e_total = 0
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
        out = out_forward.detach() - out3.detach() + out3

        return out

class channel_w(nn.Module):
    def __init__(self,out_ch):
        super(channel_w, self).__init__()
        self.w1 =torch.nn.Parameter(torch.rand(1,out_ch,1,1)*0.1,requires_grad=True)

    def forward(self,x):
        out = self.w1 * x
        return out


class HardBinaryConv(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1):
        super(HardBinaryConv, self).__init__()
        self.stride = stride
        self.padding = padding
        self.number_of_weights = in_chn * out_chn * kernel_size * kernel_size
        self.shape = (out_chn, in_chn, kernel_size, kernel_size)
        #self.weight = nn.Parameter(torch.rand((self.number_of_weights,1)) * 0.001, requires_grad=True)
        self.weight = nn.Parameter(torch.rand((self.shape)) * 0.001, requires_grad=True)
        # self.binary = ReActsign(in_chn)
        self.binary = BinaryActivation()

    def forward(self, x):
        #real_weights = self.weights.view(self.shape)
        real_weights = self.weight
        a = x
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        #print(scaling_factor, flush=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        binary_a = self.binary(a)
        #print(binary_weights, flush=True)
        y = F.conv2d(binary_a, binary_weights, stride=self.stride, padding=self.padding)

        return y

class IRConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(IRConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input):
        w = self.weight
        a = input
        bw = w - w.view(w.size(0), -1).mean(-1).view(w.size(0), 1, 1, 1)
        bw = bw / bw.view(bw.size(0), -1).std(-1).view(bw.size(0), 1, 1, 1)
        sw = torch.pow(torch.tensor([2]*bw.size(0)).cuda().float(), (torch.log(bw.abs().view(bw.size(0), -1).mean(-1)) / math.log(2)).round().float()).view(bw.size(0), 1, 1, 1).detach()
        bw = BinActive().apply(bw)
        ba = BinActive().apply(a)
        bw = bw * sw
        output = F.conv2d(ba, bw, self.bias,
                          self.stride, self.padding,
                          self.dilation, self.groups)
        return output

# class HardBinaryConv(nn.Module):
#     def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1):
#         super(HardBinaryConv, self).__init__()
#         self.stride = stride
#         self.padding = padding
#         self.number_of_weights = in_chn * out_chn * kernel_size * kernel_size
#         self.shape = (out_chn, in_chn, kernel_size, kernel_size)
#         #self.weight = nn.Parameter(torch.rand((self.number_of_weights,1)) * 0.001, requires_grad=True)
#         self.weight = nn.Parameter(torch.rand((self.shape)) * 0.001, requires_grad=True)
#
#     def forward(self, x):
#         #real_weights = self.weights.view(self.shape)
#         real_weights = self.weight
#         scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
#         #print(scaling_factor, flush=True)
#         scaling_factor = scaling_factor.detach()
#         binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
#         cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
#         binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
#         #print(binary_weights, flush=True)
#         y = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding)
#
#         return y

class HardBinaryLinear(nn.Module):
    def __init__(self, in_chn, out_chn):
        super(HardBinaryLinear, self).__init__()
        self.number_of_weights = in_chn * out_chn
        self.shape = (out_chn, in_chn)
        self.weights = nn.Parameter(torch.rand((self.number_of_weights,1)) * 0.001, requires_grad=True)
        # self.shift = u

    def forward(self, x):
        real_weights = self.weights.view(self.shape)
        # w_mean = torch.mean(real_weights)
        # w_std = torch.std(real_weights)
        scaling_factor = torch.mean(abs(real_weights))
        #print(scaling_factor, flush=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        #print(binary_weights, flush=True)
        y = F.linear(x, binary_weights)

        return y

