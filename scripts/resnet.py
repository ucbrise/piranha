# https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
# A100: https://azure.microsoft.com/en-us/blog/azure-announces-general-availability-of-scaleup-scaleout-nvidia-a100-gpu-instances-claims-title-of-fastest-public-cloud-super/

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import random

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

def output_hw(input_hw, kernel_size, padding, stride):
    '''
    print('called output_hw: input_hw:', input_hw, 'kernel_size:', kernel_size, 'padding:', padding, 'stride:', stride)
    print('result', (
        int(((input_hw[0] + 2*padding - kernel_size) / stride) + 1),
        int(((input_hw[1] + 2*padding - kernel_size) / stride) + 1)
    ))
    '''
    return (
        int(((input_hw[0] + 2*padding - kernel_size) / stride) + 1),
        int(((input_hw[1] + 2*padding - kernel_size) / stride) + 1)
    )


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, input_hw, in_planes, planes, conv_layer, stride=1):
        super(BasicBlock, self).__init__()
        print('creating block: input_hw', input_hw, 'in_planes', in_planes, 'planes', planes, 'stride', stride, '\n')
        self.conv1 = conv_layer(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.LayerNorm([planes, *output_hw(input_hw, 3, 1, stride)], eps=(2**-8))
        #self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv_layer(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        #self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = nn.LayerNorm([planes, *output_hw(output_hw(input_hw, 3, 1, stride), 3, 1, 1)], eps=(2**-8))

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                conv_layer(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.LayerNorm([self.expansion * planes, *output_hw(input_hw, 1, 0, stride)], eps=(2**-8))
                #nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

    def dump_weights(self):
        print('dumping block weights:')
        print('---')
        print()

        print()
        print('conv1 - filter')
        print_as_columnwise(self.conv1.weight)

        print()
        print('conv2 - filter')
        print_as_columnwise(self.conv2.weight)

        print()
        print('shortcut conv - filter')
        print_as_columnwise(self.shortcut[0].weight)

        print()
        print('ln1 - gamma')
        print_as_columnwise(self.bn1.weight)

        print()
        print('ln1 - beta')
        print_as_columnwise(self.bn1.bias)

        print()
        print('ln2 - gamma')
        print_as_columnwise(self.bn2.weight)

        print()
        print('ln2 - beta')
        print_as_columnwise(self.bn2.bias)

        print()
        print('shortcut ln - gamma')
        print_as_columnwise(self.shortcut[1].weight)

        print()
        print('shortcut ln - beta')
        print_as_columnwise(self.shortcut[1].bias)


class ResNet(nn.Module):
    def __init__(self, conv_layer, linear_layer, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv_layer = conv_layer

        self.conv1 = conv_layer(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = linear_layer(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        print('make_layer:', 'block', block, 'planes', planes, 'num_blocks', num_blocks, 'stride', stride)
        strides = [stride] + [1] * (num_blocks - 1)
        print('\tstrides:', strides)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, self.conv_layer, stride))
            self.in_planes = planes * block.expansion
        #print('\tlayers:', layers)
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# NOTE: Only supporting default (kaiming_init) initializaition.
def resnet18(conv_layer, linear_layer, init_type, **kwargs):
    assert init_type == "kaiming_normal", "only supporting default init for Resnets"
    return ResNet(conv_layer, linear_layer, BasicBlock, [2, 2, 2, 2], **kwargs)


def test():
    net = resnet18(nn.Conv2d, nn.Linear, "kaiming_normal")
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())


def print_as_columnwise(arr):
    #print(arr.shape)
    if len(arr.shape) == 4:
        print('NHWC:', np.array2string(arr.cpu().detach().numpy().transpose(0, 3, 2, 1).flatten(), separator=',', max_line_width=np.inf))
    elif len(arr.shape) == 3:
        print('NHWC:', np.array2string(arr.cpu().detach().numpy().transpose(2, 1, 0).flatten(), separator=',', max_line_width=np.inf))
    elif len(arr.shape) == 2:
        print('NHWC:', np.array2string(arr.cpu().detach().numpy().transpose(1, 0).flatten(), separator=',', max_line_width=np.inf))


# Testing basic block computation
#    batch size = 2, im height = 3, im width = 3, Din = 2, Dout = 4, stride = 2
input_hw = (3, 3)

'''
blk = BasicBlock(input_hw=input_hw, in_planes=2, planes=4, conv_layer=nn.Conv2d, stride=2)
print(blk)
'''

im = torch.randn(2, 2, *input_hw)

class blk_model(nn.Module):

    def input_hook(self, grad):
        self.grad_wrt_input = grad

    def output_hook(self, grad):
        self.grad_wrt_output = grad

    def __init__(self):
        super(blk_model, self).__init__()

        self.first_layer = nn.Linear(3*3*2, 3*3*2, bias=True)
        self.blk_layer = BasicBlock(input_hw=input_hw, in_planes=2, planes=4, conv_layer=nn.Conv2d, stride=2)
        self.final_layer = nn.Linear(2*2*4, 10, bias=True)

    def forward(self, x):
        return self.final_layer(
            self.blk_layer(
                self.first_layer(x).view(2, 2, 3, 3)
            ).view(2, -1)
        )

    def output(self, x):
        blk_input = self.first_layer(x).view(2, 2, 3, 3)
        blk_input.register_hook(self.input_hook)

        out = self.blk_layer(blk_input)
        out.register_hook(self.output_hook)

        return blk_input, out, self.final_layer(out.view(2, -1))

blk_m = blk_model()

print(blk_m)

blk_m.blk_layer.dump_weights()

epochs = 1
optimizer = optim.SGD(blk_m.parameters(), lr=2**(-3))
optimizer.zero_grad()
criterion = nn.CrossEntropyLoss()

print('im shape', im.shape)
blk_in, blk_out, final_out = blk_m.output(im.reshape(2, -1))

print('blk in')
print_as_columnwise(blk_in)

print('blk out')
print_as_columnwise(blk_out)

print('final out shape', final_out.shape)
target = torch.empty(2, dtype=torch.long).random_(10)
print('target shape', target.shape)
loss = criterion(final_out, target)
loss.backward()

print('grad_wrt_layer_output')
print_as_columnwise(blk_m.grad_wrt_output)

print('grad_wrt_layer_input')
print_as_columnwise(blk_m.grad_wrt_input)

