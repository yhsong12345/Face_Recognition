import torch.nn as nn


from models.common import Conv




class LResNet(nn.Module):
    def __init__(self, block, num_blocks, act, num_classes=10):
        super().__init__()
        self.in_planes = 64

        self.conv = Conv(3, self.in_planes, k=7, s=2, act=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, act=act)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, act=act)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, act=act)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, act=act)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(512 * block.expansion, num_classes)


        #weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=1e-3)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, planes, num_blocks, stride, act):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, act))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)


    def forward(self, x):
        o = self.conv(x)
        o = self.layer1(o)
        o = self.layer2(o)
        o = self.layer3(o)
        o = self.layer4(o)
        o = self.avgpool(o)
        o = o.view(o.size(0), -1)
        o = self.linear(o)

        return o