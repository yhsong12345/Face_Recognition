import torch.nn as nn



def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.ReLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Apply convolution and activation without batch normalization."""
        return self.act(self.conv(x))



class SELayer(nn.Module):
    def __init__(self, channel, act, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            act,
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)



class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, c1, c2, s=1, act=True):
        super().__init__()

        self.conv1 = Conv(c1, c2, k=3, s=s, p=1, act=False)
        self.conv2 = Conv(c2, c1, k=3, s=s, p=1, act=False)
        self.act = self.conv1.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()


        if s != 1 or c2 != c1:
            self.shortcut = Conv(c1, self.expansion * c2, k=1, s=s, p=1, act=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        o = self.conv1(x)
        o = self.conv2(o)
        o += self.shortcut(x)
        return self.act(o)



class SEBasicBlock(BasicBlock):
    def __init__(self, c1, c2, s=1, act=True, reduction=16):
        super().__init__(c1, c2, s, act)

        self.se = SELayer(c2, act, reduction)


    def forward(self, x):
        o = self.conv1(x)
        o = self.conv2(o)
        o = self.se(o)
        o += self.shortcut(x)
        return self.act(o)




class IBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, c1, c2, s=1, act=True):
        super().__init__()
        self.bn = nn.BatchNorm2d(c1)
        self.conv1 = Conv(c1, c2, k=3, s=1, p=1, act=act)
        self.conv2 = Conv(c2, c1, k=3, s=2, p=1, act=False)

        if s != 1 or c2 != c1:
            self.shortcut = Conv(c1, self.expansion * c2, k=1, s=s, p=1, act=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        o = self.conv2(self.conv1(self.bn(x)))
        o += self.shortcut(x)
        return o



class ISEBasicBlock(IBasicBlock):
    def __init__(self, c1, c2, s=1, act=True, reduction=16):
        super().__init__(c1, c2, s, act)
        self.se = SELayer(c2, act, reduction)

    def forward(self, x):
        o = self.conv2(self.conv1(self.bn(x)))
        o = self.se(o)
        o += self.shortcut(x)
        return o



class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, c1, c2, s=1, act=True):
        super().__init__()
        self.conv1 = Conv(c1, c2, k=1, act=False)
        self.conv2 = Conv(c2, c2, k=3, act=False)
        self.conv3 = Conv(c2, c2 * self.expansion, k=1, act=False)
        self.act = self.conv1.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        if s !=1 or c1 != c2:
            self.shortcut = Conv(c1, c2 * self.expansion, k=1, s=s, act=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        o = self.conv1(x)
        o = self.conv2(o)
        o = self.conv3(o)
        o += self.shortcut(x)
        return self.act(o)


class SEBottleneck(Bottleneck):
    def __init__(self, c1, c2, s=1, act=True, reduction=16):
        super().__init__(c1, c2, s, act)

        self.se = SELayer(c2, act, reduction)

    def forward(self, x):
        o = self.conv1(x)
        o = self.conv2(o)
        o = self.conv3(o)
        o = self.se(o)
        o += self.shortcut(x)
        return self.act(o)


class IBottleneck(nn.Module):
    expansion = 4
    def __init__(self, c1, c2, s=1, act=True):
        super().__init__()
        self.bn = nn.BatchNorm2d(c1)
        self.conv1 = Conv(c1, c2, k=1, act=act)
        self.conv2 = Conv(c2, c2, k=3, act=act)
        self.conv3 = Conv(c2, c2 * self.expansion, k=1, act=False)

        if s !=1 or c1 != c2:
            self.shortcut = Conv(c1, c2 * self.expansion, k=1, s=s, act=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        o = self.bn(x)
        o = self.conv1(o)
        o = self.conv2(o)
        o = self.conv3(o)
        o += self.shortcut(x)
        return o



class ISEBottleneck(IBottleneck):
    def __init__(self, c1, c2, s=1, act=True, reduction=16):
        super().__init__(c1, c2, s, act)
        self.se = SELayer(c2, act, reduction)


    def forward(self, x):
        o = self.bn(x)
        o = self.conv1(o)
        o = self.conv2(o)
        o = self.conv3(o)
        o = self.se(o)
        o += self.shortcut(x)
        return o