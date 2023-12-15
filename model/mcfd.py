import torch
from torch import nn
import torch.nn.functional as F
import pytorch_wavelets as wavelets

class SPP_Max(nn.Module):
    def __init__(self, dim=32):
        super(SPP_Max, self).__init__()
        self.levels = [2, 3, 4]
        self.conv1 = nn.Conv2d(dim * 3, dim, kernel_size=1, padding=0)
        self.conv2 = nn.Sequential(
            nn.Conv2d(dim * 3, dim, kernel_size=1, padding=0),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.Conv2d(dim, dim, kernel_size=1, padding=0),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
        )
        self.c1 = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.c2 = nn.Parameter(torch.tensor(0.5), requires_grad=True)

    def forward(self, x):
        pool_outputs = []
        for level in self.levels:
            pool_size = (int(x.size(2) // level), int(x.size(3) // level))
            pool = nn.AdaptiveMaxPool2d(pool_size)
            out = pool(x)
            out = F.interpolate(out, size=(x.size()[2] // 2, x.size()[3] // 2), mode='bilinear', align_corners=True)
            pool_outputs.append(out)
        spp = torch.cat(pool_outputs, dim=1)

        spp = self.conv1(spp) * self.c1 + self.conv2(spp) * self.c2
        return spp


class SPP_Avg(nn.Module):
    def __init__(self, dim=32):
        super(SPP_Avg, self).__init__()
        self.levels = [2, 3, 4]
        self.conv1 = nn.Conv2d(dim * 3, dim, kernel_size=1, padding=0)
        self.conv2 = nn.Sequential(
            nn.Conv2d(dim * 3, dim, kernel_size=1, padding=0),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.Conv2d(dim, dim, kernel_size=1, padding=0),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        )
        self.c1 = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.c2 = nn.Parameter(torch.tensor(0.5), requires_grad=True)

    def forward(self, x):
        pool_outputs = []
        for level in self.levels:
            pool_size = (int(x.size(2) // level), int(x.size(3) // level))
            pool = nn.AdaptiveAvgPool2d(pool_size)
            out = pool(x)
            out = F.interpolate(out, size=(x.size()[2] // 2, x.size()[3] // 2), mode='bilinear', align_corners=True)
            pool_outputs.append(out)
        spp = torch.cat(pool_outputs, dim=1)

        spp = self.conv1(spp) * self.c1 + self.conv2(spp) * self.c2
        return spp


class DMCSLayer(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(DMCSLayer, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=1, padding=0, stride=1, bias=False),
            nn.ReLU(inplace=True),  
            nn.Conv2d(outchannel, outchannel, kernel_size=3, padding=1, stride=1, bias=False),
            nn.ReLU(inplace=True),  
            nn.Conv2d(outchannel, outchannel, kernel_size=1, padding=0, stride=1, bias=False),
            nn.ReLU(inplace=True),  
            nn.Conv2d(outchannel, outchannel, kernel_size=3, padding=1, stride=1, bias=False),
            nn.ReLU(inplace=True),  
            nn.Conv2d(outchannel, outchannel, kernel_size=1, padding=0, stride=1, bias=False),
            nn.ReLU(inplace=True),  
            nn.Conv2d(outchannel, outchannel, kernel_size=3, padding=1, stride=2, bias=False),
            nn.ReLU(inplace=True),  
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=1, padding=0, stride=1, bias=False),
            nn.ReLU(inplace=True),  
            nn.Conv2d(outchannel, outchannel, kernel_size=2, padding=1, stride=1, bias=False),
            nn.ReLU(inplace=True),  
            nn.Conv2d(outchannel, outchannel, kernel_size=3, padding=0, stride=2, bias=False),
            nn.ReLU(inplace=True),  
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, padding=1, stride=2, bias=False),
        )
        self.maxpool = SPP_Max(dim=inchannel)
        self.avgpool = SPP_Avg(dim=inchannel)
        self.avgattention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(outchannel, outchannel // 16, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel // 16, outchannel, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.maxattention = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(outchannel, outchannel // 16, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel // 16, outchannel, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

        self.c1 = nn.Parameter(torch.tensor(1 / 3.), requires_grad=True)
        self.c2 = nn.Parameter(torch.tensor(1 / 3.), requires_grad=True)
        self.c3 = nn.Parameter(torch.tensor(1 / 3.), requires_grad=True)

        self.o1 = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.o2 = nn.Parameter(torch.tensor(0.5), requires_grad=True)

    def forward(self, x):
        out = self.conv1(x) * self.c1 + self.conv2(x) * self.c2 + self.conv3(x) * self.c3
        res_avg = self.avgpool(x)
        res_max = self.maxpool(x)
        out1 = out + res_avg
        out2 = out + res_max
        out1 = out1 * self.avgattention(out1)
        out2 = out2 * self.maxattention(out2)

        return self.o1 * out1 + self.o2 * out2


class MCFD(nn.Module):
    def __init__(self, sensing_rate):
        super(MCFD, self).__init__()
        self.sensing_rate = sensing_rate
        self.base = 32
        self.blocksize = 32
        n_feats_dmcs = 64

        if sensing_rate == 0.5:
            # 0.5000
            self.dmcs = nn.Sequential(
                nn.Conv2d(1, n_feats_dmcs, kernel_size=3, padding=1, stride=1, bias=False),
                DMCSLayer(n_feats_dmcs, n_feats_dmcs),
                nn.Conv2d(n_feats_dmcs, 2, kernel_size=1, padding=0, stride=1, bias=False),
            )
            self.m1 = 2
            self.m2 = 2
            self.amplify = 0

        elif sensing_rate == 0.25:
            # 0.2500
            self.dmcs = nn.Sequential(
                nn.Conv2d(1, n_feats_dmcs, kernel_size=3, padding=1, stride=1, bias=False),
                DMCSLayer(n_feats_dmcs, n_feats_dmcs),
                DMCSLayer(n_feats_dmcs, n_feats_dmcs),
                nn.Conv2d(n_feats_dmcs, 4, kernel_size=1, padding=0, stride=1, bias=False)
            )
            self.m1 = 4
            self.m2 = 4
            self.amplify = 4

        elif sensing_rate == 0.125:
            # 0.1250
            self.dmcs = nn.Sequential(
                nn.Conv2d(1, n_feats_dmcs, kernel_size=3, padding=1, stride=1, bias=False),
                DMCSLayer(n_feats_dmcs, n_feats_dmcs),
                DMCSLayer(n_feats_dmcs, n_feats_dmcs),
                nn.Conv2d(n_feats_dmcs, 2, kernel_size=1, padding=0, stride=1, bias=False)
            )
            self.m1 = 2
            self.m2 = 4
            self.amplify = 4

        elif sensing_rate == 0.0625:
            # 0.0625
            self.dmcs = nn.Sequential(
                nn.Conv2d(1, n_feats_dmcs, kernel_size=3, padding=1, stride=1, bias=False),
                DMCSLayer(n_feats_dmcs, n_feats_dmcs),
                DMCSLayer(n_feats_dmcs, n_feats_dmcs),
                DMCSLayer(n_feats_dmcs, n_feats_dmcs),
                nn.Conv2d(n_feats_dmcs, 4, kernel_size=1, padding=0, stride=1, bias=False),
            )
            self.m1 = 4
            self.m2 = 8
            self.amplify = 8

        elif sensing_rate == 0.03125:
            # 0.03125

            self.dmcs = nn.Sequential(
                nn.Conv2d(1, n_feats_dmcs, kernel_size=3, padding=1, stride=1, bias=False),
                DMCSLayer(n_feats_dmcs, n_feats_dmcs),
                DMCSLayer(n_feats_dmcs, n_feats_dmcs),
                DMCSLayer(n_feats_dmcs, n_feats_dmcs),
                nn.Conv2d(n_feats_dmcs, 2, kernel_size=1, padding=0, stride=1, bias=False),
            )
            self.m1 = 2
            self.m2 = 8
            self.amplify = 8

        elif sensing_rate == 0.015625:
            # 0.015625

            self.dmcs = nn.Sequential(
                nn.Conv2d(1, n_feats_dmcs, kernel_size=3, padding=1, stride=1, bias=False),
                DMCSLayer(n_feats_dmcs, n_feats_dmcs),
                DMCSLayer(n_feats_dmcs, n_feats_dmcs),
                DMCSLayer(n_feats_dmcs, n_feats_dmcs),
                DMCSLayer(n_feats_dmcs, n_feats_dmcs),
                nn.Conv2d(n_feats_dmcs, 4, kernel_size=1, padding=0, stride=1, bias=False),
            )
            self.m1 = 4
            self.m2 = 16
            self.amplify = 12

        self.initial = nn.Sequential(
            nn.Conv2d(self.m1, self.m2 ** 2, kernel_size=1, padding=0, stride=1, bias=False),
        )
        self.base = int(self.base + self.amplify)

        self.conv1 = nn.Conv2d(1, self.base, kernel_size=3, padding=1, stride=1, bias=True)
        self.conv2 = nn.Conv2d(self.base, 1, kernel_size=3, padding=1, stride=1, bias=True)

        self.num_layers = 8

        modules_tmp = [
            FDRM(dim=self.base, m1=self.m1, m2=self.m2, dmcs=self.dmcs, init=self.initial, layers=_) for _ in
            range(self.num_layers)
        ]

        self.fdrms = nn.ModuleList(modules_tmp)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.dmcs(x)
        x = self.initial(y)
        initial = nn.PixelShuffle(self.m2)(x)
        out = self.relu(self.conv1(initial))

        for i in range(self.num_layers):
            out = self.fdrms[i](out, y)

        out = self.conv2(out)
        return out + initial, initial


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out) * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class cbam_block(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7):
        super(cbam_block, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = self.channelattention(x)
        x = x * self.spatialattention(x)
        return x


class MulAttBlock(nn.Module):
    def __init__(self, dim):
        super(MulAttBlock, self).__init__()
        self.dim = dim
        self.conv_resize1 = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, kernel_size=3, stride=2, padding=1, bias=True, groups=4),
            nn.Conv2d(self.dim, self.dim, kernel_size=3, stride=2, padding=1, bias=True, groups=4),
        )
        self.conv_resize2 = nn.Sequential(
            nn.Conv2d(dim, dim * 4, kernel_size=3, stride=1, padding=1, bias=False, groups=4),
            nn.PixelShuffle(2),
            nn.Conv2d(dim, dim * 4, kernel_size=3, stride=1, padding=1, bias=False, groups=4),
            nn.PixelShuffle(2)
        )
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=7, padding=3, stride=1, bias=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, bias=True)
        self.conv3 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, bias=True)
        self.scale = dim ** -0.5
        self.conv4 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, bias=True)

    def forward(self, x_in):
        x = self.conv_resize1(x_in)

        B, C, H, W = x.shape
        x = self.conv1(x)
        x = x.reshape(B, C, H // 8, 8, W // 8, 8).permute(0, 2, 4, 1, 3, 5).reshape(-1, C, 8, 8)
        x1 = self.conv2(x).reshape(-1, C, 8 * 8)
        x2 = self.conv3(x).reshape(-1, C, 8 * 8).transpose(1, 2)
        att = (x2 @ x1) * self.scale
        att = att.softmax(dim=1)
        x = (x.reshape(-1, C, 8 * 8) @ att).reshape(-1, C, 8, 8)
        x = self.conv4(x)
        x = x.reshape(B, H // 8, W // 8, C, 8, 8).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)

        x = self.conv_resize2(x)
        out = x_in + x
        return out


class ResBlock(nn.Module):
    def __init__(self, dim):
        super(ResBlock, self).__init__()

        self.res = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
        )

    def forward(self, x):
        return x + self.res(x)


class FDRM(nn.Module):
    def __init__(self, dim=32, m1=2, m2=2, dmcs=None, init=None, layers=None):
        super(FDRM, self).__init__()
        self.dmcs = dmcs
        self.init = init
        self.dim = dim
        self.m1 = m1
        self.m2 = m2
        self.x_conv1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=2, padding=0, stride=2, bias=True),
            nn.ReLU(),
        )
        self.x_conv2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=2, padding=0, stride=2, bias=True),
            nn.ReLU(),
        )
        self.x_conv3 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=2, bias=True),
            nn.ReLU(),
        )

        self.mix0 = nn.Sequential(
            nn.Conv2d(m1, (m2 ** 2), kernel_size=1, padding=0, stride=1, bias=True),
            nn.ReLU(),
            nn.PixelShuffle(m2),
            nn.Conv2d(1, dim, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(),
        )
        self.mix1 = nn.Sequential(
            nn.Conv2d(m1, (m2 ** 2), kernel_size=1, padding=0, stride=1, bias=True),
            nn.ReLU(),
            nn.PixelShuffle(m2),
            # 1/2
            nn.Conv2d(1, dim, kernel_size=3, padding=1, stride=2, bias=True),
            nn.ReLU(),
        )

        self.mix2 = nn.Sequential(
            nn.Conv2d(m1, (m2 ** 2), kernel_size=1, padding=0, stride=1, bias=True),
            nn.ReLU(),
            nn.PixelShuffle(m2),
            # 1/4大小
            nn.Conv2d(1, dim, kernel_size=3, padding=1, stride=4, bias=True),
            nn.ReLU(),
        )
        self.mix3 = nn.Sequential(
            nn.Conv2d(m1, (m2 ** 2), kernel_size=1, padding=0, stride=1, bias=True),
            nn.ReLU(),
            nn.PixelShuffle(m2),
            # 1/8大小
            nn.Conv2d(1, dim, kernel_size=3, padding=1, stride=8, bias=True),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1, bias=True, groups=2),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 4, kernel_size=3, padding=1, bias=True, groups=4),
            nn.ReLU(),
            nn.PixelShuffle(2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1, bias=True, groups=2),
            nn.ReLU(),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 4, kernel_size=3, padding=1, bias=True, groups=4),
            nn.ReLU(),
            nn.PixelShuffle(2)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 4, kernel_size=3, padding=1, bias=True, groups=4),
            nn.ReLU(),
            nn.PixelShuffle(2)
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
        )
        self.conv10 = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1, bias=True),
            nn.ReLU()
        )

        self.high_conv = nn.Sequential(
            nn.Conv2d(3, dim * 4, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            nn.PixelShuffle(2)
        )
        self.low_conv = nn.Sequential(
            nn.Conv2d(1, dim * 4, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            nn.PixelShuffle(2)
        )
        self.high_conv_ = nn.Sequential(
            nn.Conv2d(dim, 1, kernel_size=3, padding=1, bias=True),
            nn.ReLU()
        )
        self.low_conv_ = nn.Sequential(
            nn.Conv2d(dim, 1, kernel_size=3, padding=1, bias=True),
            nn.ReLU()
        )

        self.cat_conv = nn.Sequential(
            nn.Conv2d(2, dim, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(dim, 1, kernel_size=3, padding=1, bias=True),
            nn.ReLU()
        )

        self.conv1 = nn.Conv2d(1, self.dim, kernel_size=3, padding=1, stride=1, bias=True)
        self.conv2 = nn.Conv2d(self.dim, 1, kernel_size=3, padding=1, stride=1, bias=True)

        self.o11 = nn.Parameter(torch.tensor(0.3), requires_grad=True)
        self.o21 = nn.Parameter(torch.tensor(0.7), requires_grad=True)
        self.o12 = nn.Parameter(torch.tensor(0.3), requires_grad=True)
        self.o22 = nn.Parameter(torch.tensor(0.7), requires_grad=True)
        self.o13 = nn.Parameter(torch.tensor(0.3), requires_grad=True)
        self.o23 = nn.Parameter(torch.tensor(0.7), requires_grad=True)

        self.ca1 = cbam_block(dim)
        self.ca2 = cbam_block(dim)
        self.res = ResBlock(dim)
        self.att = MulAttBlock(dim) if (layers + 1) % 4 == 0 else None
        self.dwt = wavelets.DWTForward(J=1, wave='haar', mode='zero')

    def forward(self, x, y=None):
        y0 = self.mix0(y)
        y1 = self.mix1(y)  # 1/2
        y2 = self.mix2(y)  # 1/4
        y3 = self.mix3(y)  # 1/8

        f1 = self.x_conv1(x)
        f2 = self.x_conv2(f1)
        f3 = self.x_conv3(f2)

        f7 = self.conv3(torch.cat([f3, y3], dim=1))  # 1/8
        f8 = self.conv4(torch.cat([f7, f3], dim=1))  # 1/4

        f9 = self.conv5(torch.cat([f8, y2], dim=1))  # 1/4
        f10 = self.conv6(torch.cat([f9, f2], dim=1))  # 1/2

        f11 = self.conv7(torch.cat([f10, y1], dim=1))  # 1/2
        f12 = self.conv8(torch.cat([f11, f1], dim=1))  # 1/1

        tmp12 = self.conv2(f12)
        coeffs = self.dwt(tmp12) 
        cA, (cH, cV, cD) = coeffs[0], (coeffs[1][0][:, :, _, :, :] for _ in range(3))
        hf = self.high_conv(torch.cat((cH, cV, cD), dim=1))
        lf = self.low_conv(cA)
        hf = self.high_conv_(self.ca1(hf))
        lf = self.low_conv_(self.ca2(lf))
        tmp12 = tmp12 + self.cat_conv(torch.cat((hf, lf), dim=1))

        delta = self.dmcs(tmp12) - y
        init_delta = self.conv1(nn.PixelShuffle(self.m2)(self.init(delta)))
        f12 = f12 + self.res(init_delta)

        f12 = self.att(f12) if self.att is not None else f12
        f13 = self.conv9(torch.cat([f12, y0], dim=1))  # 1/1
        f14 = self.conv10(torch.cat([f13, x], dim=1))  # 1/1

        return f14
