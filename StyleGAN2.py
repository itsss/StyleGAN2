import os
import torch.nn.functional as torchF
import torch.nn as nn
import numpy as np
import torch
from torch.nn import ModuleList

from utils.libs import _setup_kernel, _approximate_size
from opts import TrainOptions, INFO

import copy
from tqdm import tqdm

from torchvision.utils import save_image
from matplotlib import pyplot as plt
from utils.utils import plotLossCurve


class PixelNormalization(nn.Module):
    def __init__(self, ep=1e-8):
        super(PixelNormalization, self).__init__()
        self.epslion = ep

    def forward(self, x):
        tmp = torch.mul(x,x)
        tmp_ = torch.rsqrt(torch.mean(tmp, dim=1, keepdim=True) + self.epslion)
        return x*tmp_

class BiasAdd(nn.Module):
    def __init__(self, channels, opts, act='linear', alpha=None, gain=None, mul=1):
        super(BiasAdd, self).__init__()
        self.opts = opts
        self.bias = torch.nn.Parameter((torch.zeros(channels, 1, 1) * mul))
        self.act = act
        self.alpha = alpha if alpha is not None else 0.2
        self.gain = gain if gain is not None else 1.0

    def forward(self, x):
        x += self.bias
        if(self.act == 'lrelu'): # Activation Function: Evaluate
            x = torchF.leaky_relu(x, self.alpha, inplace=True)
            x = x * np.sqrt(2)

        if(self.gain != 1):
            x = x * self.gain

        return x

class FullyConnectedLayer(nn.Module):
    def __init__(self, inc, outc, gain=1, use_wscale=True, lrmul=1.0, bias=True, act='lrelu', mode='normal'):
        super(FullyConnectedLayer, self).__init__()
        he = gain / np.sqrt(inc*outc)
        if(use_wscale):
            init_s = 1.0 / lrmul
            self.w_lrmul = he * lrmul
        else:
            init_s = he / lrmul
            self.w_lrmul = lrmul

        self.weight = torch.nn.Parameter(torch.empty(outc, inc).normal_(0, init_s))
        if(bias):
            self.bias = torch.nn.Parameter(torch.zeros(outc))
            self.b_lrmul = lrmul
        else:
            self.bias = None

        self.act = act
        self.mode = mode

    def forward(self, x):
        if(self.bias is not None and self.mode != 'modulate'):
            outval = torchF.linear(x, self.weight * self.w_lrmul, self.bias * self.b_lrmul)
        elif(self.bias is not None and self.mode == 'modulate'):
            outval = torchF.linear(x, self.weight * self.w_lrmul, self.bias * self.b_lrmul)
        else:
            outval = torchF.linear(x, self.weight * self.w_lrmul)

        if(self.act == 'lrelu'):
            outval = torchF.leaky_relu(outval, 0.2, inplace=True)
            outval = outval * np.sqrt(2)
            return outval
        elif(self.act == 'linear'):
            return outval
        return outval

class Conv2d(nn.Module):
    def __init__(self, inc, outc, ksize, gain=1, use_wscale=True, lrmul=1, bias=True, act='linear'):
        super().__init__()
        assert ksize >= 1 and ksize % 2 == 1
        he = gain / np.sqrt(inc*outc*ksize*ksize) # initial value
        self.kernel_size = ksize
        self.act = act

        if(use_wscale):
            init_s = 1.0 / lrmul
            self.w_lrmul = he * lrmul
        else:
            init_s = he / lrmul
            self.w_lrmul = lrmul

        self.weight = torch.nn.Parameter(torch.empty(outc, inc, ksize, ksize).normal_(0, init_s))
        if(bias):
            self.bias = torch.nn.Parameter(torch.zeros(outc))
            self.b_lrmul = lrmul
        else:
            self.bias = None

    def forward(self, x):
        if(self.bias is not None):
            outval = torchF.conv2d(x, self.weight * self.w_lrmul, self.bias * self.b_lrmul, padding=self.kernel_size // 2)
        else:
            outval = torchF.conv2d(x, self.weight * self.w_lrmul, padding=self.kernel_size // 2)

        if self.act == 'lrelu':
            outval = torchF.leaky_relu(outval, 0.2, inplace=True)
            outval = outval * np.sqrt(2)
            return outval
        elif self.act == 'linear':
            return outval

class FromRGB(nn.Module):
    def __init__(self, inc, outc, use_wscale=True, lrmul=1):
        super().__init__()
        self.conv = Conv2d(inc=inc, outc=outc, ksize=1, use_wscale=use_wscale, lrmul=lrmul)

    def forward(self, x):
        x, y = x
        y1 = self.conv(y)
        outval = torchF.leaky_relu(y1, 0.2, inplace=True)
        outval = outval * np.sqrt(2)
        return outval if x is None else outval+x

# Modulated convolution layer
class ModulatedConv2d(nn.Module):
    def __init__(self, inc, outc, ksize, opts, k=[1, 3, 3, 1], dlatent_size=512, up=False, down=False, demodulate=True, gain=1, use_wscale=True, lrmul=1, fused_modconv=True):
        super().__init__()
        assert ksize >= 1 and ksize % 2 == 1

        self.demodulate = demodulate
        self.fused_modconv = fused_modconv
        self.up, self.down = up, down
        self.fmaps = outc
        self.opts = opts

        self.conv = Conv2d(inc=inc, outc=outc, ksize=1, use_wscale=use_wscale, lrmul=lrmul)
        he_s = gain / np.sqrt((inc * outc * ksize * ksize))
        self.kernel_size = ksize

        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_lrmul = he_s * lrmul
        else:
            init_std = he_s / lrmul
            self.w_lrmul = lrmul

        self.w = torch.nn.Parameter(torch.empty(outc, inc, ksize, ksize).normal_(0, init_std))
        self.convH, self.convW = self.w.shape[2:]

        self.dense = FullyConnectedLayer(dlatent_size, inc, gain, lrmul=lrmul, use_wscale=use_wscale, mode='modulate', act='linear')

        if self.up:
            factor = 2
            self.k = _setup_kernel(k) * (gain * (factor ** 2))  # 4 x 4
            self.k = torch.FloatTensor(self.k).unsqueeze(0).unsqueeze(0)
            self.k = torch.flip(self.k, [2, 3])
            self.k = torch.nn.Parameter(self.k, requires_grad=False)

            self.p = self.k.shape[0] - factor - (ksize - 1)

            self.padx0, self.pady0 = (self.p + 1) // 2 + factor - 1, (self.p + 1) // 2 + factor - 1
            self.padx1, self.pady1 = self.p // 2 + 1, self.p // 2 + 1

            self.kernelH, self.kernelW = self.k.shape[2:]

    def forward(self, x):
        x, y = x
        if len(y.shape) > 2:
            y = y.squeeze(1)

        s = self.dense(y)

        self.ww = (self.w * self.w_lrmul).unsqueeze(0)
        self.ww = self.ww.repeat(s.shape[0], 1, 1, 1, 1)
        self.ww = self.ww.permute(0, 3, 4, 1, 2)
        self.ww = self.ww * s.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        self.ww = self.ww.permute(0, 1, 2, 4, 3)

        if self.demodulate:
            d = torch.mul(self.ww, self.ww)
            d = torch.rsqrt(torch.sum(d, dim=[1, 2, 3]) + 1e-8)
            self.ww = self.ww * (d.unsqueeze(1).unsqueeze(1).unsqueeze(1))

        if self.fused_modconv:
            x = x.view(1, -1, x.shape[2], x.shape[3])
            self.w_new = torch.reshape(self.ww.permute(0, 4, 3, 1, 2),
                                       (-1, x.shape[1], self.ww.shape[1], self.ww.shape[2]))
        else:
            x = x * (s.unsqueeze(-1).unsqueeze(-1))
            self.w_new = self.w * self.w_lrmul

        if self.up:
            out_c, in_c, conv_h, conv_w = self.w_new.shape[0], self.w_new.shape[1], self.w_new.shape[2], self.w_new.shape[3]

            num_groups = x.shape[1] // in_c if (x.shape[1] // in_c) >= 1 else 1
            self.w_new = self.w_new.reshape(-1, num_groups, in_c, conv_h, conv_w)
            self.w_new = self.w_new.flip([3, 4])
            self.w_new = self.w_new.permute(2, 1, 0, 3, 4)
            self.w_new = self.w_new.reshape(in_c, out_c, conv_h, conv_w)

            x = torchF.conv_transpose2d(x, self.w_new, stride=2)

            y = x.clone()
            y = y.reshape([-1, x.shape[2], x.shape[3], 1])  # N C H W ---> N*C H W 1

            in_c, in_h, in_w = x.shape[1:]
            y = y.reshape(-1, in_h, in_w, 1)

            y = torchF.pad(y, (0, 0, max(self.pady0, 0), max(self.pady1, 0), max(self.padx0, 0), max(self.padx1, 0), 0, 0))
            y = y[:, max(-self.pady0, 0): y.shape[1] - max(-self.pady1, 0), max(-self.padx0, 0): y.shape[2] - max(-self.padx1, 0), :]

            y = y.permute(0, 3, 1, 2)  # N*C H W 1 --> N*C 1 H W
            y = y.reshape(-1, 1, in_h + self.pady0 + self.pady1, in_w + self.padx0 + self.padx1)
            y = torchF.conv2d(y, self.k)
            y = y.view(-1, 1, in_h + self.pady0 + self.pady1 - self.kernelH + 1,
                       in_w + self.padx0 + self.padx1 - self.kernelW + 1)

            if in_h != y.shape[1] or in_h % 2 != 0:
                in_h = in_w = _approximate_size(in_h)
                y = torchF.interpolate(y, size=(in_h, in_w), mode='bilinear')
            y = y.permute(0, 2, 3, 1)
            x = y.reshape(-1, in_c, in_h, in_w)

        elif self.down:
            pass
        else:
            x = torchF.conv2d(x, self.w_new, padding=self.w_new.shape[2] // 2)

        if self.fused_modconv:
            x = x.reshape(-1, self.fmaps, x.shape[2], x.shape[3])

        elif self.demodulate:
            x = x * d.unsqueeze(-1).unsqueeze(-1)

        return x

class ToRGB(nn.Module):
    def __init__(self, inc, outc, res, opts, use_wscale=True, lrmul=1, gain=1, fused_modconv=True):
        super().__init__()
        assert res >= 2

        self.modulated_conv2d = ModulatedConv2d(inc=inc, outc=outc, ksize=1, up=False, use_wscale=use_wscale, lrmul=lrmul, gain=gain, demodulate=False, fused_modconv=fused_modconv, opts=opts)
        self.biasAdd = BiasAdd(opts=opts, act='linear', channels=outc)

        self.res = res
        self.opts = opts

    def forward(self, x):
        x, y, d_lat = x
        d_lat = d_lat[:, self.res * 2 - 3]

        x = self.modulated_conv2d([x, d_lat])
        t = self.biasAdd(x)

        return t if y is None else y + t


# ===================================================================================================================
# 2020. 04. 30.  Complete                                                                                           #
# ===================================================================================================================

class GLayer(nn.Module):
    def __init__(self, inc, outc, layer_idx, opts, k=[1, 3, 3, 1], randomize_noise=True, up=False, use_wscale=True, lrmul=1, fused_modconv=True, act='lrelu'):
        super().__init__()

        self.randomize_noise = randomize_noise
        self.opts = opts
        self.up = up
        self.layer_idx = layer_idx

        self.modulated_conv2d = ModulatedConv2d(inc=inc, outc=outc, ksize=3, k=k, use_wscale=use_wscale, lrmul=lrmul, demodulate=True, fused_modconv=fused_modconv, up=up, opts=opts)

        self.noise_strength = torch.nn.Parameter(torch.zeros(1))
        self.biasAdd = BiasAdd(act=act, channels=outc, opts=opts)

    def forward(self, x):
        x, d_lat = x
        if len(d_lat.shape) > 2:
            d_lat = d_lat[:, self.layer_idx]

        x = self.modulated_conv2d([x, d_lat])

        noise_vec = 0
        if self.randomize_noise:
            noise_vec = torch.randn(x.shape[0], 1, x.shape[2], x.shape[3]).to(self.opts.device)

        x = x + noise_vec * self.noise_strength
        x = self.biasAdd(x)

        return x


class Upsample2d(nn.Module):
    def __init__(self, opts, k=[1, 3, 3, 1], factor=2, down=1, gain=1):
        super().__init__()
        assert isinstance(factor, int) and factor >= 1, "factor should be more than 1 (default:2)"

        self.gain = gain
        self.factor = factor
        self.opts = opts

        self.kval = _setup_kernel(k) * (self.gain * (factor ** 2))  # 4 x 4
        self.kval = torch.FloatTensor(self.kval).unsqueeze(0).unsqueeze(0)
        self.kval = torch.flip(self.kval, [2, 3])
        self.kval = nn.Parameter(self.kval, requires_grad=False)

        self.p = self.kval.shape[0] - self.factor

        self.padx0, self.pady0 = (self.p + 1) // 2 + factor - 1, (self.p + 1) // 2 + factor - 1
        self.padx1, self.pady1 = self.p // 2, self.p // 2

        self.kernel_h, self.kernel_w = self.kval.shape[2:]
        self.down = down

    def forward(self, x):
        out = x.clone()
        out = out.reshape([-1, x.shape[2], x.shape[3], 1])  # N C H W ---> N*C H W 1

        in_c, in_h, in_w = x.shape[1:]

        out = torch.reshape(out, (-1, in_h, 1, in_w, 1, 1))
        out = torchF.pad(out, (0, 0, self.factor - 1, 0, 0, 0, self.factor - 1, 0, 0, 0, 0, 0))
        out = torch.reshape(out, (-1, 1, in_h * self.factor, in_w * self.factor))

        # 패딩 추가
        out = torchF.pad(out, (0, 0, max(self.pady0, 0), max(self.pady1, 0), max(self.padx0, 0), max(self.padx1, 0), 0, 0))
        out = out[:,
            max(-self.pady0, 0): out.shape[1] - max(-self.pady1, 0),
            max(-self.padx0, 0): out.shape[2] - max(-self.padx1, 0),
            :]

        # 필터 추가
        out = out.permute(0, 3, 1, 2)
        out = out.reshape(-1, 1, in_h * self.factor + self.pady0 + self.pady1, in_w * self.factor + self.padx0 + self.padx1)
        out = torchF.conv2d(out, self.kval)
        out = out.view(-1, 1, in_h * self.factor + self.pady0 + self.pady1 - self.kernel_h + 1, in_w * self.factor + self.padx0 + self.padx1 - self.kernel_w + 1)

        # downsample
        if in_h * self.factor != out.shape[1]:
            out = torchF.interpolate(out, size=(in_h * self.factor, in_w * self.factor), mode='bilinear')
        out = out.permute(0, 2, 3, 1)
        out = out.reshape(-1, in_c, in_h * self.factor, in_w * self.factor)

        return out


class ConvDownsample2d(nn.Module):
    def __init__(self, ksize, inc, outc, k=[1, 3, 3, 1], factor=2, gain=1, use_wscale=True, lrmul=1, bias=False, act='linear'):
        super().__init__()
        assert isinstance(factor, int) and factor >= 1, "factor must be larger than 1! (default: 2)"
        assert ksize >= 1 and ksize % 2 == 1

        he_std = gain / np.sqrt((inc * outc * ksize * ksize))  # Initial Standard
        self.kernel_size = ksize
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_lrmul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_lrmul = lrmul

        self.weight = torch.nn.Parameter(torch.empty(outc, inc, ksize, ksize).normal_(0, init_std))
        self.conv_h, self.conv_w = self.weight.shape[2:]

        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(outc))
            self.b_lrmul = lrmul
        else:
            self.bias = None

        self.gain = gain
        self.factor = factor
        self.act = act

        self.k = _setup_kernel(k) * self.gain
        self.k = torch.FloatTensor(self.k).unsqueeze(0).unsqueeze(0)
        self.k = torch.flip(self.k, [2, 3])
        self.k = nn.Parameter(self.k, requires_grad=False)

        self.p = (self.k.shape[-1] - self.factor) + (self.conv_w - 1)

        self.padx0, self.pady0 = (self.p + 1) // 2, (self.p + 1) // 2
        self.padx1, self.pady1 = self.p // 2, self.p // 2

        self.kernel_h, self.kernel_w = self.k.shape[2:]

    def forward(self, x):

        y = x.clone()
        y = y.reshape([-1, x.shape[2], x.shape[3], 1])  # N C H W ---> N*C H W 1

        in_c, in_h, in_w = x.shape[1:]
        # Upsample
        y = torch.reshape(y, (-1, in_h, in_w, 1))
        # Pad
        y = torchF.pad(y, (0, 0, max(self.pady0, 0), max(self.pady1, 0), max(self.padx0, 0), max(self.padx1, 0), 0, 0))
        y = y[:,
            max(-self.pady0, 0): y.shape[1] - max(-self.pady1, 0),
            max(-self.padx0, 0): y.shape[2] - max(-self.padx1, 0),
            :]
        # Convolution
        y = y.permute(0, 3, 1, 2)
        y = y.reshape(-1, 1, in_h + self.pady0 + self.pady1, in_w + self.padx0 + self.padx1)
        y = torchF.conv2d(y, self.k)
        y = y.view(-1, 1, in_h + self.pady0 + self.pady1 - self.kernel_h + 1, in_w + self.padx0 + self.padx1 - self.kernel_w + 1)
        # Downsample
        if in_h != y.shape[1]:
            y = torchF.interpolate(y, size=(in_h, in_w), mode='bilinear')
        y = y.permute(0, 2, 3, 1)
        y = y.reshape(-1, in_c, in_h, in_w)

        if self.bias is not None:
            x1 = torchF.conv2d(y, self.weight * self.w_lrmul, self.bias * self.b_lrmul, stride=self.factor, padding=self.conv_w // 2)
        else:
            x1 = torchF.conv2d(y, self.weight * self.w_lrmul, stride=self.factor, padding=self.conv_w // 2)

        if self.act == 'lrelu':
            out = torchF.leaky_relu(x1, 0.2, inplace=True)
            out = out * np.sqrt(2)
        else:
            out = x1

        return out


class GBlock(nn.Module):
    def __init__(self, inc, outc, layer_idx, opts, k=[1, 3, 3, 1], use_wscale=True, lrmul=1, architecture='skip'):
        super().__init__()
        self.arch = architecture
        self.conv0up = GLayer(inc, outc, layer_idx, up=True, k=k, use_wscale=use_wscale, lrmul=lrmul, opts=opts)
        self.conv1 = GLayer(outc, outc, layer_idx + 1, up=False, k=k, use_wscale=use_wscale, lrmul=lrmul, opts=opts)

    def forward(self, x):
        x, dlatent = x
        x = self.conv0up([x, dlatent])
        x = self.conv1([x, dlatent])

        return x


class DBlock(nn.Module):
    def __init__(self, in1, in2, out3, use_wscale=True, lrmul=1, resample_kernel=[1, 3, 3, 1], architecture='resnet'):
        super().__init__()
        self.arch = architecture
        self.conv0 = Conv2d(inc=in1, outc=in2, ksize=3, use_wscale=use_wscale, lrmul=lrmul, bias=True, act='lrelu')
        self.conv1_down = ConvDownsample2d(ksize=3, inc=in2, outc=out3, k=resample_kernel, bias=True, act='lrelu')
        self.res_conv2_down = ConvDownsample2d(ksize=1, inc=in1, outc=out3, k=resample_kernel, bias=False)

    def forward(self, x):
        t = x.clone()
        x = self.conv0(x)
        x = self.conv1_down(x)

        if self.arch == 'resnet':
            t = self.res_conv2_down(t)
            x = (x + t) * (1 / np.sqrt(2))
        return x

class Minibatch_stddev_layer(nn.Module):
    def __init__(self, g_size=4, num_new_features=1):
        super().__init__()

        self.group_size = g_size
        self.num_new_features = num_new_features

    def forward(self, x):
        n, c, h, w = x.shape

        group_size = min(n, self.group_size)
        y = x.view(group_size, -1, self.num_new_features, c // self.num_new_features, h, w)
        y = y - torch.mean(y, dim=0, keepdim=True)
        y = torch.mean(y ** 2, dim=0)
        y = torch.sqrt(y + 1e-8)
        y = torch.mean(y, dim=[2, 3, 4], keepdim=True)
        y = torch.mean(y, dim=2)
        y = y.repeat(group_size, 1, h, w)

        return torch.cat([x, y], 1)


class G_mapping(nn.Module):
    def __init__(self, mapping_fmaps=512, dlatent_size=512, resolution=1024, label_size=0, mapping_layers=8, normalize_latents=True, use_wscale=True, lrmul=0.01, gain=1):
        super(G_mapping, self).__init__()
        self.mapping_fmaps = mapping_fmaps
        self.mapping_layers = mapping_layers

        self.fc1 = FullyConnectedLayer(self.mapping_fmaps, dlatent_size, gain=gain, lrmul=lrmul, use_wscale=use_wscale)
        self.fc_layers = ModuleList([])
        for _ in range(2, mapping_layers + 1):
            self.fc_layers.append(FullyConnectedLayer(dlatent_size, dlatent_size, gain=gain, lrmul=lrmul, use_wscale=use_wscale))

        self.normalize_latents = normalize_latents
        self.resolution_log2 = int(np.log2(resolution))
        self.num_layers = self.resolution_log2 * 2 - 2
        self.pixel_norm = PixelNormalization()

    def forward(self, x):
        if self.normalize_latents:
            x = self.pixel_norm(x)

        out = self.fc1(x)
        for fc in self.fc_layers:
            out = fc(out)

        out = out.unsqueeze(1)
        out = out.repeat(1, self.num_layers, 1)

        return out

class G_synthesis_stylegan2(nn.Module):
    def __init__(self, opts, fmap_base=8 << 10, num_channels=3, dlatent_size=2<<8, resolution=2<<9, randomize_noise=True, fmap_decay=1.0, fmap_min=1, fmap_max=512,  architecture='skip',  use_wscale=True, lrmul=1, gain=1, act='lrelu',  resample_kernel=[1, 3, 3, 1], fused_modconv=True):
        super(G_synthesis_stylegan2, self).__init__()

        resol_log = int(np.log2(resolution))
        assert resolution == 2 ** resol_log and resolution >= 4
        self.nf = lambda stage: np.clip(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_min, fmap_max)
        assert architecture in ['orig', 'skip', 'resnet']
        num_l = resol_log * 2 - 2

        self.arch = architecture
        self.act = act
        self.resolution_log2 = resol_log
        self.opts = opts

        self.x = torch.nn.Parameter(torch.randn(1, self.nf(1), 4, 4))

        self.rgb0 = ToRGB(inc=self.nf(1), outc=num_channels, res=2, opts=opts)
        self.glayer0 = GLayer(inc=self.nf(1), outc=self.nf(1), layer_idx=0, k=resample_kernel, randomize_noise=randomize_noise, act=self.act, up=False, opts=opts)

        # RGB layer, Block layer
        self.rgb_layers = ModuleList([ToRGB(inc=self.nf(3), outc=num_channels, res=3, opts=opts, fused_modconv=fused_modconv)])
        self.block_layers = ModuleList([GBlock(inc=self.nf(2), outc=self.nf(3), layer_idx=1, opts=opts)])

        for resol in range(4, self.resolution_log2 + 1):
            self.rgb_layers.append(ToRGB(inc=self.nf(resol), outc=num_channels, res=resol, opts=opts, fused_modconv=fused_modconv))
            self.block_layers.append(GBlock(inc=self.nf(resol - 1), outc=self.nf(resol), layer_idx=(resol - 2) * 2 - 1, opts=opts))

        # Upsample-> PyTorch Upsample2d
        self.upsample2d = Upsample2d(opts=opts)
        self.tanh = torch.nn.Tanh()

    def forward(self, dlatent):
        y = None
        x = self.x.repeat(dlatent.shape[0], 1, 1, 1)
        x = self.glayer0([x, dlatent[:, 0]])
        if self.arch == 'skip':
            y = self.rgb0([x, y, dlatent])

        for res, (rgb, block) in enumerate(zip(self.rgb_layers, self.block_layers)):
            x = block([x, dlatent])
            if self.arch == 'skip':
                y = self.upsample2d(y)
            if self.arch == 'skip' or (res + 3) == self.resolution_log2:
                y = rgb([x, y, dlatent])

        return y

class Generator_stylegan2(nn.Module):
    def __init__(self, opts, return_dlatents=True, fmap_base=8 << 10, num_channels=3, mapping_fmaps=2<<8, dlatent_size=2<<8, resol=2<<9, mapping_layers=8, randomize_noise=True, fmap_decay=1.0,  fmap_min=1, fmap_max=512,  architecture='skip',  act='lrelu',  lrmul=0.01, gain=1,  truncation_psi=0.7, truncation_cutoff=8):
        super().__init__()
        assert architecture in ['orig', 'skip']

        self.return_dlatents = return_dlatents
        self.num_channels = num_channels
        self.g_mapping = G_mapping(mapping_fmaps=mapping_fmaps, dlatent_size=dlatent_size, resolution=resol, mapping_layers=mapping_layers, lrmul=lrmul, gain=gain)
        self.g_synthesis = G_synthesis_stylegan2(resolution=resol, architecture=architecture, randomize_noise=randomize_noise, fmap_base=fmap_base, fmap_min=fmap_min, fmap_max=fmap_max, fmap_decay=fmap_decay, act=act, opts=opts)

        self.truncation_cutoff = truncation_cutoff
        self.truncation_psi = truncation_psi

    def forward(self, x):
        dlatents_ = self.g_mapping(x)
        num_layers = dlatents_.shape[1]

        if self.truncation_psi and self.truncation_cutoff:
            batch_avg = torch.mean(dlatents_, dim=1, keepdim=True)
            coefs = np.ones([1, num_layers, 1], dtype=np.float32)
            for i in range(num_layers):
                coefs[:, i, :] *= self.truncation_psi
            dlatents1 = batch_avg + (dlatents_ - batch_avg) * torch.Tensor(coefs).to(dlatents_.device)

        out = self.g_synthesis(dlatents_)

        if self.return_dlatents:
            return out, dlatents_
        else:
            return out

class Discriminator_stylegan2(nn.Module):
    def __init__(self, resol=2<<9, fmap_base=8 << 10, num_channels=3, label_size=0, structure='resnet', fmap_max=512, fmap_min=1, fmap_decay=1.0, mbstd_group_size=4, mbstd_num_features=1, resample_kernel=[1, 3, 3, 1]):
        super().__init__()
        self.resol_log = int(np.log2(resol))
        assert resol == 2 ** self.resol_log and resol >= 4 and self.resol_log >= 4
        self.nf = lambda stage: np.clip(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_min, fmap_max)

        assert structure in ['orig', 'resnet']

        self.structure = structure
        self.label_size = label_size
        self.mbstd_group_size = mbstd_group_size

        self.fromrgb = FromRGB(inc=3, outc=self.nf(self.resol_log - 1), use_wscale=True)
        self.block_layers = ModuleList([])

        for res in range(self.resol_log, 4, -1):
            self.block_layers.append(DBlock(in1=self.nf(res - 1), in2=self.nf(res - 1), out3=self.nf(res - 2), resample_kernel=resample_kernel))

        for res in range(4, 2, -1):
            self.block_layers.append(DBlock(in1=self.nf(res), in2=self.nf(res - 1), out3=self.nf(res - 2), resample_kernel=resample_kernel))

        self.minibatch_stddev = Minibatch_stddev_layer(mbstd_group_size, mbstd_num_features)
        self.conv_last = Conv2d(inc=self.nf(2) + mbstd_num_features, outc=self.nf(1), ksize=3, act='lrelu')
        self.fc_last1 = FullyConnectedLayer(inc=fmap_base, outc=self.nf(0), act='lrelu')
        self.fc_last2 = FullyConnectedLayer(inc=self.nf(0), outc=1, act='linear')

    def forward(self, input):
        x_origin = None
        y = input
        # Main Layer
        x = self.fromrgb([x_origin, y])
        for dblock in self.block_layers:
            x = dblock(x)
        # Final Layer
        if self.mbstd_group_size > 1:
            x = self.minibatch_stddev(x)
        x = self.conv_last(x)

        out = x

        _, c, h, w = out.shape
        out = out.view(-1, h * w * c)

        out = self.fc_last1(out)
        # Output
        if self.label_size == 0:
            out = self.fc_last2(out)
            return out

class StyleGAN2:
    def __init__(self, opts, use_ema=True, ema_decay=0.999):
        self.start_epoch = 0
        self.opts = opts

        # create model
        self.G = Generator_stylegan2(opts=opts, fmap_base=opts.fmap_base, resol=opts.resolution, mapping_layers=opts.mapping_layers, return_dlatents=opts.return_latents, architecture='skip')
        self.D = Discriminator_stylegan2(fmap_base=opts.fmap_base, resol=opts.resolution, structure='resnet')

        # pre-trained weight
        if(os.path.exists(opts.resume)):
            INFO("Loading the pre-trained weight")
            state = torch.load(opts.resume)
            self.G.load_state_dict(state['G'])
            self.D.load_state_dict(state['D'])
            self.start_epoch = state['start_epoch']
        else:
            INFO("Pre-trained weight can't load successfully")

        # multiple GPU support
        if torch.cuda.device_count() > 1:
            INFO("Multiple GPU:" + str(torch.cuda.device_count()) + "\t GPUs")
            self.G = torch.nn.DataParallel(self.G)
            self.D = torch.nn.DataParallel(self.D)
        self.G.to(opts.device)
        self.D.to(opts.device)

        self.use_ema = use_ema
        self.ema_decay = ema_decay

        if self.use_ema:
            from utils.libs import update_average
            self.Gs = copy.deepcopy(self.G)
            # updater func:
            self.ema_updater = update_average
            Gs_beta = 0.99
            self.ema_updater(self.Gs, self.G, beta=Gs_beta)

        self.G.eval()
        self.D.eval()
        if self.use_ema:
            self.Gs.eval()

    def optimize_G(self, gen_optim, dlatent, real_batch, loss_fn):
        fake_samples = self.G(dlatent)
        loss = loss_fn.gen_loss(real_batch, fake_samples)
        gen_optim.zero_grad()
        loss.backward()
        gen_optim.step()
        if self.use_ema:
            self.ema_updater(self.Gs, self.G, self.ema_decay)

        return loss.mean().item()

    def optimize_D(self, dis_optim, dlatent, real_batch, loss_fn):
        fake_samples = self.G(dlatent)
        fake_samples = fake_samples.detach()
        loss = loss_fn.dis_loss(real_batch, fake_samples)
        dis_optim.zero_grad()
        loss.backward()
        dis_optim.step()

        return loss.mean().item()

    def train(self, data_loader, gen_optim, dis_optim, loss_fn, sch_gen, sch_dis):
        self.G.train() # Generator Train mode
        self.D.train() # Discriminator Train mode
        # train
        fix_z = torch.randn([self.opts.batch_size, 512]).to(self.opts.device)
        softplus = torch.nn.Softplus()
        LOSS_D_list = [0.0]
        LOSS_G_list = [0.0]

        for ep in range(self.start_epoch, self.opts.epoch):
            bar = tqdm(data_loader)
            Loss_D_list = []
            Loss_G_list = []
            for i, (real_img,) in enumerate(bar):
                real_img = real_img.to(self.opts.device)
                latents = torch.randn([real_img.size(0), 512]).to(self.opts.device)
                # Optimizer
                d_loss = self.optimize_D(dis_optim, latents, real_img, loss_fn)
                g_loss = self.optimize_G(gen_optim, latents, real_img, loss_fn)

                Loss_G_list.append(g_loss)
                Loss_D_list.append(d_loss)

                bar.set_description("Epoch {} [{}, {}] [G]: {} [D]: {}".format(ep, i + 1, len(data_loader), Loss_G_list[-1], Loss_D_list[-1]))

            # Save result
            LOSS_G_list.append(np.mean(Loss_G_list))
            LOSS_D_list.append(np.mean(Loss_D_list))

            with torch.no_grad():
                if self.opts.return_latents:
                    fake_img = self.G(fix_z)[0].detach().cpu()
                else:
                    fake_img = self.G(fix_z).detach().cpu()
                save_image(fake_img, os.path.join(self.opts.det, 'images', str(ep) + '.png'), nrow=4, normalize=True)

            # Save model
            state = {'G': self.G.state_dict(), 'D': self.D.state_dict(), 'LOSS_G': LOSS_G_list, 'LOSS_D': LOSS_D_list, 'start_epoch': ep}
            torch.save(state, os.path.join(self.opts.det, 'models', 'latest.pth'))

            sch_gen.step()
            sch_dis.step()

        LOSS_D_list = LOSS_D_list[1:]
        LOSS_G_list = LOSS_G_list[1:]
        plotLossCurve(self.opts, LOSS_D_list, LOSS_G_list)

if __name__ == "__main__":
    data = torch.randn(1, 3, 256, 256).cuda()
    print(torch.max(data))
    print(torch.min(data))
    d = Discriminator_stylegan2(resolution=256, structure='resnet', resample_kernel=[1, 3, 3, 1]).cuda()
    print(d(data))
