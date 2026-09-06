from constant import *
from timm.layers.conv_bn_act import ConvNormAct
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm
import torch


class ResBlockUp(nn.Module):
    def __init__(self, channel_in, channel_out):
        super().__init__()

        self.upsample = nn.Upsample(
            scale_factor=2,
            mode="nearest"
        )

        self.conv1 = ConvNormAct(
            in_channels=channel_in,
            out_channels=channel_out,
            kernel_size=3,
            padding=1,
            norm_layer=nn.BatchNorm2d,
            act_layer=nn.SiLU
        )

        self.conv2 = ConvNormAct(
            in_channels=channel_out,
            out_channels=channel_out,
            kernel_size=3,
            padding=1,
            norm_layer=nn.BatchNorm2d,
            act_layer=nn.SiLU
        )

        # Residual/Bypass
        self.shortcut = ConvNormAct(
            in_channels=channel_in,
            out_channels=channel_out
        )

        self.activation = nn.SiLU()

    def forward(self, x):
        # Conv Branch
        y = self.upsample(x)
        y = self.conv1(y)
        y = self.conv2(y)

        #Bypass Branch
        skip = self.upsample(x)
        skip = self.shortcut(skip)

        return self.activation(y+skip)


class ResBlockDown(nn.Module):
    def __init__(self, channel_in, channel_out):
        super().__init__()

        self.conv1 = ConvNormAct(
            in_channels=channel_in,
            out_channels=channel_out,
            kernel_size=3,
            padding=1,
            apply_norm=False, #no norm
            act_layer=None,
            bias=True
        )
        self.conv1.conv = spectral_norm(self.conv1.conv)

        self.conv2 = ConvNormAct(
            in_channels=channel_out,
            out_channels=channel_out,
            kernel_size=3,
            padding=1,
            apply_norm=False, #no norm
            act_layer=nn.SiLU,
            bias=True
        )
        self.conv2.conv = spectral_norm(self.conv2.conv)

        # Residual/Bypass
        self.shortcut = ConvNormAct(
            in_channels=channel_in,
            out_channels=channel_out,
            apply_norm=False, #no norm
            act_layer=nn.SiLU,
            bias=True
        )
        self.shortcut.conv = spectral_norm(self.shortcut.conv)

        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
        self.activation = nn.SiLU()

    def forward(self, x):
        # Conv Branch
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.downsample(y)

        #Bypass Branch
        skip = self.downsample(x)
        skip = self.shortcut(skip)

        return self.activation(y+skip)

class Generator(nn.Module):

    def __init__(self, latent_dim = LATENT_DIM):
        super().__init__()

        self.latent_dim = latent_dim

        self.projection = nn.Linear(latent_dim,
                                    512* (IMAGE_SIZE//32)**2
                                    )

        self.blocks = nn.Sequential(
            ResBlockUp(512,512),
            ResBlockUp(512,256),
            ResBlockUp(256,128),
            ResBlockUp(128,64),
            ResBlockUp(64,32)
        )

        self.hidden_to_rgb = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                in_channels=32,
                out_channels=IMAGE_CHANNELS,
                kernel_size=3,
                padding=1
            ),
            nn.Tanh()
        )

    def forward(self, z:torch.Tensor):
        out = self.projection(z)
        out = out.view(
            z.shape[0],
            512,
            (IMAGE_SIZE//32),(IMAGE_SIZE//32)
        )

        out = self.blocks(out)
        return self.hidden_to_rgb(out)

class Critic(nn.Module):

    def __init__(self):
        super().__init__()

        self.from_rgb = ConvNormAct(
            in_channels=IMAGE_CHANNELS,
            out_channels=32,
            kernel_size=3,
            padding=1,
            apply_norm=False,
            act_layer=nn.SiLU
        )
        self.from_rgb.conv = spectral_norm(self.from_rgb.conv)

        self.blocks = nn.Sequential(
            ResBlockDown(32,64),
            ResBlockDown(64,128),
            ResBlockDown(128,256),
            ResBlockDown(256,512),
            ResBlockDown(512,512)
        )

        self.activation = nn.SiLU()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.head = spectral_norm(nn.Linear(512,1))

    def forward(self, x):
        x = self.from_rgb(x)
        x = self.blocks(x)
        x = self.activation(x)
        x = self.avg_pool(x)
        x = x.flatten(1)
        return self.head(x)
