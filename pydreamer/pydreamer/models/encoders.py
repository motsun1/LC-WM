from typing import Optional, Union
import torch
import torch.nn as nn
import torch.distributions as D

from .functions import *
from .common import *


class MultiEncoder(nn.Module):

    def __init__(self, conf):
        super().__init__()
        self.reward_input = conf.reward_input
        if conf.reward_input:
            encoder_channels = conf.image_channels + 2  # + reward, terminal
        else:
            encoder_channels = conf.image_channels

        if conf.image_encoder == 'cnn':
            self.encoder_image = ConvEncoder(in_channels=encoder_channels,
                                             cnn_depth=conf.cnn_depth)
        elif conf.image_encoder == 'cnn224':
            self.encoder_image = ConvEncoder224(in_channels=encoder_channels,
                                               cnn_depth=conf.cnn_depth)
        elif conf.image_encoder == 'dense':
            self.encoder_image = DenseEncoder(in_dim=conf.image_size * conf.image_size * encoder_channels,
                                              out_dim=256,
                                              hidden_layers=conf.image_encoder_layers,
                                              layer_norm=conf.layer_norm)
        elif not conf.image_encoder:
            self.encoder_image = None
        else:
            assert False, conf.image_encoder

        if conf.vecobs_size:
            self.encoder_vecobs = MLP(conf.vecobs_size, 256, hidden_dim=400, hidden_layers=2, layer_norm=conf.layer_norm)
        else:
            self.encoder_vecobs = None

        assert self.encoder_image or self.encoder_vecobs, "Either image_encoder or vecobs_size should be set"
        self.out_dim = ((self.encoder_image.out_dim if self.encoder_image else 0) +
                        (self.encoder_vecobs.out_dim if self.encoder_vecobs else 0))

    def forward(self, obs: Dict[str, Tensor]) -> TensorTBE:
        # TODO:
        #  1) Make this more generic, e.g. working without image input or without vecobs
        #  2) Treat all inputs equally, adding everything via linear layer to embed_dim

        embeds = []

        if self.encoder_image:
            image = obs['image']
            T, B, C, H, W = image.shape
            if self.reward_input:
                reward = obs['reward']
                terminal = obs['terminal']
                reward_plane = reward.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand((T, B, 1, H, W))
                terminal_plane = terminal.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand((T, B, 1, H, W))
                image = torch.cat([image,  # (T,B,C+2,H,W)
                                reward_plane.to(image.dtype),
                                terminal_plane.to(image.dtype)], dim=-3)

            embed_image = self.encoder_image.forward(image)  # (T,B,E)
            embeds.append(embed_image)

        if self.encoder_vecobs:
            embed_vecobs = self.encoder_vecobs(obs['vecobs'])
            embeds.append(embed_vecobs)

        embed = torch.cat(embeds, dim=-1)  # (T,B,E+256)
        return embed


class ConvEncoder(nn.Module):

    def __init__(self, in_channels=3, cnn_depth=32, activation=nn.ELU):
        super().__init__()
        self.out_dim = cnn_depth * 32
        kernels = (4, 4, 4, 4)
        stride = 2
        d = cnn_depth
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, d, kernels[0], stride),
            activation(),
            nn.Conv2d(d, d * 2, kernels[1], stride),
            activation(),
            nn.Conv2d(d * 2, d * 4, kernels[2], stride),
            activation(),
            nn.Conv2d(d * 4, d * 8, kernels[3], stride),
            activation(),
            nn.Flatten()
        )

    def forward(self, x):
        x, bd = flatten_batch(x, 3)
        y = self.model(x)
        y = unflatten_batch(y, bd)
        return y


class ConvEncoder224(nn.Module):
    """
    Encoder designed for 224x224 images, with 6 convolutional layers and final linear projection.
    Input: 224x224 -> Output: 2048
    """
    def __init__(self, in_channels=3, cnn_depth=64, activation=nn.ELU):
        super().__init__()
        self.out_dim = 2048  # 固定出力次元
        kernels = (4, 4, 4, 4, 4, 4)
        stride = 2
        padding = 1
        d = cnn_depth
        
        # 224x224入力に対する6層の畳み込みエンコーダ
        # 出力サイズの変化: 224->112->56->28->14->7->3
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, d, kernels[0], stride, padding),
            activation(),
            nn.Conv2d(d, d * 2, kernels[1], stride, padding),
            activation(),
            nn.Conv2d(d * 2, d * 4, kernels[2], stride, padding),
            activation(),
            nn.Conv2d(d * 4, d * 8, kernels[3], stride, padding),
            activation(),
            nn.Conv2d(d * 8, d * 16, kernels[4], stride, padding),
            activation(),
            nn.Conv2d(d * 16, d * 16, kernels[5], stride, padding),
            activation(),
            nn.Flatten()
        )
        
        # 3x3x(16*d) = 9216次元 -> 2048次元に圧縮
        self.dim_reducer = nn.Linear(3 * 3 * d * 16, self.out_dim)
        
    def forward(self, x):
        x, bd = flatten_batch(x, 3)
        features = self.feature_extractor(x)
        y = self.dim_reducer(features)
        y = unflatten_batch(y, bd)
        return y


class DenseEncoder(nn.Module):

    def __init__(self, in_dim, out_dim=256, activation=nn.ELU, hidden_dim=400, hidden_layers=2, layer_norm=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        norm = nn.LayerNorm if layer_norm else NoNorm
        layers = [nn.Flatten()]
        layers += [
            nn.Linear(in_dim, hidden_dim),
            norm(hidden_dim, eps=1e-3),
            activation()]
        for _ in range(hidden_layers - 1):
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                norm(hidden_dim, eps=1e-3),
                activation()]
        layers += [
            nn.Linear(hidden_dim, out_dim),
            activation()]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x, bd = flatten_batch(x, 3)
        y = self.model(x)
        y = unflatten_batch(y, bd)
        return y
