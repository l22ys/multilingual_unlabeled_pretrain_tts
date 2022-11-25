# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import torch
from torch import nn

# copied from https://github.com/coqui-ai/TTS/blob/dev/TTS/tts/layers/generic/normalization.py
class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-4):
        """Layer norm for the 2nd dimension of the input.
        Args:
            channels (int): number of channels (2nd dimension) of the input.
            eps (float): to prevent 0 division
        Shapes:
            - input: (B, C, T)
            - output: (B, C, T)
        """
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(1, channels, 1) * 0.1)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x):
        mean = torch.mean(x, 1, keepdim=True)
        variance = torch.mean((x - mean) ** 2, 1, keepdim=True)
        x = (x - mean) * torch.rsqrt(variance + self.eps)
        x = x * self.gamma + self.beta
        return x

# copied from https://github.com/coqui-ai/TTS/blob/dev/TTS/tts/layers/generic/normalization.py
class LayerNorm2(nn.Module):
    """Layer norm for the 2nd dimension of the input using torch primitive.
    Args:
        channels (int): number of channels (2nd dimension) of the input.
        eps (float): to prevent 0 division
    Shapes:
        - input: (B, C, T)
        - output: (B, C, T)
    """

    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = torch.nn.functional.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)
