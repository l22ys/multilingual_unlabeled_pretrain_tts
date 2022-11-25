# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.


import torch
from torch import nn
from torch.nn.modules.conv import Conv1d
from omegaconf import DictConfig
from torch.nn import functional as F
# from models_.hifigan_discriminator import DiscriminatorP, MultiPeriodDiscriminator


LRELU_SLOPE = 0.1

# copied from https://github.com/coqui-ai/TTS/blob/096b35f6396d063205318fbaef10f08b9f699832/TTS/vocoder/models/hifigan_discriminator.py#L9
class DiscriminatorP(torch.nn.Module):
    """HiFiGAN Periodic Discriminator
    Takes every Pth value from the input waveform and applied a stack of convoluations.
    Note:
        if `period` is 2
        `waveform = [1, 2, 3, 4, 5, 6 ...] --> [1, 3, 5 ... ] --> convs -> score, feat`
    Args:
        x (Tensor): input waveform.
    Returns:
        [Tensor]: discriminator scores per sample in the batch.
        [List[Tensor]]: list of features from each convolutional layer.
    Shapes:
        x: [B, 1, T]
    """

    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super().__init__()
        self.period = period
        get_padding = lambda k, d: int((k * d - d) / 2)
        norm_f = nn.utils.spectral_norm if use_spectral_norm else nn.utils.weight_norm
        self.convs = nn.ModuleList(
            [
                norm_f(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
                norm_f(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
                norm_f(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
                norm_f(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
                norm_f(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
            ]
        )
        self.conv_post = norm_f(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        """
        Args:
            x (Tensor): input waveform.
        Returns:
            [Tensor]: discriminator scores per sample in the batch.
            [List[Tensor]]: list of features from each convolutional layer.
        Shapes:
            x: [B, 1, T]
        """
        feat = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            feat.append(x)
        x = self.conv_post(x)
        feat.append(x)
        x = torch.flatten(x, 1, -1)

        return x, feat

# copied from https://github.com/coqui-ai/TTS/blob/096b35f6396d063205318fbaef10f08b9f699832/TTS/tts/layers/vits/discriminator.py#L50
class DiscriminatorS(torch.nn.Module):
    """HiFiGAN Scale Discriminator. Channel sizes are different from the original HiFiGAN.
    Args:
        use_spectral_norm (bool): if `True` swith to spectral norm instead of weight norm.
    """

    def __init__(self, use_spectral_norm=False):
        super().__init__()
        norm_f = nn.utils.spectral_norm if use_spectral_norm else nn.utils.weight_norm
        self.convs = nn.ModuleList(
            [
                norm_f(Conv1d(1, 16, 15, 1, padding=7)),
                norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
                norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
                norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
                norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        """
        Args:
            x (Tensor): input waveform.
        Returns:
            Tensor: discriminator scores.
            List[Tensor]: list of features from the convolutiona layers.
        """
        feat = []
        for l in self.convs:
            x = l(x)
            x = torch.nn.functional.leaky_relu(x, 0.1)
            feat.append(x)
        x = self.conv_post(x)
        feat.append(x)
        x = torch.flatten(x, 1, -1)
        return x, feat

# copied from https://github.com/coqui-ai/TTS/blob/096b35f6396d063205318fbaef10f08b9f699832/TTS/tts/layers/vits/discriminator.py#L50
class VitsDiscriminator(nn.Module):
    """VITS discriminator wrapping one Scale Discriminator and a stack of Period Discriminator.
    ::
        waveform -> ScaleDiscriminator() -> scores_sd, feats_sd --> append() -> scores, feats
               |--> MultiPeriodDiscriminator() -> scores_mpd, feats_mpd ^
    Args:
        use_spectral_norm (bool): if `True` swith to spectral norm instead of weight norm.
    """

    def __init__(self, periods=(2, 3, 5, 7, 11), use_spectral_norm=False):
        super().__init__()
        self.nets = nn.ModuleList()
        self.nets.append(DiscriminatorS(use_spectral_norm=use_spectral_norm))
        self.nets.extend([DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods])

    def forward(self, x, x_hat=None):
        """
        Args:
            x (Tensor): ground truth waveform. 
            x_hat (Tensor): predicted waveform.
        Returns:
            List[Tensor]: discriminator scores.
            List[List[Tensor]]: list of list of features from each layers of each discriminator.
        """
        x_scores = []
        x_hat_scores = [] if x_hat is not None else None
        x_feats = []
        x_hat_feats = [] if x_hat is not None else None
        for net in self.nets:
            x_score, x_feat = net(x)
            x_scores.append(x_score)
            x_feats.append(x_feat)
            if x_hat is not None:
                x_hat_score, x_hat_feat = net(x_hat)
                x_hat_scores.append(x_hat_score)
                x_hat_feats.append(x_hat_feat)
        return x_scores, x_feats, x_hat_scores, x_hat_feats

    
class WrappedVitsDiscriminator(nn.Module):
    def __init__(self, conf: DictConfig):
        super(WrappedVitsDiscriminator, self).__init__()
        self.conf = conf
        self.disc = VitsDiscriminator(
                periods=self.conf.periods_multi_period_discriminator,
                use_spectral_norm=self.conf.use_spectral_norm_disriminator,)

    def forward(self, x, x_hat=None):
        return self.disc(x, x_hat)
    
