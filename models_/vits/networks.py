# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# copied and modified from https://github.com/coqui-ai/TTS/blob/d6ad9a05b40e9f3aa876ecfa1c0d23339f373df4/TTS/tts/layers/glow_tts/glow.py#L71
# copied and modified from https://github.com/coqui-ai/TTS/blob/096b35f6396d063205318fbaef10f08b9f699832/TTS/tts/layers/vits/networks.py#L235



from distutils.version import LooseVersion
import math

import torch
from torch import nn

from models_.glow_tts.transformer import RelativePositionTransformer, RelativePositionTransformerMultilingual
from utils_.helpers import sequence_mask
from models_.generic.wavenet import WN
from models_.generic.normalization import LayerNorm2
import torch.nn.functional as F


LRELU_SLOPE = 0.1


# copied from https://github.com/coqui-ai/TTS/blob/096b35f6396d063205318fbaef10f08b9f699832/TTS/tts/layers/vits/networks.py#L235
def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape

# copied from https://github.com/coqui-ai/TTS/blob/096b35f6396d063205318fbaef10f08b9f699832/TTS/tts/layers/vits/networks.py#L235
def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)

# copied from https://github.com/coqui-ai/TTS/blob/096b35f6396d063205318fbaef10f08b9f699832/TTS/tts/layers/vits/networks.py#L235
def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)



# copied from https://github.com/coqui-ai/TTS/blob/d6ad9a05b40e9f3aa876ecfa1c0d23339f373df4/TTS/tts/layers/glow_tts/glow.py#L71
class InvConvNear(nn.Module):
    """Invertible Convolution with input splitting as in GlowTTS paper.
    https://arxiv.org/pdf/1811.00002.pdf
    Args:
        channels (int): input and output channels.
        num_splits (int): number of splits, also H and W of conv layer.
        no_jacobian (bool): enable/disable jacobian computations.
    Note:
        Split the input into groups of size self.num_splits and
        perform 1x1 convolution separately. Cast 1x1 conv operation
        to 2d by reshaping the input for efficiency.
    """

    def __init__(self, channels, num_splits=4, no_jacobian=False, **kwargs):  # pylint: disable=unused-argument
        super().__init__()
        assert num_splits % 2 == 0
        self.channels = channels
        self.num_splits = num_splits
        self.no_jacobian = no_jacobian
        self.weight_inv = None

        if LooseVersion(torch.__version__) < LooseVersion("1.9"):
            w_init = torch.qr(torch.FloatTensor(self.num_splits, self.num_splits).normal_())[0]
        else:
            w_init = torch.linalg.qr(torch.FloatTensor(self.num_splits, self.num_splits).normal_(), "complete")[0]

        if torch.det(w_init) < 0:
            w_init[:, 0] = -1 * w_init[:, 0]
        self.weight = nn.Parameter(w_init)

    def forward(self, x, x_mask=None, reverse=False, **kwargs):  # pylint: disable=unused-argument
        """
        Shapes:
            - x: :math:`[B, C, T]`
            - x_mask: :math:`[B, 1, T]`
        """
        b, c, t = x.size()
        assert c % self.num_splits == 0
        if x_mask is None:
            x_mask = 1
            x_len = torch.ones((b,), dtype=x.dtype, device=x.device) * t 
        else:
            x_len = torch.sum(x_mask, [1, 2])

        x = x.view(b, 2, c // self.num_splits, self.num_splits // 2, t)
        x = x.permute(0, 1, 3, 2, 4).contiguous().view(b, self.num_splits, c // self.num_splits, t)

        if reverse:
            if self.weight_inv is not None:
                weight = self.weight_inv
            else:
                weight = torch.inverse(self.weight.float()).to(dtype=self.weight.dtype)
            logdet = None
        else:
            weight = self.weight
            if self.no_jacobian:
                logdet = 0 
            else:
                logdet = torch.logdet(self.weight) * (c / self.num_splits) * x_len  # [b]

        weight = weight.view(self.num_splits, self.num_splits, 1, 1)
        z = F.conv2d(x, weight)

        z = z.view(b, 2, self.num_splits // 2, c // self.num_splits, t)
        z = z.permute(0, 1, 3, 2, 4).contiguous().view(b, c, t) * x_mask 
        return z, logdet

    def store_inverse(self):
        weight_inv = torch.inverse(self.weight.float()).to(dtype=self.weight.dtype)
        self.weight_inv = nn.Parameter(weight_inv, requires_grad=False)

# copied from https://github.com/coqui-ai/TTS/blob/096b35f6396d063205318fbaef10f08b9f699832/TTS/tts/layers/vits/networks.py#L235
class TextEncoder(nn.Module):
    def __init__(
        self,
        n_vocab: int,
        out_channels: int,
        hidden_channels: int,
        hidden_channels_ffn: int,
        num_heads: int,
        num_layers: int,
        kernel_size: int,
        dropout_p: float,
        language_emb_dim: int = None,
    ):
        """Text Encoder for VITS model.
        Args:
            n_vocab (int): Number of characters for the embedding layer.
            out_channels (int): Number of channels for the output.
            hidden_channels (int): Number of channels for the hidden layers.
            hidden_channels_ffn (int): Number of channels for the convolutional layers.
            num_heads (int): Number of attention heads for the Transformer layers.
            num_layers (int): Number of Transformer layers.
            kernel_size (int): Kernel size for the FFN layers in Transformer network.
            dropout_p (float): Dropout rate for the Transformer layers.
        """
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        self.emb = nn.Embedding(n_vocab, hidden_channels, padding_idx=0)

        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5) 

        if language_emb_dim:
            hidden_channels += language_emb_dim

        self.encoder = RelativePositionTransformer(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            hidden_channels=hidden_channels,
            hidden_channels_ffn=hidden_channels_ffn,
            num_heads=num_heads,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout_p=dropout_p,
            layer_norm_type="2",
            rel_attn_window_size=4,
        )

        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, lang_emb=None):
        """
        Shapes:
            - x: :math:`[B, T]`
            - x_length: :math:`[B]`
        """
        assert x.shape[0] == x_lengths.shape[0]
        x = self.emb(x) * math.sqrt(self.hidden_channels)  # [b, t, h]

        # concat the lang emb in embedding chars
        if lang_emb is not None:
            x = torch.cat((x, lang_emb.transpose(2, 1).expand(x.size(0), x.size(1), -1)), dim=-1) # [B, time, text_emb + lang_emb]


        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)  # [b, 1, t]

        x = self.encoder(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask

        m, logs = torch.split(stats, self.out_channels, dim=1) 
        return x, m, logs, x_mask 
        
# modified version of above textencoder
class TextEncoderwithoutEmb(nn.Module):
    def __init__(
        self,
        out_channels: int,
        hidden_channels: int,
        hidden_channels_ffn: int,
        num_heads: int,
        num_layers: int,
        kernel_size: int,
        dropout_p: float,
        language_emb_dim: int = None,
    ):
        """Text Encoder for VITS model.
        Args:
            n_vocab (int): Number of characters for the embedding layer.
            out_channels (int): Number of channels for the output.
            hidden_channels (int): Number of channels for the hidden layers.
            hidden_channels_ffn (int): Number of channels for the convolutional layers.
            num_heads (int): Number of attention heads for the Transformer layers.
            num_layers (int): Number of Transformer layers.
            kernel_size (int): Kernel size for the FFN layers in Transformer network.
            dropout_p (float): Dropout rate for the Transformer layers.
        """
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        if language_emb_dim:
            hidden_channels += language_emb_dim

        self.encoder = RelativePositionTransformer(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            hidden_channels=hidden_channels,
            hidden_channels_ffn=hidden_channels_ffn,
            num_heads=num_heads,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout_p=dropout_p,
            layer_norm_type="2",
            rel_attn_window_size=4,
        )

        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, lang_emb=None):
        """
        Shapes:
            - x: :math:`[B, T]`
            - x_length: :math:`[B]`
        """
        # concat the lang emb in embedding chars
        if lang_emb is not None:
            x = torch.cat((x, lang_emb.transpose(2, 1).expand(x.size(0), x.size(1), -1)), dim=-1) # [B, time, text_emb + lang_emb]
            

        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)  # [b, 1, t]

        x = self.encoder(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask

        m, logs = torch.split(stats, self.out_channels, dim=1) 
        return x, m, logs, x_mask # x: [B, text_emb_dim + lang_emb_dim, T], m: [B, out_channels, T], logs: [B, out_channels, T], x_mask : [B,1,T] 

# modified version of above textencoder
class TextEncoderwithoutEmbMultilingual(nn.Module):
    def __init__(
        self,
        out_channels: int,
        hidden_channels: int,
        hidden_channels_ffn: int,
        num_heads: int,
        num_layers: int,
        kernel_size: int,
        dropout_p: float,
        language_emb_dim: int = None,
        num_languages: int = 1,
        heads_share: bool = True,
    ):
        """Text Encoder for VITS model.
        Args:
            n_vocab (int): Number of characters for the embedding layer.
            out_channels (int): Number of channels for the output.
            hidden_channels (int): Number of channels for the hidden layers.
            hidden_channels_ffn (int): Number of channels for the convolutional layers.
            num_heads (int): Number of attention heads for the Transformer layers.
            num_layers (int): Number of Transformer layers.
            kernel_size (int): Kernel size for the FFN layers in Transformer network.
            dropout_p (float): Dropout rate for the Transformer layers.
        """
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        if language_emb_dim:
            hidden_channels += language_emb_dim

        self.encoder = RelativePositionTransformerMultilingual(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            hidden_channels=hidden_channels,
            hidden_channels_ffn=hidden_channels_ffn,
            num_heads=num_heads,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout_p=dropout_p,
            layer_norm_type="2",
            rel_attn_window_size=4,
            num_languages= num_languages,
            heads_share = heads_share
        )

        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, lang_emb=None):
        """
        Shapes:
            - x: :math:`[B, T]`
            - x_length: :math:`[B]`
        """
        # concat the lang emb in embedding chars
        if lang_emb is not None:
            x = torch.cat((x, lang_emb.transpose(2, 1).expand(x.size(0), x.size(1), -1)), dim=-1) # [B, time, text_emb + lang_emb]
        
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)  # [b, 1, t]

        x = self.encoder(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask

        m, logs = torch.split(stats, self.out_channels, dim=1) 
        return x, m, logs, x_mask # x: [B, text_emb_dim + lang_emb_dim, T], m: [B, out_channels, T], logs: [B, out_channels, T], x_mask : [B,1,T] 


# copied from https://github.com/coqui-ai/TTS/blob/096b35f6396d063205318fbaef10f08b9f699832/TTS/tts/layers/vits/networks.py#L235
class ResidualCouplingBlock(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        num_layers,
        dropout_p=0,
        cond_channels=0,
        mean_only=False,
    ):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.half_channels = channels // 2
        self.mean_only = mean_only
        # input layer
        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        # coupling layers
        self.enc = WN(
            hidden_channels,
            hidden_channels,
            kernel_size,
            dilation_rate,
            num_layers,
            dropout_p=dropout_p,
            c_in_channels=cond_channels,
        )
        # output layer
        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  This helps with training stability
        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        """
        Note:
            Set `reverse` to True for inference.
        Shapes:
            - x: :math:`[B, C, T]`
            - x_mask: :math:`[B, 1, T]`
            - g: :math:`[B, C, 1]`
        """
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, log_scale = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats # [B, self.half_channels, T]
            log_scale = torch.zeros_like(m) 

        if not reverse:
            x1 = m + x1 * torch.exp(log_scale) * x_mask
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(log_scale, [1, 2]) # [B] 
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-log_scale) * x_mask
            x = torch.cat([x0, x1], 1)
            return x

# copied from https://github.com/coqui-ai/TTS/blob/096b35f6396d063205318fbaef10f08b9f699832/TTS/tts/layers/vits/networks.py#L235
class ResidualCouplingBlocks(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        num_layers: int,
        num_flows=4,
        cond_channels=0,
    ):
        """Redisual Coupling blocks for VITS flow layers.
        Args:
            channels (int): Number of input and output tensor channels.
            hidden_channels (int): Number of hidden network channels.
            kernel_size (int): Kernel size of the WaveNet layers.
            dilation_rate (int): Dilation rate of the WaveNet layers.
            num_layers (int): Number of the WaveNet layers.
            num_flows (int, optional): Number of Residual Coupling blocks. Defaults to 4.
            cond_channels (int, optional): Number of channels of the conditioning tensor. Defaults to 0.
        """
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.num_layers = num_layers
        self.num_flows = num_flows
        self.cond_channels = cond_channels

        self.flows = nn.ModuleList()
        for _ in range(num_flows):
            self.flows.append(
                ResidualCouplingBlock(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    num_layers,
                    cond_channels=cond_channels,
                    mean_only=True, 
                )
            )

    def forward(self, x, x_mask, g=None, reverse=False):
        """
        Note:
            Set `reverse` to True for inference.
        Shapes:
            - x: :math:`[B, C, T]`
            - x_mask: :math:`[B, 1, T]`
            - g: :math:`[B, C, 1]`
        """
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
                x = torch.flip(x, [1])
        else:
            for flow in reversed(self.flows):
                x = torch.flip(x, [1])
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x

# modified version of above ResidualCouplingBlocks
# - mean_only = True -> False : eliminate volume preserving constraint
class ResidualCouplingBlocksScale(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        num_layers: int,
        num_flows=4,
        cond_channels=0,
    ):
        """Redisual Coupling blocks for VITS flow layers.
        Args:
            channels (int): Number of input and output tensor channels.
            hidden_channels (int): Number of hidden network channels.
            kernel_size (int): Kernel size of the WaveNet layers.
            dilation_rate (int): Dilation rate of the WaveNet layers.
            num_layers (int): Number of the WaveNet layers.
            num_flows (int, optional): Number of Residual Coupling blocks. Defaults to 4.
            cond_channels (int, optional): Number of channels of the conditioning tensor. Defaults to 0.
        """
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.num_layers = num_layers
        self.num_flows = num_flows
        self.cond_channels = cond_channels

        self.flows = nn.ModuleList()
        for _ in range(num_flows):
            self.flows.append(
                ResidualCouplingBlock(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    num_layers,
                    cond_channels=cond_channels,
                    mean_only=False,
                )
            )

    def forward(self, x, x_mask, g=None, reverse=False):
        """
        Note:
            Set `reverse` to True for inference.
        Shapes:
            - x: :math:`[B, C, T]`
            - x_mask: :math:`[B, 1, T]`
            - g: :math:`[B, C, 1]`
        """
        if not reverse:
            logdet = 0
            for flow in self.flows:
                x, logdet_i = flow(x, x_mask, g=g, reverse=reverse)
                x = torch.flip(x, [1])
                logdet += logdet_i # [B]
            return x, logdet
        else:
            for flow in reversed(self.flows):
                x = torch.flip(x, [1])
                x = flow(x, x_mask, g=g, reverse=reverse)
            return x
        
# copied from https://github.com/coqui-ai/TTS/blob/096b35f6396d063205318fbaef10f08b9f699832/TTS/tts/layers/vits/networks.py#L235
class PosteriorEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        num_layers: int,
        cond_channels=0,
    ):
        """Posterior Encoder of VITS model.
        ::
            x -> conv1x1() -> WaveNet() (non-causal) -> conv1x1() -> split() -> [m, s] -> sample(m, s) -> z
        Args:
            in_channels (int): Number of input tensor channels.
            out_channels (int): Number of output tensor channels.
            hidden_channels (int): Number of hidden channels.
            kernel_size (int): Kernel size of the WaveNet convolution layers.
            dilation_rate (int): Dilation rate of the WaveNet layers.
            num_layers (int): Number of the WaveNet layers.
            cond_channels (int, optional): Number of conditioning tensor channels. Defaults to 0.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.num_layers = num_layers
        self.cond_channels = cond_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WN(
            hidden_channels, hidden_channels, kernel_size, dilation_rate, num_layers, c_in_channels=cond_channels
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        """
        Shapes:
            - x: :math:`[B, C, T]`
            - x_lengths: :math:`[B, 1]`
            - g: :math:`[B, C, 1]`
        """
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype) # [B,1,T]
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        mean, log_scale = torch.split(stats, self.out_channels, dim=1)
        z = (mean + torch.randn_like(mean) * torch.exp(log_scale)) * x_mask 
        return z, mean, log_scale, x_mask

    
class LinearPosteriorEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.pre = nn.Conv1d(in_channels, out_channels, 1)
        self.layer_norm = LayerNorm2(out_channels)
        self.proj = nn.Conv1d(out_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths):
        """
        Shapes:
            - x: :math:`[B, C, T]`
            - x_lengths: :math:`[B, 1]`
        """
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.layer_norm(x)
        stats = self.proj(x) * x_mask
        mean, log_scale = torch.split(stats, self.out_channels, dim=1)
        z = (mean + torch.randn_like(mean) * torch.exp(log_scale)) * x_mask
        return z, mean, log_scale, x_mask

    
class CustomSpeakerEmb(nn.Module):
    def __init__(self, num_speaker, speaker_embedding_dim, speaker_encoder, w2v_extractor, initialized=False) -> None:
        super().__init__()
        self.speaker_encoder = speaker_encoder
        self.w2v_extractor = w2v_extractor

        self.emb_s = nn.Embedding(num_speaker, speaker_embedding_dim)
        if initialized:
            self.initialized = torch.zeros((num_speaker))
        else:
            self.initialized = torch.ones((num_speaker))

    def forward(self, sid, cropped_waveform):
        if self.initialized.sum() > 0: 
            uninitialized_spks = []
            for s in sid:
                if self.initialized[s] == 1:
                    uninitialized_spks.append(s)

            if uninitialized_spks != []:
                with torch.no_grad():
                    w2v_1st_block_feature = self.w2v_extractor.reduced_forward(cropped_waveform, output_hidden_states = True)[0][1] #[B, t, dim]
                    w2v_1st_block_feature = w2v_1st_block_feature.permute((0, 2, 1)) # [B, dim, t]
                    g = self.speaker_encoder(w2v_1st_block_feature) # [B, emb_dim]
                
                    for s in uninitialized_spks:
                        if self.initialized[s] == 1:
                            self.emb_s.weight[s] = g[s]
                            self.initialized[s] = 0
                        else:
                            pass
        
        g = self.emb_s(sid)
        return g