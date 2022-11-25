# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import math
from typing import Optional
import torch.nn as nn
from omegaconf import DictConfig
from typing import Optional, Dict
from torch.nn import functional as F
import torch
from models_.vits.networks import TextEncoder, TextEncoderwithoutEmbMultilingual, PosteriorEncoder, ResidualCouplingBlocks
from utils_.helpers import maximum_path, rand_segments, sequence_mask, segment, generate_path
from models_.vits.stochastic_duration_predictor import StochasticDurationPredictor
from models_.glow_tts.duration_predictor import DurationPredictor
from models_.hifigan_generator import HifiganGenerator
from models_.reduced_w2v import ReducedWav2Vec2ForPreTraining
from models_.ecapa import Speaker

# copied and modified from https://github.com/coqui-ai/TTS/blob/096b35f6396d063205318fbaef10f08b9f699832/TTS/tts/models/vits.py
# - 
class Base(nn.Module):
    def __init__(self) -> None:
        super(Base, self).__init__()
      
    @staticmethod
    def _set_x_lengths(x, aux_input):
        if "x_lengths" in aux_input and aux_input["x_lengths"] is not None:
            return aux_input["x_lengths"]
        return torch.tensor(x.shape[1:2]).to(x.device)
    
    @staticmethod
    def _set_cond_input(aux_input : Dict = {"d_vectors": None, "speaker_ids": None, "language_ids": None}):
        sid, g, lid, durations = None, None, None, None
        if "speaker_ids" in aux_input and aux_input["speaker_ids"] is not None:
            sid = aux_input["speaker_ids"] 
            if sid.ndim == 0:
                sid = sid.unsqueeze_(0)
        if "d_vectors" in aux_input and aux_input["d_vectors"] is not None: 
            g = F.normalize(aux_input["d_vectors"]).unsqueeze(-1)
            if g.ndim == 2:
                g = g.unsqueeze_(0) 

        if "language_ids" in aux_input and aux_input["language_ids"] is not None:
            lid = aux_input["language_ids"]
            if lid.ndim == 0:
                lid = lid.unsqueeze_(0) 
        
        if "durations" in aux_input and aux_input["durations"] is not None:
            durations = aux_input["durations"]

        return sid, g, lid, durations

    def forward_mas_without_dp(self, z_p, m_p, logs_p, x, x_mask, y_mask):
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)
        with torch.no_grad():
            o_scale = torch.exp(-2 * logs_p)
            logp1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1]).unsqueeze(-1)  # [b, t, 1]
            logp2 = torch.einsum("klm, kln -> kmn", [o_scale, -0.5 * (z_p**2)])
            logp3 = torch.einsum("klm, kln -> kmn", [m_p * o_scale, z_p])
            logp4 = torch.sum(-0.5 * (m_p**2) * o_scale, [1]).unsqueeze(-1)  # [b, t, 1]
            logp = logp2 + logp3 + logp1 + logp4
            attn = maximum_path(logp, attn_mask.squeeze(1)).unsqueeze(1).detach()  # [b, 1, t, t']
        return attn

    def forward_mas(self, outputs, z_p, m_p, logs_p, x, x_mask, y_mask, g, lang_emb):
        # find the alignment path
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)
        with torch.no_grad():
            o_scale = torch.exp(-2 * logs_p)
            logp1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1]).unsqueeze(-1)  # [b, t, 1]
            logp2 = torch.einsum("klm, kln -> kmn", [o_scale, -0.5 * (z_p**2)])
            logp3 = torch.einsum("klm, kln -> kmn", [m_p * o_scale, z_p])
            logp4 = torch.sum(-0.5 * (m_p**2) * o_scale, [1]).unsqueeze(-1)  # [b, t, 1]
            logp = logp2 + logp3 + logp1 + logp4
            attn = maximum_path(logp, attn_mask.squeeze(1)).unsqueeze(1).detach()  # [b, 1, t, t']

        # duration predictor
        attn_durations = attn.sum(3)
        if self.conf.use_sdp:
            loss_duration = self.duration_predictor(
                x.detach() if self.conf.detach_dp_input else x,
                x_mask,
                attn_durations,
                g=g.detach() if self.conf.detach_dp_input and g is not None else g,
                lang_emb=lang_emb.detach() if self.conf.detach_dp_input and lang_emb is not None else lang_emb,
            )
            loss_duration = loss_duration / torch.sum(x_mask)
        else:
            attn_log_durations = torch.log(attn_durations + 1e-6) * x_mask
            log_durations = self.duration_predictor(
                x.detach() if self.conf.detach_dp_input else x,
                x_mask,
                g=g.detach() if self.conf.detach_dp_input and g is not None else g,
                lang_emb=lang_emb.detach() if self.conf.detach_dp_input and lang_emb is not None else lang_emb,
            )
            loss_duration = torch.sum((log_durations - attn_log_durations) ** 2, [1, 2]) / torch.sum(x_mask)
        outputs["loss_duration"] = loss_duration
        return outputs, attn

    def upsampling_z(self, z, slice_ids=None, y_lengths=None, y_mask=None):
        spec_segment_size = self.conf.spec_segment_size
        if self.conf.encoder_sample_rate:
            # recompute the slices and spec_segment_size if needed
            slice_ids = slice_ids * int(self.conf.interpolate_factor) if slice_ids is not None else slice_ids
            spec_segment_size = spec_segment_size * int(self.conf.interpolate_factor)
            # interpolate z if needed
            if self.conf.interpolate_z:
                z = torch.nn.functional.interpolate(z, scale_factor=[self.conf.interpolate_factor], mode="linear").squeeze(0)
                # recompute the mask if needed
                if y_lengths is not None and y_mask is not None:
                    y_mask = (
                        sequence_mask(y_lengths * self.conf.interpolate_factor, None).to(y_mask.dtype).unsqueeze(1)
                    )  # [B, 1, T_dec_resampled]

        return z, spec_segment_size, slice_ids, y_mask

# copied and modified from https://github.com/coqui-ai/TTS/blob/096b35f6396d063205318fbaef10f08b9f699832/TTS/tts/models/vits.py
# - 
class P1(Base):
    def __init__(self, conf: DictConfig) -> None:
        super(P1, self).__init__()
        self.conf = conf
        self.inference_noise_scale_dp = conf.inference_noise_scale_dp
        self.length_scale = conf.length_scale
        self.inference_noise_scale = conf.inference_noise_scale
        self.max_inference_len = conf.max_inference_len
        
        # language_embedding
        if self.conf.embedded_language_dim is not None:
            self.use_lang_emb = True
            self.emb_l_ = nn.Embedding(self.conf.num_languages, self.conf.embedded_language_dim)
            torch.nn.init.xavier_uniform_(self.emb_l_.weight)
        else:
            self.use_lang_emb = False
        
        # self.reduced_w2v
        self.w2v_extractor = ReducedWav2Vec2ForPreTraining.from_pretrained(self.conf.w2v)
        self.w2v_extractor.leave_only_1layer()
        # freeze w2v
        for param in self.w2v_extractor.parameters():
            param.requires_grad = False

        # self.speaker_encoder
        self.speaker_encoder = Speaker(self.conf.speaker_encoder.c_in, self.conf.speaker_encoder.c_mid, self.conf.embedded_speaker_dim)
            
         # text_encoder
        self.text_encoder = TextEncoder(
            self.conf.num_chars,
            self.conf.hidden_channels,
            self.conf.hidden_channels,
            self.conf.hidden_channels_ffn_text_encoder,
            self.conf.num_heads_text_encoder,
            self.conf.num_layers_text_encoder,
            self.conf.kernel_size_text_encoder,
            self.conf.dropout_p_text_encoder,
            language_emb_dim=self.conf.embedded_language_dim,
        ) 

        self.posterior_encoder = PosteriorEncoder(
            self.conf.out_channels,
            self.conf.hidden_channels,
            self.conf.hidden_channels,
            kernel_size=self.conf.kernel_size_posterior_encoder,
            dilation_rate=self.conf.dilation_rate_posterior_encoder,
            num_layers=self.conf.num_layers_posterior_encoder,
            cond_channels=self.conf.embedded_speaker_dim,
        )

        self.flow = ResidualCouplingBlocks(
            self.conf.hidden_channels,
            self.conf.hidden_channels,
            kernel_size=self.conf.kernel_size_flow,
            dilation_rate=self.conf.dilation_rate_flow,
            num_layers=self.conf.num_layers_flow,
            cond_channels=self.conf.embedded_speaker_dim,
        )

        self.waveform_decoder = HifiganGenerator(
            self.conf.hidden_channels,
            1,
            self.conf.resblock_type_decoder,
            self.conf.resblock_dilation_sizes_decoder,
            self.conf.resblock_kernel_sizes_decoder,
            self.conf.upsample_kernel_sizes_decoder,
            self.conf.upsample_initial_channel_decoder,
            self.conf.upsample_rates_decoder,
            inference_padding=0,
            cond_channels=self.conf.embedded_speaker_dim,
            conv_pre_weight_norm=False,
            conv_post_weight_norm=False,
            conv_post_bias=False,
        )

    def forward(self, 
        x,
        x_lengths,
        y,
        y_lengths,
        waveform,
        cropped_waveform,
        aux_input={"d_vectors": None, "speaker_ids": None, "language_ids": None}) -> Dict:
        #
        assert aux_input['language_ids'] != None

        outputs = {}
        _, g, lid, _ = self._set_cond_input(aux_input)

        with torch.no_grad():
            w2v_1st_block_feature = self.w2v_extractor.reduced_forward(cropped_waveform, output_hidden_states = True)[0][1] #[B, t, dim]
            w2v_1st_block_feature = w2v_1st_block_feature.permute((0, 2, 1)) # [B, dim, t]

        g = self.speaker_encoder(w2v_1st_block_feature) # [B, emb_dim]
        g = g.unsqueeze(-1) # [B, emb_dim, 1]

        
        # language embedding
        lang_emb = None
        if self.use_lang_emb:
            lang_emb = self.emb_l_(lid).unsqueeze(-1) # [B, emb_dim, 1] 인듯.

        # text encoder
        x, m_p, logs_p, x_mask = self.text_encoder(x, x_lengths, lang_emb=lang_emb)
        # x: [B, text_emb_dim + lang_emb_dim, T], m: [B, out_channels, T], logs: [B, out_channels, T], x_mask : [B,1,T] 

        # posterior encoder
        z, m_q, logs_q, y_mask = self.posterior_encoder(y, y_lengths, g=g) 

        # flow layers
        z_p = self.flow(z, y_mask, g=g)
        
        # duration predictor
        attn = self.forward_mas_without_dp(z_p, m_p, logs_p, x, x_mask, y_mask)
        # expand prior
        m_p = torch.einsum("klmn, kjm -> kjn", [attn, m_p])
        logs_p = torch.einsum("klmn, kjm -> kjn", [attn, logs_p])

        # select a random feature segment for the waveform decoder
        z_slice, slice_ids = rand_segments(z, y_lengths, self.conf.spec_segment_size, let_short_samples=True, pad_short=True)
        
        # interpolate z if needed
        z_slice, spec_segment_size, slice_ids, _ = self.upsampling_z(z_slice, slice_ids=slice_ids) # 내 코드는 upsampling 따로 안함 - encoder_sample_rate 가 None 이라.

        o = self.waveform_decoder(z_slice, g=g)

        wav_seg = segment(
            waveform,
            slice_ids * self.conf.audio.hop_length,
            spec_segment_size * self.conf.audio.hop_length,
            pad_short=True,
        )

        gt_spk_emb, syn_spk_emb = None, None

        outputs.update(
            {
                "model_outputs": o,
                "alignments": attn.squeeze(1),
                "m_p": m_p,
                "logs_p": logs_p,
                "z": z,
                "z_p": z_p,
                "m_q": m_q,
                "logs_q": logs_q,
                "waveform_seg": wav_seg,
                "gt_spk_emb": gt_spk_emb,
                "syn_spk_emb": syn_spk_emb,
                "slice_ids": slice_ids,
            }
        )
        return outputs


# copied and modified from https://github.com/coqui-ai/TTS/blob/096b35f6396d063205318fbaef10f08b9f699832/TTS/tts/models/vits.py
# -
class P2(Base):
    def __init__(self, conf: DictConfig) -> None:
        super(P2, self).__init__()
        self.conf = conf
        self.inference_noise_scale_dp = conf.inference_noise_scale_dp
        self.length_scale = conf.length_scale
        self.inference_noise_scale = conf.inference_noise_scale
        self.max_inference_len = conf.max_inference_len

        if self.conf.embedded_language_dim is not None:
            self.use_lang_emb = True
            self.emb_l_ = nn.Embedding(self.conf.num_languages, self.conf.embedded_language_dim)
            torch.nn.init.xavier_uniform_(self.emb_l_.weight)
        else:
            self.use_lang_emb = False

        # speaker embedding
        self.emb_s_ = nn.Embedding(self.conf.num_speakers, self.conf.embedded_speaker_dim)

        self.text_encoder = TextEncoder(
            self.conf.num_chars,
            self.conf.hidden_channels,
            self.conf.hidden_channels,
            self.conf.hidden_channels_ffn_text_encoder,
            self.conf.num_heads_text_encoder,
            self.conf.num_layers_text_encoder,
            self.conf.kernel_size_text_encoder,
            self.conf.dropout_p_text_encoder,
            language_emb_dim=self.conf.embedded_language_dim,
        ) 

        self.posterior_encoder = PosteriorEncoder(
            self.conf.out_channels,
            self.conf.hidden_channels,
            self.conf.hidden_channels,
            kernel_size=self.conf.kernel_size_posterior_encoder,
            dilation_rate=self.conf.dilation_rate_posterior_encoder,
            num_layers=self.conf.num_layers_posterior_encoder,
            cond_channels=self.conf.embedded_speaker_dim,
        )

        self.flow = ResidualCouplingBlocks(
            self.conf.hidden_channels,
            self.conf.hidden_channels,
            kernel_size=self.conf.kernel_size_flow,
            dilation_rate=self.conf.dilation_rate_flow,
            num_layers=self.conf.num_layers_flow,
            cond_channels=self.conf.embedded_speaker_dim,
        )

        self.waveform_decoder = HifiganGenerator(
            self.conf.hidden_channels,
            1,
            self.conf.resblock_type_decoder,
            self.conf.resblock_dilation_sizes_decoder,
            self.conf.resblock_kernel_sizes_decoder,
            self.conf.upsample_kernel_sizes_decoder,
            self.conf.upsample_initial_channel_decoder,
            self.conf.upsample_rates_decoder,
            inference_padding=0,
            cond_channels=self.conf.embedded_speaker_dim,
            conv_pre_weight_norm=False,
            conv_post_weight_norm=False,
            conv_post_bias=False,
        )

    def forward(self, 
        x,
        x_lengths,
        y,
        y_lengths,
        waveform,
        cropped_waveform,
        aux_input={"d_vectors": None, "speaker_ids": None, "language_ids": None}) -> Dict:
        
        assert aux_input['language_ids'] != None
        assert aux_input['speaker_ids'] != None    
        outputs = {}
        sid, _, lid, _ = self._set_cond_input(aux_input)

        g = self.emb_s_(sid)
        g = g.unsqueeze(-1) # [B, emb_dim, 1]
        
        # language embedding
        lang_emb = None
        if self.use_lang_emb:
            lang_emb = self.emb_l_(lid).unsqueeze(-1) # [B, emb_dim, 1]
        

        # text encoder
        x, m_p, logs_p, x_mask = self.text_encoder(x, x_lengths, lang_emb=lang_emb)
        

        # posterior encoder
        z, m_q, logs_q, y_mask = self.posterior_encoder(y, y_lengths, g=g)

        # flow layers
        z_p = self.flow(z, y_mask, g=g)
        
        attn = self.forward_mas_without_dp(z_p, m_p, logs_p, x, x_mask, y_mask)
    
        # expand prior
        m_p = torch.einsum("klmn, kjm -> kjn", [attn, m_p])
        logs_p = torch.einsum("klmn, kjm -> kjn", [attn, logs_p])

        # select a random feature segment for the waveform decoder
        z_slice, slice_ids = rand_segments(z, y_lengths, self.conf.spec_segment_size, let_short_samples=True, pad_short=True)
        
        # interpolate z if needed
        z_slice, spec_segment_size, slice_ids, _ = self.upsampling_z(z_slice, slice_ids=slice_ids)

        o = self.waveform_decoder(z_slice, g=g)

        wav_seg = segment(
            waveform,
            slice_ids * self.conf.audio.hop_length,
            spec_segment_size * self.conf.audio.hop_length,
            pad_short=True,
        ) 
        
        gt_spk_emb, syn_spk_emb = None, None

        outputs.update(
            {
                "model_outputs": o,
                "alignments": attn.squeeze(1),
                "m_p": m_p,
                "logs_p": logs_p,
                "z": z,
                "z_p": z_p,
                "m_q": m_q,
                "logs_q": logs_q,
                "waveform_seg": wav_seg,
                "gt_spk_emb": gt_spk_emb,
                "syn_spk_emb": syn_spk_emb,
                "slice_ids": slice_ids,
                "loss_duration": None
            }
        )
        return outputs

# copied and modified from https://github.com/coqui-ai/TTS/blob/096b35f6396d063205318fbaef10f08b9f699832/TTS/tts/models/vits.py
# - 
class P3(Base): 
    def __init__(self, conf: DictConfig) -> None:
        super(P3, self).__init__()
        self.conf = conf
        self.inference_noise_scale_dp = conf.inference_noise_scale_dp
        self.length_scale = conf.length_scale
        self.inference_noise_scale = conf.inference_noise_scale
        self.max_inference_len = conf.max_inference_len
        
        self.emb_char = nn.Embedding(self.conf.num_chars, self.conf.hidden_channels, padding_idx=0)
        nn.init.normal_(self.emb_char.weight, 0.0, self.conf.hidden_channels**-0.5)

        
        if self.conf.embedded_language_dim is not None:
            self.use_lang_emb = True
            self.emb_l_ = nn.Embedding(self.conf.num_languages, self.conf.embedded_language_dim)
            torch.nn.init.xavier_uniform_(self.emb_l_.weight)
        else:
            self.use_lang_emb = False

        # speaker embedding
        self.emb_s_ = nn.Embedding(self.conf.num_speakers, self.conf.embedded_speaker_dim)


        # text_encoder
        self.text_encoder = TextEncoderwithoutEmbMultilingual(
            self.conf.hidden_channels,
            self.conf.hidden_channels,
            self.conf.hidden_channels_ffn_text_encoder,
            self.conf.num_heads_text_encoder,
            self.conf.num_layers_text_encoder,
            self.conf.kernel_size_text_encoder,
            self.conf.dropout_p_text_encoder,
            language_emb_dim = self.conf.embedded_language_dim, # null로 할 수도 있고 선택사항
            num_languages = self.conf.num_languages,
            heads_share = self.conf.heads_share
        )
        self.num_languages = self.conf.num_languages

        self.posterior_encoder = PosteriorEncoder(
            self.conf.out_channels,
            self.conf.hidden_channels,
            hidden_channels=self.conf.pos_enc_hidden_channels,
            kernel_size=self.conf.kernel_size_posterior_encoder,
            dilation_rate=self.conf.dilation_rate_posterior_encoder,
            num_layers=self.conf.num_layers_posterior_encoder,
            cond_channels=self.conf.embedded_speaker_dim,
        )

        self.flow = ResidualCouplingBlocks(
            self.conf.hidden_channels,
            self.conf.hidden_channels,
            kernel_size=self.conf.kernel_size_flow,
            dilation_rate=self.conf.dilation_rate_flow,
            num_layers=self.conf.num_layers_flow,
            cond_channels=self.conf.embedded_speaker_dim,
        )



        self.waveform_decoder = HifiganGenerator(
            self.conf.hidden_channels,
            1,
            self.conf.resblock_type_decoder,
            self.conf.resblock_dilation_sizes_decoder,
            self.conf.resblock_kernel_sizes_decoder,
            self.conf.upsample_kernel_sizes_decoder,
            self.conf.upsample_initial_channel_decoder,
            self.conf.upsample_rates_decoder,
            inference_padding=0,
            cond_channels=self.conf.embedded_speaker_dim,
            conv_pre_weight_norm=False,
            conv_post_weight_norm=False,
            conv_post_bias=False,
        )

    def forward(self, 
        x,
        x_lengths,
        y,
        y_lengths,
        waveform,
        cropped_waveform,
        aux_input={"d_vectors": None, "speaker_ids": None, "language_ids": None}) -> Dict:
        """
        Forward pass of the model.
        Args:
            x (torch.tensor): Batch of input character sequence IDs.
            x_lengths (torch.tensor): Batch of input character sequence lengths. 
            y (torch.tensor): Batch of input spectrograms.
            y_lengths (torch.tensor): Batch of input spectrogram lengths.
            waveform (torch.tensor): Batch of ground truth waveforms per sample.
            cropped_waveform (torch.tensor): Batch of 'cropped' ground truth waveforms for calculating 'speaker embedding'

            aux_input (dict, optional): Auxiliary inputs for multi-speaker and multi-lingual training.
                Defaults to {"d_vectors": None, "speaker_ids": None, "language_ids": None}.
        Returns:
            Dict: model outputs keyed by the output name.
        Shapes:
            - x: :math:`[B, T_seq]`
            - x_lengths: :math:`[B]`
            - y: :math:`[B, C, T_spec]`
            - y_lengths: :math:`[B]`
            - waveform: :math:`[B, 1, T_wav]`
            - cropped_waveform :math: '[B, t_wav]' # t_wav < T_wav
            - d_vectors: :math:`[B, C, 1]`
            - speaker_ids: :math:`[B]`
            - language_ids: :math:`[B]`
        Return Shapes:
            - model_outputs: :math:`[B, 1, T_wav]`
            - alignments: :math:`[B, T_seq, T_dec]`
            - z: :math:`[B, C, T_dec]`
            - z_p: :math:`[B, C, T_dec]`
            - m_p: :math:`[B, C, T_dec]`
            - logs_p: :math:`[B, C, T_dec]`
            - m_q: :math:`[B, C, T_dec]`
            - logs_q: :math:`[B, C, T_dec]`
            - waveform_seg: :math:`[B, 1, spec_seg_size * hop_length]`
            - gt_spk_emb: :math:`[B, 1, speaker_encoder.proj_dim]`
            - syn_spk_emb: :math:`[B, 1, speaker_encoder.proj_dim]`
        """
        #
        assert aux_input['language_ids'] != None
        

        outputs = {}
        sid, g, lid, _ = self._set_cond_input(aux_input)
        assert ((lid.view(-1, self.num_languages) - lid.view(-1, self.num_languages).type(torch.float32).mean(dim=0)) != 0).sum() == 0
        # language 순서대로 batch가 구성되어 있음을 보장

        g = self.emb_s_(sid)
        g = g.unsqueeze(-1) # [B, emb_dim, 1]

        
        # language embedding
        lang_emb = None
        if self.use_lang_emb:
            lang_emb = self.emb_l_(lid).unsqueeze(-1) # [B, emb_dim, 1] 인듯.


        # text encoder
        assert x.shape[0] == x_lengths.shape[0]
        x = self.emb_char(x) * math.sqrt(self.conf.hidden_channels)  # [b, t, h]
        x, m_p, logs_p, x_mask = self.text_encoder(x, x_lengths, lang_emb=lang_emb)
        # x: [B, text_emb_dim + lang_emb_dim, T], m: [B, out_channels, T], logs: [B, out_channels, T], x_mask : [B,1,T] 

        # posterior encoder
        z, m_q, logs_q, y_mask = self.posterior_encoder(y, y_lengths, g=g) # y의 시간축과 z의 시간축이 같은지 확인 필요 - 같네 - 아래에서 z랑 y_lengths 를 같이 쓴거만으로도 유추할 수 있기는 하겠다 mo

        # flow layers
        z_p = self.flow(z, y_mask, g=g)
        
        # duration predictor - duration predictor 없게끔 수정해볼 것
        # outputs, attn = self.forward_mas(outputs, z_p, m_p, logs_p, x, x_mask, y_mask, g=g, lang_emb=lang_emb)
        attn = self.forward_mas_without_dp(z_p, m_p, logs_p, x, x_mask, y_mask)

        # expand prior
        m_p = torch.einsum("klmn, kjm -> kjn", [attn, m_p])
        logs_p = torch.einsum("klmn, kjm -> kjn", [attn, logs_p])

        # select a random feature segment for the waveform decoder
        z_slice, slice_ids = rand_segments(z, y_lengths, self.conf.spec_segment_size, let_short_samples=True, pad_short=True)
        # z의 segment 부분과 slice_ids는 start_ix 들
        # interpolate z if needed
        z_slice, spec_segment_size, slice_ids, _ = self.upsampling_z(z_slice, slice_ids=slice_ids) # 내 코드는 upsampling 따로 안함 - encoder_sample_rate 가 None 이라.

        o = self.waveform_decoder(z_slice, g=g)

        wav_seg = segment(
            waveform,
            slice_ids * self.conf.audio.hop_length,
            spec_segment_size * self.conf.audio.hop_length,
            pad_short=True,
        ) # fft_size + (spec_segment_size-1) * self.conf.audio.hop_length 이 더 정확한거 아닌가? - mo -> 근데 vits 공식 구현도 여기처럼 되어 있어서 일단. - 어차피 약간 잘리는거고, 다 똑같이 공평하게 잘리는거라 별 상관없을거 같기는 함. mo
        # 아 근데 이렇게 해야 o와 wav_seg의 shape가 똑같아지네. waveform_decoder의 upsampling 방식 때문에 어쩔 수 없이라도 이렇게 해야 하는거 같기도?? mo

        gt_spk_emb, syn_spk_emb = None, None

        outputs.update(
            {
                "model_outputs": o,
                "alignments": attn.squeeze(1),
                "m_p": m_p,
                "logs_p": logs_p,
                "z": z,
                "z_p": z_p,
                "m_q": m_q,
                "logs_q": logs_q,
                "waveform_seg": wav_seg,
                "gt_spk_emb": gt_spk_emb,
                "syn_spk_emb": syn_spk_emb,
                "slice_ids": slice_ids,
            }
        )
        return outputs

        
# copied and modified from https://github.com/coqui-ai/TTS/blob/096b35f6396d063205318fbaef10f08b9f699832/TTS/tts/models/vits.py
# -
class T1(Base):
    def __init__(self, conf: DictConfig) -> None:
        super(T1, self).__init__()
        self.conf = conf
        self.inference_noise_scale_dp = conf.inference_noise_scale_dp
        self.length_scale = conf.length_scale
        self.inference_noise_scale = conf.inference_noise_scale
        self.max_inference_len = conf.max_inference_len

        # language_embedding
        self.emb_l = nn.Embedding(self.conf.num_languages, self.conf.embedded_language_dim)
        torch.nn.init.xavier_uniform_(self.emb_l.weight)

        # self.reduced_w2v
        self.w2v_extractor = ReducedWav2Vec2ForPreTraining.from_pretrained(self.conf.w2v)
        self.w2v_extractor.leave_only_1layer()
        # freeze w2v
        for param in self.w2v_extractor.parameters():
            param.requires_grad = False

        # self.speaker_encoder - fine tuning 
        self.speaker_encoder = Speaker(self.conf.speaker_encoder.c_in, self.conf.speaker_encoder.c_mid, self.conf.embedded_speaker_dim)
        
        # text_encoder - from scratch - change name
        self.text_encoder_ = TextEncoder(
            self.conf.num_chars,
            self.conf.hidden_channels,
            self.conf.hidden_channels,
            self.conf.hidden_channels_ffn_text_encoder,
            self.conf.num_heads_text_encoder,
            self.conf.num_layers_text_encoder,
            self.conf.kernel_size_text_encoder,
            self.conf.dropout_p_text_encoder,
            language_emb_dim=self.conf.embedded_language_dim,
        ) 

        # fine tuning
        self.posterior_encoder = PosteriorEncoder(
            self.conf.out_channels,
            self.conf.hidden_channels,
            self.conf.hidden_channels,
            kernel_size=self.conf.kernel_size_posterior_encoder,
            dilation_rate=self.conf.dilation_rate_posterior_encoder,
            num_layers=self.conf.num_layers_posterior_encoder,
            cond_channels=self.conf.embedded_speaker_dim,
        )

        # fine tuning
        self.flow = ResidualCouplingBlocks(
            self.conf.hidden_channels,
            self.conf.hidden_channels,
            kernel_size=self.conf.kernel_size_flow,
            dilation_rate=self.conf.dilation_rate_flow,
            num_layers=self.conf.num_layers_flow,
            cond_channels=self.conf.embedded_speaker_dim,
        )

        if self.conf.use_sdp:
            self.duration_predictor = StochasticDurationPredictor(
                self.conf.hidden_channels,
                192,
                3,
                self.conf.dropout_p_duration_predictor,
                4,
                cond_channels=self.conf.embedded_speaker_dim if self.conf.condition_dp_on_speaker else 0,
                language_emb_dim=self.conf.embedded_language_dim,
            )
        else:
            self.duration_predictor = DurationPredictor(
                self.conf.hidden_channels,
                256,
                3,
                self.conf.dropout_p_duration_predictor,
                cond_channels=self.embedded_speaker_dim,
                language_emb_dim=self.embedded_language_dim,
            )

        self.waveform_decoder = HifiganGenerator(
            self.conf.hidden_channels,
            1,
            self.conf.resblock_type_decoder,
            self.conf.resblock_dilation_sizes_decoder,
            self.conf.resblock_kernel_sizes_decoder,
            self.conf.upsample_kernel_sizes_decoder,
            self.conf.upsample_initial_channel_decoder,
            self.conf.upsample_rates_decoder,
            inference_padding=0,
            cond_channels=self.conf.embedded_speaker_dim,
            conv_pre_weight_norm=False,
            conv_post_weight_norm=False,
            conv_post_bias=False,
        )

    @torch.no_grad()
    def inference(self, x, cropped_waveform, aux_input={"x_lengths": None, "d_vectors": None, "speaker_ids": None, "language_ids": None, "durations": None}):
        # x : [1, text_length]
        English_id = 7
        aux_input['language_ids'] = torch.LongTensor([English_id] * x.size(0)).to(x.device)

        _, _, lid, _ = self._set_cond_input(aux_input)
        x_lengths = self._set_x_lengths(x, aux_input)
                
        with torch.no_grad():
            w2v_1st_block_feature = self.w2v_extractor.reduced_forward(cropped_waveform, output_hidden_states = True)[0][1] #[B, t, dim]
            w2v_1st_block_feature = w2v_1st_block_feature.permute((0, 2, 1)) # [B, dim, t]

        g = self.speaker_encoder(w2v_1st_block_feature) # [B, emb_dim]
        g = g.unsqueeze(-1) # [B, emb_dim, 1]
        
        # language embedding
        lang_emb = None
        if lid is not None:
            lang_emb = self.emb_l(lid).unsqueeze(-1)
        
        x, m_p, logs_p, x_mask = self.text_encoder_(x, x_lengths, lang_emb=lang_emb)
        durations = None
        
        if durations == None:
            if self.conf.use_sdp:
                logw = self.duration_predictor(
                    x,
                    x_mask,
                    g=g if self.conf.condition_dp_on_speaker else None,
                    reverse=True,
                    noise_scale=self.inference_noise_scale_dp,
                    lang_emb=lang_emb,
                )
            else:
                logw = self.duration_predictor(
                    x, x_mask, g=g if self.conf.condition_dp_on_speaker else None, lang_emb=lang_emb
                )
            w = torch.exp(logw) * x_mask * self.length_scale
        else:
            assert durations.shape[-1] == x.shape[-1]
            w = durations.unsqueeze(0)

        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = sequence_mask(y_lengths, None).to(x_mask.dtype).unsqueeze(1)  # [B, 1, T_dec]

        attn_mask = x_mask * y_mask.transpose(1, 2)  # [B, 1, T_enc] * [B, T_dec, 1]
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1).transpose(1, 2))

        m_p = torch.matmul(attn.transpose(1, 2), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.transpose(1, 2), logs_p.transpose(1, 2)).transpose(1, 2)

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * self.inference_noise_scale
        z = self.flow(z_p, y_mask, g=g, reverse=True)

        z, _, _, y_mask = self.upsampling_z(z, y_lengths=y_lengths, y_mask=y_mask)

        o = self.waveform_decoder((z * y_mask)[:, :, : self.max_inference_len], g=g)

        outputs = {
            "model_outputs": o,
            "alignments": attn.squeeze(1),
            "durations": w_ceil,
            "z": z,
            "z_p": z_p,
            "m_p": m_p,
            "logs_p": logs_p,
            "y_mask": y_mask,
        }
        return outputs

    def forward(self, 
        x,
        x_lengths,
        y,
        y_lengths,
        waveform,
        cropped_waveform,
        aux_input={"d_vectors": None, "speaker_ids": None, "language_ids": None}) -> Dict:

        assert aux_input['language_ids'] != None

        outputs = {}
        _, _, lid, _ = self._set_cond_input(aux_input)

        with torch.no_grad():
            w2v_1st_block_feature = self.w2v_extractor.reduced_forward(cropped_waveform, output_hidden_states = True)[0][1] #[B, t, dim]
            w2v_1st_block_feature = w2v_1st_block_feature.permute((0, 2, 1)) # [B, dim, t]

        g = self.speaker_encoder(w2v_1st_block_feature) # [B, emb_dim]
        g = g.unsqueeze(-1) # [B, emb_dim, 1]

        
        # language embedding
        lang_emb = None
        if lid is not None:
            lang_emb = self.emb_l(lid).unsqueeze(-1) 

        # text encoder
        x, m_p, logs_p, x_mask = self.text_encoder_(x, x_lengths, lang_emb=lang_emb)
        # x: [B, text_emb_dim + lang_emb_dim, T], m: [B, out_channels, T], logs: [B, out_channels, T], x_mask : [B,1,T] 

        # posterior encoder
        z, m_q, logs_q, y_mask = self.posterior_encoder(y, y_lengths, g=g)

        # flow layers
        z_p = self.flow(z, y_mask, g=g)
        
        # duration predictor
        outputs, attn = self.forward_mas(outputs, z_p, m_p, logs_p, x, x_mask, y_mask, g=g, lang_emb=lang_emb)
        
        # expand prior
        m_p = torch.einsum("klmn, kjm -> kjn", [attn, m_p])
        logs_p = torch.einsum("klmn, kjm -> kjn", [attn, logs_p])

        # select a random feature segment for the waveform decoder
        z_slice, slice_ids = rand_segments(z, y_lengths, self.conf.spec_segment_size, let_short_samples=True, pad_short=True)
        
        # interpolate z if needed
        z_slice, spec_segment_size, slice_ids, _ = self.upsampling_z(z_slice, slice_ids=slice_ids) 

        o = self.waveform_decoder(z_slice, g=g)

        wav_seg = segment(
            waveform,
            slice_ids * self.conf.audio.hop_length,
            spec_segment_size * self.conf.audio.hop_length,
            pad_short=True,
        ) 
        
        gt_spk_emb, syn_spk_emb = None, None

        outputs.update(
            {
                "model_outputs": o,
                "alignments": attn.squeeze(1),
                "m_p": m_p,
                "logs_p": logs_p,
                "z": z,
                "z_p": z_p,
                "m_q": m_q,
                "logs_q": logs_q,
                "waveform_seg": wav_seg,
                "gt_spk_emb": gt_spk_emb,
                "syn_spk_emb": syn_spk_emb,
                "slice_ids": slice_ids,
            }
        )
        return outputs


class T1F1(T1):
    def __init__(self, conf: DictConfig) -> None:
        super(T1F1, self).__init__(conf)
        
        # freeze speaker_encoder
        for param in self.speaker_encoder.parameters():
            param.requires_grad = False

        # freeze posterior_encoder
        for param in self.posterior_encoder.parameters():
            param.requires_grad = False

        # freeze waveform_decoder
        for param in self.waveform_decoder.parameters():
            param.requires_grad = False


class T1F2(T1):
    def __init__(self, conf: DictConfig) -> None:
        super(T1F2, self).__init__(conf)
        

# copied and modified from https://github.com/coqui-ai/TTS/blob/096b35f6396d063205318fbaef10f08b9f699832/TTS/tts/models/vits.py
# -
class T2(Base):
    def __init__(self, conf: DictConfig) -> None:
        super(T2, self).__init__()
        self.conf = conf
        self.inference_noise_scale_dp = conf.inference_noise_scale_dp
        self.length_scale = conf.length_scale
        self.inference_noise_scale = conf.inference_noise_scale
        self.max_inference_len = conf.max_inference_len

        # language_embedding
        self.emb_l = nn.Embedding(self.conf.num_languages, self.conf.embedded_language_dim)
        torch.nn.init.xavier_uniform_(self.emb_l.weight)

        # speaker embedding - from scratch
        self.emb_s = nn.Embedding(self.conf.num_speakers, self.conf.embedded_speaker_dim)

        # text_encoder - from scratch - change name
        self.text_encoder_ = TextEncoder(
            self.conf.num_chars,
            self.conf.hidden_channels,
            self.conf.hidden_channels,
            self.conf.hidden_channels_ffn_text_encoder,
            self.conf.num_heads_text_encoder,
            self.conf.num_layers_text_encoder,
            self.conf.kernel_size_text_encoder,
            self.conf.dropout_p_text_encoder,
            language_emb_dim=self.conf.embedded_language_dim,
        ) 

        # posterior_encoder
        self.posterior_encoder = PosteriorEncoder(
            self.conf.out_channels,
            self.conf.hidden_channels,
            self.conf.hidden_channels,
            kernel_size=self.conf.kernel_size_posterior_encoder,
            dilation_rate=self.conf.dilation_rate_posterior_encoder,
            num_layers=self.conf.num_layers_posterior_encoder,
            cond_channels=self.conf.embedded_speaker_dim,
        )

        # flow - fine tuning
        self.flow = ResidualCouplingBlocks(
            self.conf.hidden_channels,
            self.conf.hidden_channels,
            kernel_size=self.conf.kernel_size_flow,
            dilation_rate=self.conf.dilation_rate_flow,
            num_layers=self.conf.num_layers_flow,
            cond_channels=self.conf.embedded_speaker_dim,
        )

        # from_scratch
        if self.conf.use_sdp:
            self.duration_predictor = StochasticDurationPredictor(
                self.conf.hidden_channels,
                192,
                3,
                self.conf.dropout_p_duration_predictor,
                4,
                cond_channels=self.conf.embedded_speaker_dim if self.conf.condition_dp_on_speaker else 0,
                language_emb_dim=self.conf.embedded_language_dim,
            )
        else:
            self.duration_predictor = DurationPredictor(
                self.conf.hidden_channels,
                256,
                3,
                self.conf.dropout_p_duration_predictor,
                cond_channels=self.embedded_speaker_dim,
                language_emb_dim=self.embedded_language_dim,
            )

        # generator
        self.waveform_decoder = HifiganGenerator(
            self.conf.hidden_channels,
            1,
            self.conf.resblock_type_decoder,
            self.conf.resblock_dilation_sizes_decoder,
            self.conf.resblock_kernel_sizes_decoder,
            self.conf.upsample_kernel_sizes_decoder,
            self.conf.upsample_initial_channel_decoder,
            self.conf.upsample_rates_decoder,
            inference_padding=0,
            cond_channels=self.conf.embedded_speaker_dim,
            conv_pre_weight_norm=False,
            conv_post_weight_norm=False,
            conv_post_bias=False,
        )

    @torch.no_grad()
    def inference(self, x, aux_input={"x_lengths": None, "d_vectors": None, "speaker_ids": None, "language_ids": None, "durations": None}):
        # x : [1, text_length]
        x_lengths = self._set_x_lengths(x, aux_input)
        durations = None
        sid, _, lid, _ = self._set_cond_input(aux_input)
        
        # language embedding
        lang_emb = self.emb_l(lid).unsqueeze(-1)
        
        # speaker embedding
        g = self.emb_s(sid)
        g = g.unsqueeze(-1)
        
        x, m_p, logs_p, x_mask = self.text_encoder_(x, x_lengths, lang_emb=lang_emb)

        if durations == None:
            if self.conf.use_sdp:
                logw = self.duration_predictor(
                    x,
                    x_mask,
                    g=g if self.conf.condition_dp_on_speaker else None,
                    reverse=True,
                    noise_scale=self.inference_noise_scale_dp,
                    lang_emb=lang_emb,
                )
            else:
                logw = self.duration_predictor(
                    x, x_mask, g=g if self.conf.condition_dp_on_speaker else None, lang_emb=lang_emb
                )
            w = torch.exp(logw) * x_mask * self.length_scale
        else:
            assert durations.shape[-1] == x.shape[-1]
            w = durations.unsqueeze(0)

        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = sequence_mask(y_lengths, None).to(x_mask.dtype).unsqueeze(1)  # [B, 1, T_dec]

        attn_mask = x_mask * y_mask.transpose(1, 2)  # [B, 1, T_enc] * [B, T_dec, 1]
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1).transpose(1, 2))

        m_p = torch.matmul(attn.transpose(1, 2), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.transpose(1, 2), logs_p.transpose(1, 2)).transpose(1, 2)

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * self.inference_noise_scale
        z = self.flow(z_p, y_mask, g=g, reverse=True)

        z, _, _, y_mask = self.upsampling_z(z, y_lengths=y_lengths, y_mask=y_mask)

        o = self.waveform_decoder((z * y_mask)[:, :, : self.max_inference_len], g=g)

        outputs = {
            "model_outputs": o,
            "alignments": attn.squeeze(1),
            "durations": w_ceil,
            "z": z,
            "z_p": z_p,
            "m_p": m_p,
            "logs_p": logs_p,
            "y_mask": y_mask,
        }
        return outputs

    def forward(self, 
        x,
        x_lengths,
        y,
        y_lengths,
        waveform,
        cropped_waveform,
        aux_input={"d_vectors": None, "speaker_ids": None, "language_ids": None}) -> Dict:

        assert aux_input['language_ids'] != None
        assert aux_input['speaker_ids'] != None

        outputs = {}
        sid, _, lid, _ = self._set_cond_input(aux_input)

        g = self.emb_s(sid)
        g = g.unsqueeze(-1) # [B, emb_dim, 1]

        
        # language embedding
        lang_emb = self.emb_l(lid).unsqueeze(-1) # [B, emb_dim, 1]

        # text encoder
        x, m_p, logs_p, x_mask = self.text_encoder_(x, x_lengths, lang_emb=lang_emb)
        # x: [B, text_emb_dim + lang_emb_dim, T], m: [B, out_channels, T], logs: [B, out_channels, T], x_mask : [B,1,T] 

        # posterior encoder
        z, m_q, logs_q, y_mask = self.posterior_encoder(y, y_lengths, g=g) 

        # flow layers
        z_p = self.flow(z, y_mask, g=g)
        
        # duration predictor
        if self.conf.condition_dp_on_speaker:
            g_dp = g
        else:
            g_dp = None

        # duration predictor
        outputs, attn = self.forward_mas(outputs, z_p, m_p, logs_p, x, x_mask, y_mask, g=g_dp, lang_emb=lang_emb)
        
        # expand prior
        m_p = torch.einsum("klmn, kjm -> kjn", [attn, m_p])
        logs_p = torch.einsum("klmn, kjm -> kjn", [attn, logs_p])

        # select a random feature segment for the waveform decoder
        z_slice, slice_ids = rand_segments(z, y_lengths, self.conf.spec_segment_size, let_short_samples=True, pad_short=True)
        
        # interpolate z if needed
        z_slice, spec_segment_size, slice_ids, _ = self.upsampling_z(z_slice, slice_ids=slice_ids) 

        o = self.waveform_decoder(z_slice, g=g)

        wav_seg = segment(
            waveform,
            slice_ids * self.conf.audio.hop_length,
            spec_segment_size * self.conf.audio.hop_length,
            pad_short=True,
        ) 

        gt_spk_emb, syn_spk_emb = None, None

        outputs.update(
            {
                "model_outputs": o,
                "alignments": attn.squeeze(1),
                "m_p": m_p,
                "logs_p": logs_p,
                "z": z,
                "z_p": z_p,
                "m_q": m_q,
                "logs_q": logs_q,
                "waveform_seg": wav_seg,
                "gt_spk_emb": gt_spk_emb,
                "syn_spk_emb": syn_spk_emb,
                "slice_ids": slice_ids,
            }
        )
        return outputs
    
class T2F1(T2):
    def __init__(self, conf: DictConfig):
        super(T2F1, self).__init__(conf)

        # freeze posterior_encoder
        for param in self.posterior_encoder.parameters():
            param.requires_grad = False

        # freeze waveform_decoder
        for param in self.waveform_decoder.parameters():
            param.requires_grad = False

class T2F2(T2):
    def __init__(self, conf: DictConfig):
        super(T2F2, self).__init__(conf)