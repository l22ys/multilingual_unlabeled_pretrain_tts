import torch
from omegaconf import DictConfig
from torch.nn.utils.rnn import pad_sequence

class DataCollator:
    def __init__(self, conf: DictConfig):
        pass

    def __call__(self,batch):
        tokens = []
        token_lengths = []
        specs = []
        spec_lengths = []
        waveform = []
        cropped_waveform = []
        mels = []
        language_ids = []

        for item in batch:
            token, token_length, wav, cropped_wav, spec, spec_length, mel, aux_input = item
            spec = torch.transpose(spec.squeeze(0),0,1) # [t, spec_dim]
            wav = wav.squeeze() # [L]
            mel = torch.transpose(mel.squeeze(0),0,1) # [t, mel_dim]

            tokens.append(token)
            token_lengths.append(token_length)
            specs.append(spec)
            spec_lengths.append(spec_length)
            waveform.append(wav)
            cropped_waveform.append(cropped_wav)
            mels.append(mel)
            language_ids.append(aux_input['language_id'])

        tokens = pad_sequence(tokens, batch_first= True) # [B, T]
        token_lengths = torch.LongTensor(token_lengths) # [B]
        specs = pad_sequence(specs, batch_first=True) # [B,t,spec_dim]
        specs = torch.transpose(specs, 1,2) # [B,spec_dim,t]

        spec_lengths = torch.LongTensor(spec_lengths) # [B]
        waveform = pad_sequence(waveform, batch_first=True) # [B,L]
        waveform = waveform.unsqueeze(1) #[B,1,L]

        cropped_waveform = torch.stack(cropped_waveform, dim=0) #[B, crop_len]
        mels = pad_sequence(mels, batch_first=True) # [B, t, mel_dim]
        mels = torch.transpose(mels, 1,2) # [B, mel_dim, t]
        language_ids = torch.LongTensor(language_ids) #[B]

        result = {}
        result['tokens'] = tokens
        result['token_lengths'] = token_lengths
        result['specs'] = specs
        result['spec_lengths'] = spec_lengths
        result['waveform'] = waveform
        result['cropped_waveform'] = cropped_waveform
        result['mel'] = mels
        result['aux_input'] = {"d_vectors": None, "speaker_ids": None, "language_ids": language_ids}
        

        return result

class DataCollator_withSID:
    def __init__(self, conf: DictConfig):
        pass


    def __call__(self,batch):
        tokens = []
        token_lengths = []
        specs = []
        spec_lengths = []
        waveform = []
        cropped_waveform = []
        mels = []
        language_ids = []
        speaker_ids = []

        for item in batch:
            token, token_length, wav, cropped_wav, spec, spec_length, mel, aux_input = item
            spec = torch.transpose(spec.squeeze(0),0,1) # [t, spec_dim]
            wav = wav.squeeze() # [L]
            mel = torch.transpose(mel.squeeze(0),0,1) # [t, mel_dim]

            tokens.append(token)
            token_lengths.append(token_length)
            specs.append(spec)
            spec_lengths.append(spec_length)
            waveform.append(wav)
            cropped_waveform.append(cropped_wav)
            mels.append(mel)
            language_ids.append(aux_input['language_id'])
            speaker_ids.append(aux_input['speaker_id'])

        tokens = pad_sequence(tokens, batch_first= True) # [B, T]
        token_lengths = torch.LongTensor(token_lengths) # [B]
        specs = pad_sequence(specs, batch_first=True) # [B,t,spec_dim]
        specs = torch.transpose(specs, 1,2) # [B,spec_dim,t]

        spec_lengths = torch.LongTensor(spec_lengths) # [B]
        waveform = pad_sequence(waveform, batch_first=True) # [B,L]
        waveform = waveform.unsqueeze(1) #[B,1,L]

        cropped_waveform = torch.stack(cropped_waveform, dim=0) #[B, crop_len]
        mels = pad_sequence(mels, batch_first=True) # [B, t, mel_dim]
        mels = torch.transpose(mels, 1,2) # [B, mel_dim, t]
        language_ids = torch.LongTensor(language_ids) #[B]
        speaker_ids = torch.LongTensor(speaker_ids) #[B]

        result = {}
        result['tokens'] = tokens
        result['token_lengths'] = token_lengths
        result['specs'] = specs
        result['spec_lengths'] = spec_lengths
        result['waveform'] = waveform
        result['cropped_waveform'] = cropped_waveform
        result['mel'] = mels
        result['aux_input'] = {"d_vectors": None, "speaker_ids": speaker_ids, "language_ids": language_ids}
        

        return result