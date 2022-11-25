from omegaconf import DictConfig

from torch.utils.data import Dataset
from tqdm import tqdm
from models_.vits.mel import spec_to_mel, wav_to_spec
import numpy as np
import torch
import soundfile as sf
# from utils_.text import cleaners
from utils_.text import cleaned_text_to_sequence

English_id = 7

class Libritts_ZSML(Dataset):
    def __init__(self, conf: DictConfig) -> None:
        super(Libritts_ZSML, self).__init__()
        self.conf = conf
        self.datas, self.lengths = self.get_datas(conf.data_path)

    def get_datas(self, data_path):
        # data_path : file_path of 'data_info_added.txt'
        datas = []
        lengths = []
        with open(data_path, 'r') as f:
            raw_datas = f.readlines()

        for line in tqdm(raw_datas):
            wav_path, text , leng, _ = line.strip().split('|')
            leng = float(leng)
            lang_id = English_id

            text = cleaned_text_to_sequence(text) # list of int
            text = np.array(text)
            text = text.astype(np.int64)
            datas.append( (wav_path, lang_id, text))

            lengths.append(leng)

        return datas, lengths

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, idx):
        wav_path, lang_id, text = self.datas[idx]

        token = torch.from_numpy(text) # [T]
        token_length = len(text)

        wav, sr = sf.read(wav_path)
        assert sr == 16e3
        wav = wav.astype(np.float32)

        # zero mean - unit variance normalize
        if len(wav) < self.conf.crop_size:            
            cropped_wav = np.concatenate([wav, np.zeros(self.conf.crop_size - len(wav)).astype(np.float32)])
            cropped_wav = (cropped_wav - cropped_wav.mean()) / np.sqrt(cropped_wav.var() + 1e-7)
        else:
            start_ix = np.random.randint( len(wav) - self.conf.crop_size + 1)
            cropped_wav = wav[start_ix: start_ix+self.conf.crop_size]
            cropped_wav = (cropped_wav - cropped_wav.mean()) / np.sqrt(cropped_wav.var() + 1e-7)
            
        cropped_wav = torch.from_numpy(cropped_wav) #[crop_len]

        wav = (wav - wav.mean()) / np.sqrt(wav.var() + 1e-7)
        wav = torch.from_numpy(wav)

        wav = wav.unsqueeze(0)
        wav = wav.unsqueeze(0) # [1,1,T]

        spec = wav_to_spec(y=wav, n_fft=self.conf.fft_size, hop_length=self.conf.hop_length,
                            win_length=self.conf.win_length, center=False) # [1,spec_dim,t]

        spec_length = spec.size(-1)

        mel = spec_to_mel(spec, n_fft=self.conf.fft_size, num_mels= self.conf.num_mels, 
                        sample_rate=self.conf.sample_rate, fmin = self.conf.mel_fmin, fmax = self.conf.mel_fmax) # [1, mel_dim, t]
       

        aux_input = {"d_vector": None, "speaker_id": None, "language_id": lang_id}

        return token, token_length, wav, cropped_wav, spec, spec_length, mel, aux_input


class Libritts_SID(Dataset):
    def __init__(self, conf: DictConfig) -> None:
        super(Libritts_SID, self).__init__()
        self.conf = conf
        self.datas, self.lengths, self.speaker_dict = self.get_datas(conf.data_path)

    def get_datas(self, data_path):
        # data_path : file_path of 'data_info_added.txt'
        datas = []
        lengths = []
        speakers = []
        with open(data_path, 'r') as f:
            raw_datas = f.readlines()
        
        for line in tqdm(raw_datas):
            wav_path, text , leng, sid = line.strip().split('|')
            leng = float(leng)
            lang_id = English_id
            speakers.append(sid)
            
            text = cleaned_text_to_sequence(text) # list of int
            text = np.array(text)
            text = text.astype(np.int64)
            datas.append( (wav_path, lang_id, text, sid))

            lengths.append(leng)

        speakers = list(set(speakers))
        speaker_dict = {}
        for i in range(len(speakers)):
            speaker_dict[speakers[i]] = i

        return datas, lengths, speaker_dict

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, idx):
        wav_path, lang_id, text, sid = self.datas[idx]
        sid = self.speaker_dict[sid]        

        token = torch.from_numpy(text) # [T]
        token_length = len(text)

        wav, sr = sf.read(wav_path)
        assert sr == 16e3
        wav = wav.astype(np.float32)

        # zero mean - unit variance normalize
        if len(wav) < self.conf.crop_size:
            # cropped_wav = F.pad(wav, (0, self.conf.crop_size - len(wav)), value= 0)
            cropped_wav = np.concatenate([wav, np.zeros(self.conf.crop_size - len(wav)).astype(np.float32)])
            cropped_wav = (cropped_wav - cropped_wav.mean()) / np.sqrt(cropped_wav.var() + 1e-7)
        else:
            start_ix = np.random.randint( len(wav) - self.conf.crop_size + 1)
            cropped_wav = wav[start_ix: start_ix+self.conf.crop_size]
            cropped_wav = (cropped_wav - cropped_wav.mean()) / np.sqrt(cropped_wav.var() + 1e-7)
            
        # cropped_wav = cropped_wav.astype(np.float32)
        cropped_wav = torch.from_numpy(cropped_wav) #[crop_len]

        wav = (wav - wav.mean()) / np.sqrt(wav.var() + 1e-7)
        wav = torch.from_numpy(wav)

        wav = wav.unsqueeze(0)
        wav = wav.unsqueeze(0) # [1,1,T] - cpu에서 계산하는 것보다 gpu에서 계산하는게 빠르지만.. 우선 편의성을 위해

        spec = wav_to_spec(y=wav, n_fft=self.conf.fft_size, hop_length=self.conf.hop_length,
                            win_length=self.conf.win_length, center=False) # [1,spec_dim,t]

        spec_length = spec.size(-1)

        mel = spec_to_mel(spec, n_fft=self.conf.fft_size, num_mels= self.conf.num_mels, 
                        sample_rate=self.conf.sample_rate, fmin = self.conf.mel_fmin, fmax = self.conf.mel_fmax) # [1, mel_dim, t]
        
        aux_input = {"d_vector": None, "speaker_id": sid, "language_id": lang_id}

        return token, token_length, wav, cropped_wav, spec, spec_length, mel, aux_input