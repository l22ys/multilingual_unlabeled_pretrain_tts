import random
from omegaconf import DictConfig
from torch.utils.data import Dataset
from tqdm import tqdm
from models_.vits.mel import spec_to_mel, wav_to_spec
import numpy as np
import torch
import soundfile as sf
import torch.nn.functional as F
import random

Lang_to_id = {'dutch' : 0, 
            'french' : 1,
            'german' : 2,
            'italian' : 3,
            'polish' : 4,
            'portuguese' : 5,
            'spanish' : 6}

class MLSDatasetTrain(Dataset):
    def __init__(self, conf: DictConfig) -> None:
        super(MLSDatasetTrain, self).__init__()
        self.conf = conf
        self.seed = 10
        self.datas, self.lengths, self.speaker_dict = self.get_datas(conf.data_path)
        self.datas = self.datas[:int(len(self.datas) * 0.9)]
        self.lengths = self.lengths[:int(len(self.lengths) * 0.9)]


    def get_datas(self, data_path):
        # data_path : file_path of 'data_info_added.txt'
        datas = []
        lengths = []
        with open(data_path, 'r') as f:
            raw_datas = f.readlines()
        random.Random(self.seed).shuffle(raw_datas)
        speakers = []

        for line in tqdm(raw_datas):
            wav_path, speaker_id , language, pseudo_phn = line.strip().split('|')
            speaker_id = '{}_{}'.format(language, speaker_id)
            speakers.append(speaker_id)

            lang_id = Lang_to_id[language]

            pseudo_phn = pseudo_phn.split(' ')
            leng = len(pseudo_phn)

            ix = 0 
            while ix+1 < len(pseudo_phn):
                if pseudo_phn[ix] == pseudo_phn[ix+1]:
                    pseudo_phn.pop(ix)
                else:
                    ix += 1
            pseudo_phn = np.array(pseudo_phn)
            pseudo_phn = pseudo_phn.astype(np.int)
            pseudo_phn += 1 # because pad_id = 0

            datas.append( (wav_path, lang_id, pseudo_phn, speaker_id) )
            lengths.append(leng)

        speakers = list(set(speakers))
        speaker_dict = {}
        for i in range(len(speakers)):
            speaker_dict[speakers[i]] = i

        print('speaker num : ', len(speakers))

        return datas, lengths, speaker_dict

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, idx):
        wav_path, lang_id, pseudo_phn, speaker_id = self.datas[idx]
        speaker_id = self.speaker_dict[speaker_id]

        token = torch.from_numpy(pseudo_phn) # [T]
        token_length = len(pseudo_phn)

        wav, sr = sf.read(wav_path)
        assert sr == 16e3
        wav = wav.astype(np.float32)

        # zero mean - unit variance normalize
        if len(wav) < self.conf.crop_size:
            cropped_wav = F.pad(wav, (0, self.conf.crop_size - len(wav)), value= 0)
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
        
        aux_input = {"d_vector": None, "speaker_id": speaker_id, "language_id": lang_id}

        return token, token_length, wav, cropped_wav, spec, spec_length, mel, aux_input

class MLSDatasetEval(Dataset):
    def __init__(self, conf: DictConfig) -> None:
        super(MLSDatasetEval, self).__init__()
        self.conf = conf
        self.seed = 10
        self.datas, self.lengths, self.speaker_dict = self.get_datas(conf.data_path)
        self.datas = self.datas[int(len(self.datas) * 0.9):]
        self.lengths = self.lengths[int(len(self.lengths) * 0.9):]


    def get_datas(self, data_path):
        # data_path : file_path of 'data_info_added.txt'
        datas = []
        lengths = []
        with open(data_path, 'r') as f:
            raw_datas = f.readlines()
        random.Random(self.seed).shuffle(raw_datas)
        speakers = []

        for line in tqdm(raw_datas):
            wav_path, speaker_id , language, pseudo_phn = line.strip().split('|')
            speaker_id = '{}_{}'.format(language, speaker_id)
            speakers.append(speaker_id)
            lang_id = Lang_to_id[language]

            pseudo_phn = pseudo_phn.split(' ')
            leng = len(pseudo_phn)

            ix = 0 
            while ix+1 < len(pseudo_phn):
                if pseudo_phn[ix] == pseudo_phn[ix+1]:
                    pseudo_phn.pop(ix)
                else:
                    ix += 1
            pseudo_phn = np.array(pseudo_phn)
            pseudo_phn = pseudo_phn.astype(np.int)
            pseudo_phn += 1 # because pad_id = 0

            datas.append( (wav_path, lang_id, pseudo_phn, speaker_id) )
            lengths.append(leng)

        speakers = list(set(speakers))
        speaker_dict = {}
        for i in range(len(speakers)):
            speaker_dict[speakers[i]] = i

        return datas, lengths, speaker_dict

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, idx):
        wav_path, lang_id, pseudo_phn, speaker_id = self.datas[idx]
        speaker_id = self.speaker_dict[speaker_id]

        token = torch.from_numpy(pseudo_phn) # [T]
        token_length = len(pseudo_phn)

        wav, sr = sf.read(wav_path)
        assert sr == 16e3
        wav = wav.astype(np.float32)

        # zero mean - unit variance normalize
        if len(wav) < self.conf.crop_size:
            cropped_wav = F.pad(wav, (0, self.conf.crop_size - len(wav)), value= 0)
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
    
        aux_input = {"d_vector": None, "speaker_id": speaker_id, "language_id": lang_id}

        return token, token_length, wav, cropped_wav, spec, spec_length, mel, aux_input

class MLSLanguageBatchTrain(Dataset):
    def __init__(self, conf: DictConfig) -> None:
        super(MLSLanguageBatchTrain, self).__init__()
        self.conf = conf
        self.seed = 10
        self.speaker_dict = self.get_speaker(conf.data_paths)
        self.datas, self.lengths_dict = self.get_datas(conf.data_paths)
        
        smallest_wav_num = np.inf
        smallest_wav_num_lang = None
        for lang_key in self.lengths_dict:
            if len(self.lengths_dict[lang_key]) < smallest_wav_num:
                smallest_wav_num = len(self.lengths_dict[lang_key])
                smallest_wav_num_lang = lang_key

        self.smallest_wav_num_lang = smallest_wav_num_lang

        self.lang_idx_lists = {}
        for lang_key in self.datas:
            self.lang_idx_lists[lang_key] = list(range(len(self.datas[lang_key])))


    def __len__(self): 
        return len(self.lengths_dict[self.smallest_wav_num_lang])

    def get_speaker(self, paths):
        speakers = []

        for i in range(len(paths)):
            data_path = paths[i]
            with open(data_path, 'r') as f:
                raw_datas = f.readlines()
            
            for line in tqdm(raw_datas):
                _, speaker_id , language, _ = line.strip().split('|')
                speaker_id = '{}_{}'.format(language, speaker_id)
                speakers.append(speaker_id)
        
        speakers = list(set(speakers))
        speaker_dict = {}
        for i in range(len(speakers)):
            speaker_dict[speakers[i]] = i

        print(len(speaker_dict))
        return speaker_dict

    def get_datas(self, paths):
        datas = {}
        lengths = {}
 
        for i in range(len(paths)):
            data_path = paths[i]
            with open(data_path, 'r') as f:
                raw_datas = f.readlines()
            random.Random(self.seed).shuffle(raw_datas)
            raw_datas = raw_datas[:int(len(raw_datas) * 0.9)]

            for line in tqdm(raw_datas):
                wav_path, speaker_id , language, pseudo_phn = line.strip().split('|')
                speaker_id = '{}_{}'.format(language, speaker_id)
                lang_id = Lang_to_id[language]
                pseudo_phn = pseudo_phn.split(' ')
                leng = len(pseudo_phn)

                ix = 0 
                while ix+1 < len(pseudo_phn):
                    if pseudo_phn[ix] == pseudo_phn[ix+1]:
                        pseudo_phn.pop(ix)
                    else:
                        ix += 1
                pseudo_phn = np.array(pseudo_phn)
                pseudo_phn = pseudo_phn.astype(np.int)
                pseudo_phn += (128 * lang_id + 1) # consider padding and language index
                
                if language not in datas.keys():
                    datas[language] = []
                    datas[language].append( (wav_path, lang_id, pseudo_phn, speaker_id) )
                    lengths[language] = []
                    lengths[language].append(leng)
                else:
                    datas[language].append( (wav_path, lang_id, pseudo_phn, speaker_id) )
                    lengths[language].append(leng)
        
        return datas, lengths


    def __getitem__(self, idx):
        item = {}

        for lang_key in self.lang_idx_lists:
            
            if len(self.lang_idx_lists[lang_key]) == 0:
                self.lang_idx_lists[lang_key] = list(range(len(self.datas[lang_key])))

            random_ix = np.random.randint(0,len(self.lang_idx_lists[lang_key]))
            data_ix = self.lang_idx_lists[lang_key].pop(random_ix)
            data = self.datas[lang_key][data_ix]

            wav_path, lang_id, pseudo_phn, speaker_id = data

            speaker_id = self.speaker_dict[speaker_id]

            token = torch.from_numpy(pseudo_phn) # [T]
            token_length = len(pseudo_phn)

            wav, sr = sf.read(wav_path)
            assert sr == 16e3
            wav = wav.astype(np.float32)

            # zero mean - unit variance normalize
            if len(wav) < self.conf.crop_size:
                cropped_wav = F.pad(wav, (0, self.conf.crop_size - len(wav)), value= 0)
                cropped_wav = (cropped_wav - cropped_wav.mean()) / np.sqrt(cropped_wav.var() + 1e-7)
            else:
                start_ix = np.random.randint( len(wav) - self.conf.crop_size + 1)
                cropped_wav = wav[start_ix: start_ix+self.conf.crop_size]
                cropped_wav = (cropped_wav - cropped_wav.mean()) / np.sqrt(cropped_wav.var() + 1e-7)
                
            cropped_wav = torch.from_numpy(cropped_wav) #[crop_len]

            wav = (wav - wav.mean()) / np.sqrt(wav.var() + 1e-7)
            wav = torch.from_numpy(wav)

            wav = wav.unsqueeze(0)
            wav = wav.unsqueeze(0) 

            spec = wav_to_spec(y=wav, n_fft=self.conf.fft_size, hop_length=self.conf.hop_length,
                                win_length=self.conf.win_length, center=False) # [1,spec_dim,t]

            spec_length = spec.size(-1)

            mel = spec_to_mel(spec, n_fft=self.conf.fft_size, num_mels= self.conf.num_mels, 
                            sample_rate=self.conf.sample_rate, fmin = self.conf.mel_fmin, fmax = self.conf.mel_fmax) # [1, mel_dim, t]
            

            aux_input = {"d_vector": None, "speaker_id": speaker_id, "language_id": lang_id}

            item[lang_key] = (token, token_length, wav, cropped_wav, spec, spec_length, mel, aux_input)

        return item

class MLSLanguageBatchEval(Dataset):
    def __init__(self, conf: DictConfig) -> None:
        super(MLSLanguageBatchEval, self).__init__()
        self.conf = conf
        self.seed = 10
        self.speaker_dict = self.get_speaker(conf.data_paths)


        self.datas, self.lengths_dict = self.get_datas(conf.data_paths)
        
        smallest_wav_num = np.inf
        smallest_wav_num_lang = None
        for lang_key in self.lengths_dict:
            if len(self.lengths_dict[lang_key]) < smallest_wav_num:
                smallest_wav_num = len(self.lengths_dict[lang_key])
                smallest_wav_num_lang = lang_key

        self.smallest_wav_num_lang = smallest_wav_num_lang
        self.lang_idx_lists = {}
        for lang_key in self.datas:
            self.lang_idx_lists[lang_key] = list(range(len(self.datas[lang_key])))


    def __len__(self): 
        return len(self.lengths_dict[self.smallest_wav_num_lang])

    def get_speaker(self, paths):
        speakers = []

        for i in range(len(paths)):
            data_path = paths[i]
            with open(data_path, 'r') as f:
                raw_datas = f.readlines()
            
            for line in tqdm(raw_datas):
                _, speaker_id , language, _ = line.strip().split('|')
                speaker_id = '{}_{}'.format(language, speaker_id)
                speakers.append(speaker_id)
        
        speakers = list(set(speakers))
        speaker_dict = {}
        for i in range(len(speakers)):
            speaker_dict[speakers[i]] = i

        return speaker_dict

    def get_datas(self, paths):
        datas = {}
        lengths = {}

        for i in range(len(paths)):
            data_path = paths[i]
            with open(data_path, 'r') as f:
                raw_datas = f.readlines()
            random.Random(self.seed).shuffle(raw_datas)
            raw_datas = raw_datas[int(len(raw_datas) * 0.9):]

            for line in tqdm(raw_datas):
                wav_path, speaker_id , language, pseudo_phn = line.strip().split('|')
                speaker_id = '{}_{}'.format(language, speaker_id)
                lang_id = Lang_to_id[language]
                pseudo_phn = pseudo_phn.split(' ')
                leng = len(pseudo_phn)

                ix = 0 
                while ix+1 < len(pseudo_phn):
                    if pseudo_phn[ix] == pseudo_phn[ix+1]:
                        pseudo_phn.pop(ix)
                    else:
                        ix += 1
                pseudo_phn = np.array(pseudo_phn)
                pseudo_phn = pseudo_phn.astype(np.int)
                pseudo_phn += (128 * lang_id + 1) 
                
                if language not in datas.keys():
                    datas[language] = []
                    datas[language].append( (wav_path, lang_id, pseudo_phn, speaker_id) )
                    lengths[language] = []
                    lengths[language].append(leng)
                else:
                    datas[language].append( (wav_path, lang_id, pseudo_phn, speaker_id) )
                    lengths[language].append(leng)
    
        return datas, lengths

    def __getitem__(self, idx):
        item = {}

        for lang_key in self.lang_idx_lists:
            
            if len(self.lang_idx_lists[lang_key]) == 0: 
                self.lang_idx_lists[lang_key] = list(range(len(self.datas[lang_key])))

            random_ix = np.random.randint(0,len(self.lang_idx_lists[lang_key]))
            data_ix = self.lang_idx_lists[lang_key].pop(random_ix)
            data = self.datas[lang_key][data_ix]

            wav_path, lang_id, pseudo_phn, speaker_id = data
            speaker_id = self.speaker_dict[speaker_id]
            token = torch.from_numpy(pseudo_phn) # [T]
            token_length = len(pseudo_phn)

            wav, sr = sf.read(wav_path)
            assert sr == 16e3
            wav = wav.astype(np.float32)

            # zero mean - unit variance normalize
            if len(wav) < self.conf.crop_size:
                cropped_wav = F.pad(wav, (0, self.conf.crop_size - len(wav)), value= 0)
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
            
            aux_input = {"d_vector": None, "speaker_id": speaker_id, "language_id": lang_id}

            item[lang_key] = (token, token_length, wav, cropped_wav, spec, spec_length, mel, aux_input)

        return item