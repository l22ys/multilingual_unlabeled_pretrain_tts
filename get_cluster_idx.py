'''
Codes are copied and modified from "https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/unsupervised/scripts/wav2vec_apply_cluster_faiss.py"
'''

import argparse
import os.path as osp
import numpy as np
import tqdm
import torch

from collections import namedtuple

import faiss

import soundfile as sf
from glob import glob
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2ForPreTraining
import sys
import torch.nn.functional as F

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='directory of data') 
    parser.add_argument('--checkpoint', type=str, help='huggingface checkpoint for wav2vec model (if using wav2vec features)', required=True, default="facebook/wav2vec2-large-xlsr-53")
    parser.add_argument('--path', help='path to pca and centroids', required=True) 
    parser.add_argument('--layer', '-l', type=int, help='which layer to read', default=15) 
    parser.add_argument('--gpu', type=int, required=True)

    args = parser.parse_args()

    return args

faiss_spec = namedtuple("faiss_spec", ["pca", "norm", "n_clus", "sphere", "spec_str"])

def parse_faiss_specs(specs_str):
    specs = []
    for ss in specs_str.split():
        comps = ss.split("_")
        pca = 0
        norm = False
        n_clus = 0
        sphere = False
        for c in comps:
            if c.startswith("PCA"):
                pca = int(c[3:])
            elif c == "NORM":
                norm = True
            elif c.startswith("CLUS"):
                n_clus = int(c[4:])
            elif c == "SPHERICAL":
                sphere = True
        assert n_clus > 0
        specs.append(
            faiss_spec(pca=pca, norm=norm, n_clus=n_clus, sphere=sphere, spec_str=ss)
        )
    return specs

class Wav2Vec2FeatureExtractor:
    def __init__(self, ckpt, gpu, layer):
        model = Wav2Vec2ForPreTraining.from_pretrained(ckpt)
        self.gpu = gpu
        self.layer = layer
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        self.model = model.to(gpu)
    
    def read_audio(self, fname):
        wav, sr = sf.read(fname)
        assert sr == 16e3

        # zero mean / unit variance normalize
        wav = (wav - wav.mean()) / np.sqrt(wav.var() + 1e-7)
        return wav

    def get_feats(self, loc):
        x = self.read_audio(loc)
        with torch.no_grad():
            source = torch.from_numpy(x).view(1, -1).float().to(self.gpu)
            wav2vec2_output = self.model(source, output_hidden_states=True)
            res = wav2vec2_output.hidden_states[self.layer] # [1, Seq_length, feature_dim]
            return res.squeeze(0)

    def get_feats_leng(self, loc):
        x = self.read_audio(loc)
        return self.model._get_feat_extract_output_lengths(len(x))

def get_iterator(args):
    files = []
    speaker_ids = []
    languages = []

    langs = glob(args.data + '/*')
    for lang in langs:
        language = lang.split('/')[-1]
        with open(lang + '/data_info.txt', 'r') as f:
            lines = f.read().strip().split('\n')
            for line in lines:
                file_path, speaker_id, _ = line.split('|')
                
                files.append(file_path)
                speaker_ids.append(speaker_id)
                languages.append(language)

    num = len(files)
    extractor = Wav2Vec2FeatureExtractor(args.checkpoint, args.gpu, args.layer)

    def iterate():
        for ix in range(len(files)):
            fname = files[ix]
            language = languages[ix]
            speaker_id = speaker_ids[ix]

            feats = extractor.get_feats(fname)
            yield feats, fname, speaker_id, language

    return iterate, num

def get_iterator_one_language(args):
    files = []
    speaker_ids = []
    language = args.data.split('/')[-1]

    with open(args.data + '/data_info.txt', 'r') as f:
        lines = f.read().strip().split('\n')
        for line in lines:
            file_path, speaker_id, _ = line.split('|')
            
            files.append(file_path)
            speaker_ids.append(speaker_id)
            
    num = len(files)
    extractor = Wav2Vec2FeatureExtractor(args.checkpoint, args.gpu, args.layer)

    def iterate():
        for ix in range(len(files)):
            fname = files[ix]
            speaker_id = speaker_ids[ix]

            feats = extractor.get_feats(fname)
            yield feats, fname, speaker_id, language

    return iterate, num

def main():
    args = parse_args()
    spec = osp.basename(args.path)

    try: 
        faiss_spec = parse_faiss_specs(spec.rstrip("/"))[0] 
    except:
        print(spec)
        raise Exception

    print("Faiss Spec:", faiss_spec, file=sys.stderr)

    if faiss_spec.pca: 
        A = torch.from_numpy(np.load(osp.join(args.path, "pca_A.npy"))).cuda()
        b = torch.from_numpy(np.load(osp.join(args.path, "pca_b.npy"))).cuda()
        print("Loaded PCA", file=sys.stderr)

    centroids = np.load(osp.join(args.path, "centroids.npy"))
    print("Loaded centroids", centroids.shape, file=sys.stderr)

    res = faiss.StandardGpuResources()
    index_flat = (
        faiss.IndexFlatL2(centroids.shape[1]) 
        if not faiss_spec.sphere
        else faiss.IndexFlatIP(centroids.shape[1])
    )
    faiss_index = faiss.index_cpu_to_gpu(res, args.gpu, index_flat)
    faiss_index.add(centroids)

    generator, num = get_iterator(args) 
    iterator = generator()

    with open(args.path + '/data_info_added.txt', 'w') as fp:
        with torch.no_grad():
            for f, file_path, speaker_id, language in tqdm.tqdm(iterator, total=num):
                if faiss_spec.pca:
                    f = torch.mm(f, A) + b
                if faiss_spec.norm:
                    f = F.normalize(f, p=2, dim=-1)

                f = f.cpu().numpy() # shape : [Seq_length, feature_dim]

                _, z = faiss_index.search(f, 1) 
                cluster_ix_seq = " ".join(str(x.item()) for x in z)

                fp.write('{}|{}|{}|{}\n'.format(file_path, speaker_id, language, cluster_ix_seq))


def main_per_language():
    args = parse_args()
    spec = osp.basename(args.path)

    try: 
        faiss_spec = parse_faiss_specs(spec.rstrip("/"))[0]
    except:
        print(spec)
        raise Exception

    print("Faiss Spec:", faiss_spec, file=sys.stderr)

    if faiss_spec.pca: 
        A = torch.from_numpy(np.load(osp.join(args.path, "pca_A.npy"))).cuda()
        b = torch.from_numpy(np.load(osp.join(args.path, "pca_b.npy"))).cuda()
        print("Loaded PCA", file=sys.stderr)

    centroids = np.load(osp.join(args.path, "centroids.npy"))
    print("Loaded centroids", centroids.shape, file=sys.stderr)

    res = faiss.StandardGpuResources()
    index_flat = (
        faiss.IndexFlatL2(centroids.shape[1])
        if not faiss_spec.sphere
        else faiss.IndexFlatIP(centroids.shape[1])
    )
    faiss_index = faiss.index_cpu_to_gpu(res, args.gpu, index_flat)
    faiss_index.add(centroids)

    generator, num = get_iterator_one_language(args) 
    iterator = generator()

    with open(args.path + '/data_info_added.txt', 'w') as fp:
        with torch.no_grad():
            for f, file_path, speaker_id, language in tqdm.tqdm(iterator, total=num):
                if faiss_spec.pca:
                    f = torch.mm(f, A) + b
                if faiss_spec.norm:
                    f = F.normalize(f, p=2, dim=-1)

                f = f.cpu().numpy()

                _, z = faiss_index.search(f, 1) 
                cluster_ix_seq = " ".join(str(x.item()) for x in z) 

                fp.write('{}|{}|{}|{}\n'.format(file_path, speaker_id, language, cluster_ix_seq))


if __name__ == '__main__':
    # main()
    main_per_language()