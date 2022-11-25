'''
Codes are copied and modified from "https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/unsupervised/scripts/wav2vec_cluster_faiss.py"
'''

import argparse
import gc
import os
import os.path as osp
import numpy as np
import tqdm
import torch

from collections import namedtuple

import faiss

import soundfile as sf
from glob import glob
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2ForPreTraining

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='directory of data') 
    parser.add_argument('--save-dir', help='where to save the output', required=True)
    parser.add_argument('--checkpoint', type=str, help='huggingface checkpoint for wav2vec model (if using wav2vec features)', required=True, default="facebook/wav2vec2-large-xlsr-53")    
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

        # zero mean - unit variance normalize
        wav = (wav - wav.mean()) / np.sqrt(wav.var() + 1e-7)
        return wav

    def get_feats(self, loc):
        x = self.read_audio(loc)
        with torch.no_grad():
            source = torch.from_numpy(x).view(1, -1).float().to(self.gpu)
            wav2vec2_output = self.model(source, output_hidden_states=True)
            res = wav2vec2_output.hidden_states[self.layer] # [1, Seq_length, feature_dim]
            return res.squeeze(0) # [Seq_length, feature_dim]

    def get_feats_leng(self, loc):
        x = self.read_audio(loc)
        return self.model._get_feat_extract_output_lengths(len(x))

def get_iterator(args):
    # args.data : directory which has subdirectories containing 'data_info.txt'
    files = []

    langs = glob(args.data + '/*')
    for lang in langs:
        with open(lang + '/data_info.txt', 'r') as f:
            lines = f.read().strip().split('\n')
            for line in lines:
                file_path, _, _ = line.split('|')
                files.append(file_path)
    num = len(files)
    extractor = Wav2Vec2FeatureExtractor(args.checkpoint, args.gpu, args.layer)
    
    def iterate1():
        for fname in files:
            output_leng = extractor.get_feats_leng(fname)
            yield output_leng

    def iterate2():
        for fname in files:
            feats = extractor.get_feats(fname)
            yield feats.cpu().numpy() # array - shape : [Seq_length, feature_dim]

    return iterate1, iterate2 , num
    
def get_iterator_one_language(args):
    # args.data : directory containing 'data_info.txt'
    files = []

    lang = args.data
    
    with open(lang + '/data_info.txt', 'r') as f:
        lines = f.read().strip().split('\n')
        for line in lines:
            file_path, _, _ = line.split('|')
            files.append(file_path)
    num = len(files)
    extractor = Wav2Vec2FeatureExtractor(args.checkpoint, args.gpu, args.layer)
    
    def iterate1():
        for fname in files:
            output_leng = extractor.get_feats_leng(fname)
            yield output_leng

    def iterate2():
        for fname in files:
            feats = extractor.get_feats(fname)
            yield feats

    return iterate1, iterate2 , num

def main():
    args = parse_args()
    faiss_specs = parse_faiss_specs('CLUS128') 

    feat_path = osp.join(args.save_dir, "features")

    if osp.exists(feat_path + ".npy"):
        feats = np.load(feat_path + ".npy")
    else:
        generator1, generator2, num = get_iterator(args)
        iterator1 = generator1()

        total_length = 0
        for leng in tqdm.tqdm(iterator1, total=num):
            total_length += leng

        iterator2 = generator2()

        feats = np.zeros((total_length, 1024), dtype = np.float32)
        start_ix = 0

        for f in tqdm.tqdm(iterator2, total=num):
            leng = f.shape[0]
            assert f.shape[1] == 1024

            feats[start_ix:start_ix+leng] = f
            start_ix += leng

        del iterator1
        del iterator2
        del generator1
        del generator2


        print(feats.shape)

        os.makedirs(args.save_dir, exist_ok=True)
        
        gc.collect()
        torch.cuda.empty_cache()

    reload = False
    for spec in faiss_specs:
        print("Processing spec", spec)

        if reload: 
            print("Reloading...")
            del feats
            gc.collect()
            feats = np.load(feat_path + ".npy")

        save_path = osp.join(args.save_dir, spec.spec_str) 
        os.makedirs(save_path, exist_ok=True)
        d = feats.shape[-1] 
        x = feats
        if spec.pca > 0: 
            print("Computing PCA")
            pca = faiss.PCAMatrix(d, spec.pca)
            pca.train(x)
            d = spec.pca
            b = faiss.vector_to_array(pca.b)
            A = faiss.vector_to_array(pca.A).reshape(pca.d_out, pca.d_in)
            np.save(osp.join(save_path, "pca_A"), A.T)
            np.save(osp.join(save_path, "pca_b"), b)
            print("Applying PCA")
            x = pca.apply_py(x)
        
        if spec.norm:
            reload = spec.pca <= 0
            print("Normalizing")
            faiss.normalize_L2(x)

        print("Computing kmeans")


        kmeans = faiss.Kmeans(
            d,
            spec.n_clus,
            niter=50,
            verbose=True,
            spherical=spec.sphere,
            max_points_per_centroid=feats.shape[0],
            gpu=True,
            nredo=3, 
        )
        kmeans.train(x)
        np.save(osp.join(save_path, "centroids"), kmeans.centroids) 
        del kmeans
        del x
        gc.collect()

def main_per_language(): # apply clustering for each language
    args = parse_args()
    faiss_specs = parse_faiss_specs('CLUS128')

    feat_path = osp.join(args.save_dir, "features")

    if osp.exists(feat_path + ".npy"):
        feats = np.load(feat_path + ".npy")
    else:
        generator1, generator2, num = get_iterator_one_language(args)
        iterator1 = generator1()

        total_length = 0
        for leng in tqdm.tqdm(iterator1, total=num):
            total_length += leng

        iterator2 = generator2()

        feats = torch.zeros((total_length, 1024), dtype=torch.float32).to(args.gpu)
        start_ix = 0

        for f in tqdm.tqdm(iterator2, total=num):
            leng = f.shape[0]
            assert f.shape[1] == 1024

            feats[start_ix:start_ix+leng] = f
            start_ix += leng

        del iterator1
        del iterator2
        del generator1
        del generator2


        feats = feats.detach().cpu().numpy()
        print(feats.shape)

        os.makedirs(args.save_dir, exist_ok=True)
        
        gc.collect()
        torch.cuda.empty_cache()

    reload = False
    for spec in faiss_specs:
        print("Processing spec", spec)

        if reload:
            print("Reloading...")
            del feats
            gc.collect()
            feats = np.load(feat_path + ".npy")

        save_path = osp.join(args.save_dir, spec.spec_str) 
        os.makedirs(save_path, exist_ok=True)
        d = feats.shape[-1] 
        x = feats
        if spec.pca > 0: 
            print("Computing PCA")
            pca = faiss.PCAMatrix(d, spec.pca)
            pca.train(x)
            d = spec.pca
            b = faiss.vector_to_array(pca.b)
            A = faiss.vector_to_array(pca.A).reshape(pca.d_out, pca.d_in)
            np.save(osp.join(save_path, "pca_A"), A.T)
            np.save(osp.join(save_path, "pca_b"), b)
            print("Applying PCA")
            x = pca.apply_py(x)
        
        if spec.norm:
            reload = spec.pca <= 0
            print("Normalizing")
            faiss.normalize_L2(x)

        print("Computing kmeans")


        kmeans = faiss.Kmeans( 
            d,
            spec.n_clus,
            niter=200, 
            verbose=True,
            spherical=spec.sphere,
            max_points_per_centroid=feats.shape[0],
            gpu=True, 
            nredo=10, 
        )
        kmeans.train(x)
        np.save(osp.join(save_path, "centroids"), kmeans.centroids) # save centroids
        del kmeans
        del x
        gc.collect()


if __name__ == '__main__':
    # main()
    main_per_language()