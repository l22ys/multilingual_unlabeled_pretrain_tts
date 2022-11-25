# Leveraging Unlabeled Multilingual Speech Corpora with Clustering for Speech Synthesis



We propose to pretrain a TTS model with unlabeled multilingual speech corpora and then fine-tune it with a small paired speech-text dataset.

The overall procedures are as follows:

1) Make pseudo phoneme sequence corresponding to each unlabeled waveform using wav2vec 2.0 XLSR-53 and k-means clustering
2) Pre-train a model with those waveforms & pseudo phoneme sequences as if they are paired speech-text data
3) Fine-tune the model with a small paired dataset of target language

## Pre-requisites
1. Python 3.7
2. Clone this repository
3. Install python packages. Please refer to requirements.txt
4. Download datasets
    
    +) We used a subset of the MLS dataset for a multilingual speech dataset, and a subset of the LibriTTS for a low-resource dataset.
5. Build Cython version Monotonic Alignment Search if you want
```
cd utils_/monotonic_align
python setup.py build_ext --inplace
```
    
## Get corresponding pseudo phoneme sequence
```
# get and save centroids
python get_cluster.py data_dir_path --save-dir save_dir_path --checkpoint facebook/wav2vec2-large-xlsr-53 --gpu gpu_index



# get pseudo phoneme sequences using the nearest centroids for each waveform
python get_cluster_idx.py data_dir_path --checkpoint facebook/wav2vec2-large-xlsr-53 --path centroids_dir --gpu gpu_index
```
 
 ## Pre-training Examples
 ```
 # pre-train with the P1 setting
 python main_.py --config configs/P1.yaml --train
 
 # pre-train with the P2 setting
 python main_.py --config configs/P2.yaml --train
 
 ```
        
        
## Fine-tuning Examples
```
# fine-tune the model pre-trained with the P1 setting.
python main_.py --config configs/P1T1F2.yaml --train

# fine-tune the model pre-trained with the P2 setting.
python main_.py --config configs/P2T2F2.yaml --train
```

## License
Some of our codes are copied and modifed from 
1) [facebookresearch/fairseq](https://github.com/facebookresearch/fairseq) (MIT license)
2) [coqui-ai/TTS](https://github.com/coqui-ai/TTS) (MPL-2.0 license)
3) [huggingface/transformers](https://github.com/huggingface/transformers) (Apache-2.0 license)
4) [VITS implementation](https://github.com/jaywalnut310/vits/) (MIT license)
5) [NANSY implementation](https://github.com/dhchoi99/NANSY)

Our codes are under the MPL-2.0 license.