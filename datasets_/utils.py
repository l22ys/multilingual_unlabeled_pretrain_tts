'''
Copyright 2018 The Hugging Face team

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

from omegaconf import DictConfig
from typing import Union, Optional, List, Iterator
from torch.utils.data import DataLoader, Sampler, DistributedSampler, Dataset
import torch
import math

# Copied from 'https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_pt_utils.py'
def get_length_grouped_indices(lengths, batch_size, mega_batch_mult=None, generator=None) -> List[int]:
    """
    Return a list of indices so that each slice of `batch_size` consecutive indices correspond to elements of similar
    lengths. To do this, the indices are:

    - randomly permuted
    - grouped in mega-batches of size `mega_batch_mult * batch_size`
    - sorted by length in each mega-batch

    The result is the concatenation of all mega-batches, with the batch of `batch_size` containing the element of
    maximum length placed first, so that an OOM happens sooner rather than later.
    """
    # Default for mega_batch_mult: 50 or the number to get 4 megabatches, whichever is smaller.
    if mega_batch_mult is None:
        mega_batch_mult = min(len(lengths) // (batch_size * 4), 50)
        # Just in case, for tiny datasets
        if mega_batch_mult == 0:
            mega_batch_mult = 1

    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator) 
    megabatch_size = mega_batch_mult * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [list(sorted(megabatch, key=lambda i: lengths[i], reverse=True)) for megabatch in megabatches]

    # The rest is to get the biggest batch first.
    # Since each megabatch is sorted by descending length, the longest element is the first
    megabatch_maximums = [lengths[megabatch[0]] for megabatch in megabatches]
    max_idx = torch.argmax(torch.tensor(megabatch_maximums)).item()
    # Switch to put the longest element in first position
    megabatches[0][0], megabatches[max_idx][0] = megabatches[max_idx][0], megabatches[0][0]

    return [i for megabatch in megabatches for i in megabatch]

# Copied and modified from 'https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_pt_utils.py'
class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        dataset,
        lengths,
        model_input_name,
        generator,
    ):
        if dataset is None and lengths is None:
            raise ValueError("One of dataset and lengths must be provided.")

        self.batch_size = batch_size
        if lengths is None:
            if model_input_name is None:
                raise NotImplementedError
            lengths = [len(feature[model_input_name]) for feature in dataset] 
        
        self.lengths = lengths
        self.generator = generator

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        indices = get_length_grouped_indices(self.lengths, self.batch_size, generator=self.generator)
        return iter(indices)

# Copied and modified from 'https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_pt_utils.py'
class DistributedLengthGroupedSampler(DistributedSampler):
    r"""
    Distributed Sampler that samples indices in a way that groups together features of the dataset of roughly the same
    length while keeping a bit of randomness.
    """
    # Copied and adapted from PyTorch DistributedSampler.
    def __init__(
        self,
        batch_size: int,
        dataset,
        num_replicas: int,
        rank: int, # kind of gpu_number
        seed: int,
        drop_last: bool = False, 
        lengths = None,
        model_input_name = None,
    ):
        if dataset is None and lengths is None:
            raise ValueError("One of dataset and lengths must be provided.")
        if num_replicas is None:
            raise Exception('DDP mode needs num_replicas = world_size')
        if rank is None:
            raise Exception('DDP mode needs rank')

        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last

        if lengths is None:
            if model_input_name is None:
                raise NotImplementedError
            lengths = [len(feature[model_input_name]) for feature in dataset]
        self.lengths = lengths

        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.lengths) % self.num_replicas != 0:
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil((len(self.lengths) - self.num_replicas) / self.num_replicas)
        else:
            self.num_samples = math.ceil(len(self.lengths) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        self.seed = seed

    def __iter__(self) -> Iterator:
        # Deterministically shuffle based on epoch and seed
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch) 
        indices = get_length_grouped_indices(self.lengths, self.batch_size, generator=g)

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            indices += indices[: (self.total_size - len(indices))]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


def get_sampler_for_loader(dataset: Dataset, conf_dataloader: DictConfig, is_ddp:bool, rank: int) -> Optional[Union[Sampler, DistributedSampler]]: 
    # make sure that all process have same seed for random number generator when using DDP 
    if is_ddp:
        if conf_dataloader.is_group_by_length:
            lengths = dataset.lengths
            model_input_name = None 
            return DistributedLengthGroupedSampler(
                    conf_dataloader.batch_size * conf_dataloader.gradient_accumulation_steps,
                    dataset= dataset,
                    num_replicas=conf_dataloader.world_size,
                    rank=rank,
                    lengths=lengths,
                    model_input_name=model_input_name,
                    seed=conf_dataloader.seed,
                    drop_last= conf_dataloader.drop_last
                )
        else:
            return DistributedSampler(
                    dataset,
                    num_replicas=conf_dataloader.world_size,
                    rank=rank,
                    seed=conf_dataloader.seed,
                )

    else: 
        if conf_dataloader.is_group_by_length:
            lengths = dataset.lengths
            model_input_name = None
            generator = torch.Generator()
            generator.manual_seed(conf_dataloader.seed)
            return LengthGroupedSampler(
                    conf_dataloader.batch_size * conf_dataloader.gradient_accumulation_steps,
                    dataset= dataset,
                    lengths=lengths,
                    model_input_name=model_input_name,
                    generator=generator,
                )

        else:
            return None
    
