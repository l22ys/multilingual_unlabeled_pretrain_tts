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

from optparse import Option
import os
from omegaconf import DictConfig, OmegaConf
from typing import Tuple, Dict, Union, Optional
import importlib
from torch.utils.data import DataLoader, Dataset
from datasets_.utils import get_sampler_for_loader
import torch.nn as nn
import torch
import math
from transformers.optimization import get_scheduler
import json
from collections import OrderedDict

def dict_to_json(data: Dict, file_path:str):
    with open(file_path, 'w') as f:
        json.dump(data, f, ensure_ascii=False) 


def json_to_dict(file_path:str) -> Dict:
    data = {}
    with open(file_path, 'r') as f:
        data = json.load(f, encoding='cp949')
    return data
    
def ensure_conf(conf: Union[str, DictConfig]) -> DictConfig:
    if type(conf) == str:
        if conf.endswith('.yaml'):
            conf = OmegaConf.load(conf)
            return conf
        else:
            raise NotImplementedError
    elif conf == None:
        conf = """
        
        """
        conf = OmegaConf.create(conf)
        return conf
    elif type(conf) == DictConfig:
        return conf
    else:
        raise NotImplementedError

# copied from 'https://github.com/dhchoi99/NANSY/blob/master/utils/util.py'
def cycle(iterable):
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)

def load_checkpoint(model, path, key, model_type='model', strict=True):
    if model_type == 'model':
        # there is 'modules.' prefix if checkpoint file was made with DDP 
        data = torch.load(path, map_location='cpu') 

        model_state_dict = []
        for param_key in data[key]['model_state_dict']:
            if param_key.startswith('module.'):
                model_state_dict.append((param_key.replace('module.',''), data[key]['model_state_dict'][param_key]))
                
        if len(model_state_dict) != 0: # when it is a checkpoint of a model trained with DDP
            model_state_dict = OrderedDict(model_state_dict)
            model.load_state_dict(model_state_dict, strict= strict)
        else:
            model.load_state_dict(data[key]['model_state_dict'], strict=strict)
        return model

    elif model_type == 'optim':
        data = torch.load(path, map_location='cpu')
        optim = model
        optim.load_state_dict(data[key]['optimizer_state_dict'])
        return optim
    elif model_type == 'scheduler':
        data = torch.load(path, map_location='cpu')
        scheduler = model
        scheduler.load_state_dict(data[key]['scheduler_state_dict'])
        return scheduler
    elif model_type == 'scaler':
        data = torch.load(path, map_location='cpu') 
        scaler = model
        scaler.load_state_dict(data['scaler'])
        return scaler
    else:
        raise NotImplementedError

# copied from 'https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_pt_utils.py'
def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result

# copied and modified from 'https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py'
# - Use input as warmup_steps
def get_warmup_steps(warmup_steps, warmup_ratio: Optional[float]= None, num_training_steps: Optional[int]= None ):
        """
        Get number of steps used for a linear warmup.

        """
        warmup_steps = (
            warmup_steps if warmup_steps > 0 else math.ceil(num_training_steps * warmup_ratio)
        )
        return warmup_steps

def build_datasets_from_config(conf: DictConfig, is_ddp: bool, rank : int = 0) -> Tuple[ Dict[str, Dataset], Dict, Dict ] :

    datasets = {}
    loaders = {}
    iterators = {}

    for key, key_conf in conf.items():
        conf_dataset = key_conf['conf_dataset']
        module, cls = conf_dataset['class'].rsplit('.',1)
        D = getattr(importlib.import_module(module, package=None),cls)
        datasets[key] = D(conf_dataset)


        if 'conf_dataloader' in key_conf.keys():
            conf_dataloader = key_conf['conf_dataloader']
            sampler = get_sampler_for_loader(datasets[key], conf_dataloader, is_ddp = is_ddp, rank=rank)
            
            if conf_dataloader['conf_collate_fn'] == None:
                collate_fn = None
            else:                
                fn_module, Fn = conf_dataloader['conf_collate_fn']['class'].rsplit('.',1)
                collate_fn = getattr(importlib.import_module(fn_module,package=None), Fn)

                if isinstance(collate_fn, type): # when collate_fn is not 'function' but 'Class' 
                    collate_fn = collate_fn(conf_dataloader['conf_collate_fn'])

            loaders[key] = DataLoader(datasets[key], batch_size = conf_dataloader.batch_size, sampler = sampler, 
                                        num_workers=conf_dataloader.num_workers, collate_fn=collate_fn,
                                        pin_memory=conf_dataloader.pin_memory, drop_last=conf_dataloader.drop_last)


            iterators[key] = cycle(loaders[key])
    
    return datasets, loaders, iterators

def build_models_from_config(conf: DictConfig) ->Dict[str, nn.Module]:

    models = {}
    for key, conf_models in conf.items():
        module, cls = conf_models['class'].rsplit('.',1)
        M = getattr(importlib.import_module(module, package=None), cls)
        m = M(conf_models)
        
        if 'ckpt' in conf_models.keys() and os.path.isfile(conf_models['ckpt']):
            if 'ckpt_key_pair' in conf_models.keys():
                m = load_checkpoint(m, conf_models['ckpt'], conf_models['ckpt_key_pair'][key], strict=conf_models['ckpt_strict'])
                print('complete loading model checkpoint')
            else:
                m = load_checkpoint(m, conf_models['ckpt'], key, strict=conf_models['ckpt_strict'])
                print('complete loading model checkpoint')
        models[key] = m

    return models



def build_optimizers_schedulers_from_config(models: Dict[str, nn.Module], conf:DictConfig, num_training_steps: Optional[int],
                                            gradient_accumulation_steps: int, loss_type: str) -> Tuple[Dict, Dict]:

    optims = {}
    schedulers = {}

    for key in models.keys():
        model = models[key]
        conf_optim = conf[key].optim
        if conf_optim == None:
            continue 

        decay_parameters = get_parameter_names(model, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if (n in decay_parameters) and (p.requires_grad == True)],
                    "weight_decay": conf_optim['weight_decay'],
                },
                {
                    "params": [p for n, p in model.named_parameters() if (n not in decay_parameters) and (p.requires_grad == True)],
                    "weight_decay": 0.0,
                },
            ] 

        optim_module, optim_cls = conf_optim['class'].rsplit('.',1)
        O = getattr(importlib.import_module(optim_module, package=None), optim_cls)
        kwargs = conf_optim['kwargs']

        if loss_type == 'sum':
            pass
        elif loss_type == 'mean':
            kwargs['lr'] = kwargs['lr'] / gradient_accumulation_steps
        else:
            raise NotImplementedError('loss type must be either sum or mean')

        if 'eps' in kwargs.keys():
            kwargs['eps'] = float(kwargs['eps']) # beacuse yaml file take such as 1e-8 as str type

        optim = O(optimizer_grouped_parameters, **kwargs)

        if 'ckpt' in conf_optim.keys() and os.path.isfile(conf_optim['ckpt']):
            if 'ckpt_key_pair' in conf_optim.keys():
                optim = load_checkpoint(optim, conf_optim['ckpt'], key=conf_optim['ckpt_key_pair'][key], model_type='optim')
                print('complete loading optim checkpoint')
            else:
                optim = load_checkpoint(optim, conf_optim['ckpt'], key, model_type='optim')
                print('complete loading optim checkpoint')

        optims[key] = optim

        conf_scheduler = conf_optim['scheduler']
        if not conf_scheduler['use_scheduler']:
            continue

        assert conf_scheduler.SchedulerType in ['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup']

        scheduler = get_scheduler(
                name=conf_scheduler.SchedulerType,
                optimizer=optim,
                num_warmup_steps=conf_scheduler.warmup_steps,
                num_training_steps=num_training_steps,
            )
        
        
        if 'ckpt' in conf_optim.keys() and os.path.isfile(conf_optim['ckpt']): 
            scheduler = load_checkpoint(scheduler, conf_optim['ckpt'], key, model_type='scheduler')

        schedulers[key] = scheduler

    return optims, schedulers

def build_losses_from_config(conf: DictConfig) -> Dict[str, nn.Module]:

    losses = {}

    for key, conf_loss in conf.items():
        module, cls = conf_loss['class'].rsplit('.',1)
        L = getattr(importlib.import_module(module, package=None), cls)
        loss = L(conf_loss)
        losses[key] = loss

    return losses

def build_scaler_from_config(conf: DictConfig):
    
    scaler = torch.cuda.amp.GradScaler(enabled=conf.fp16.use_amp)

    for key in conf.models.keys(): 
        conf_model = conf.models[key]
        if 'optim' in conf_model.keys():
            conf_optim = conf_model['optim']

            if conf_optim == None:
                continue

            if 'ckpt' in conf_optim.keys() and os.path.isfile(conf_optim['ckpt']):
                scaler = load_checkpoint(scaler, conf_optim['ckpt'], key, model_type = 'scaler')
                return scaler
        else:
            continue

    return scaler 

class EndException(Exception):
    pass