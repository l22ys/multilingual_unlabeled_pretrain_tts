from omegaconf import DictConfig, listconfig
from typing import Tuple,Optional,Dict
from utils_.utils import build_datasets_from_config, build_models_from_config, build_optimizers_schedulers_from_config, dict_to_json, json_to_dict, \
                            build_optimizers_schedulers_from_config, build_scaler_from_config, build_losses_from_config
import torch.multiprocessing as mp
import os
import torch.distributed as dist
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
from tqdm import trange
from torch import nn
import numpy as np
from glob import glob
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import librosa.display
from utils_.helpers import segment
from models_.vits.mel import wav_to_mel
from utils_.utils import EndException

class Trainer:
    def __init__(self, conf: DictConfig) -> None:
        self.conf = conf
        self.gpu = self.conf.gpu # 'gpu number' or 'list of gpu numbers'(= DataDistributedParallel)
        if type(self.gpu) == listconfig.ListConfig:
            self.num_gpus = len(self.gpu)
            self.is_ddp = True
        else:
            self.num_gpus = 1
            self.is_ddp = False

    def train(self):
        if self.is_ddp:
            self.train_ddp()
        else:
            self.train_on_one_gpu(rank= self.gpu, world_size=1, self= self) # call static method 

    def train_ddp(self):
        assert self.is_ddp == True
        world_size = self.num_gpus
        mp.spawn(self.train_on_one_gpu, args=(world_size, self), nprocs=world_size, join=True)

    @staticmethod
    def train_on_one_gpu(rank, world_size, self):
        np.random.seed(self.conf.np_seed)
        wandb_config = {}
        for key in self.conf.for_wandb.config.keys(): 
            wandb_config[key] = self.conf.for_wandb.config[key] # Dictionary

        self.rank = rank
        
        self.accumulation_step = 0
        self.global_step = 0 
        self.global_epoch = 0
        self.run_id = None # for wandb
        self.best_eval_loss = np.inf
        

        if self.conf.trainer_data_path != None:
            trainer_data = json_to_dict(self.conf.trainer_data_path)
            self.accumulation_step = trainer_data['accumulation_step']
            self.global_step = trainer_data['global_step']
            self.global_epoch = trainer_data['global_epoch']
            self.run_id = trainer_data['run_id']
            self.best_eval_loss = trainer_data['best_eval_loss']
            

        if self.is_ddp:
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "29500"
            
            dist.init_process_group("nccl", rank=rank, world_size=world_size)
            if rank == 0:
                if self.conf.wandb_enable:
                    if self.run_id == None:
                        run = wandb.init(project=self.conf.for_wandb.project, name=self.conf.for_wandb.name, config=wandb_config)
                        self.run_id = run.id
                    else:
                        wandb.init(project=self.conf.for_wandb.project, name=self.conf.for_wandb.name, config=wandb_config, id=self.run_id ,resume='must' )
            
            dist.barrier()   

        else:
            if self.conf.wandb_enable:
                if self.run_id == None:
                    run = wandb.init(project=self.conf.for_wandb.project, name=self.conf.for_wandb.name, config=wandb_config)
                    self.run_id = run.id
                else:
                    wandb.init(project=self.conf.for_wandb.project, name=self.conf.for_wandb.name, config=wandb_config, id=self.run_id ,resume='must' )
                
        

        self.datasets, self.loaders, self.iterators = self.build_datasets()
        self.num_training_steps = self.get_num_training_steps()
        self.eval_loss_data = {}
        self.models = self.build_models()

        if self.is_ddp:
            for key, model in self.models.items():
                self.models[key] = DDP(model, device_ids=[rank], find_unused_parameters=self.conf.find_unused_parameters)

        if (not self.is_ddp) or (self.is_ddp and self.rank == 0):
            if self.conf.wandb_enable:
                for model in self.models.values():
                    # wandb.watch(model) 
                    pass

        if self.is_ddp:
            dist.barrier()

        self.optims, self.schedulers = self.build_optims_schedulers()
        self.scaler = self.build_scaler()
        self.losses = self.build_losses()
        pbar_epoch = trange(self.conf.total_epochs, position=0)
        pbar_epoch.set_description_str('Epoch')

        
        for _ in pbar_epoch:            
            epoch_eval_loss = self.train_epoch()
                        
            self.global_epoch += 1
            
            if self.global_epoch % self.conf.logging.save_freq == 0: 
                self.save(epoch_eval_loss)

            if self.is_ddp:
                dist.barrier() 

        if self.conf.wandb_enable:
            wandb.finish()

        if self.is_ddp:
            dist.barrier()

    def get_num_training_steps(self) -> int:
        
        num_training_steps = (self.conf.total_epochs * len(self.loaders['train'])) // self.conf.datasets['train']['conf_dataloader']['gradient_accumulation_steps']
        return num_training_steps

    def build_datasets(self): 
        
        datasets, loaders, iterators = build_datasets_from_config(self.conf.datasets, self.is_ddp, self.rank)
        
        return datasets, loaders, iterators

    def build_models(self):
       
        models = build_models_from_config(self.conf.models)

        for key in models.keys():
            models[key] = models[key].to('cuda:{}'.format(self.rank))

        if self.conf.gradient_checkpointing:
            for key in models.keys():
                models[key].gradient_checkpointing_enable()

        return models

    def build_optims_schedulers(self):
    
        optims, schedulers = build_optimizers_schedulers_from_config(self.models, self.conf.models, self.num_training_steps, self.conf.gradient_accumulation_steps, self.conf.loss_type) 

        for key in optims.keys():
            for state in optims[key].state.values():
                for k,v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to('cuda:{}'.format(self.rank))

        return optims, schedulers

    def build_scaler(self):

        scaler = build_scaler_from_config(self.conf)
        
        return scaler

    def build_losses(self):

        losses = build_losses_from_config(self.conf.losses)

        for key, loss in losses.items():
            losses[key] = loss.to('cuda:{}'.format(self.rank))

        return losses



    def train_epoch(self):
        pbar_step = trange(len(self.loaders['train']), position=1) 
        pbar_step.set_description_str('STEP')
        epoch_eval_loss = None
        eval_loss_num = 0

        for _ in pbar_step:
            train_data = next(self.iterators['train'])

            loss_train_data, log_train_data = self.train_step(train_data)

            if (self.global_step) % self.conf.logging.wandb_freq == 1 and (self.accumulation_step % self.conf.gradient_accumulation_steps == 0):
                self.wandb_log(loss_train_data, log_train_data, mode = 'train')
                
                for i in range(self.conf.gradient_accumulation_steps): 
                    eval_data = next(self.iterators['eval'])
                    with torch.no_grad():
                        loss_eval_data, log_eval_data = self.eval_step(eval_data, i+1)
                        
                self.wandb_log(loss_eval_data, log_eval_data, mode= 'eval')
                if epoch_eval_loss == None:
                    epoch_eval_loss = loss_eval_data['gen_loss']
                else:
                    epoch_eval_loss += loss_eval_data['gen_loss']
                eval_loss_num += 1
        
        try:
            epoch_eval_loss /= eval_loss_num
        except:
            return np.inf
        
        return epoch_eval_loss.detach().cpu().item()

    def train_step(self, batch: Dict)-> Tuple[Optional[Dict],Optional[Dict]]:

        loss_data = {}    
        logs_data = {}
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                logs_data[key] = value.detach() 
                batch[key] = value.to('cuda:{}'.format(self.rank)) # ensure data on 'device'
            elif isinstance(value, dict):
                logs_copy_value = {}
                for k, v in value.items():
                    if isinstance(v, torch.Tensor):
                        logs_copy_value[k] = v.detach()
                        batch[key][k] = v.to('cuda:{}'.format(self.rank))
                    else:
                        logs_copy_value[k] = v
                logs_data[key] = logs_copy_value
            else:
                logs_data[key] = value
        
        for key in self.models.keys():
            self.models[key].train()

        scale_before = self.scaler.get_scale()

        with torch.cuda.amp.autocast(enabled=self.conf.fp16.use_amp): 
            scores_disc_fake, scores_disc_real, wav_pred_seg, wav_gt_seg = self.forward_model(batch, 0) 

        logs_data['wav_pred_seg'] = wav_pred_seg.detach().cpu()
        logs_data['wav_gt_seg'] =wav_gt_seg.detach().cpu()
        
        # Discriminator update
        loss_data['discriminator'] = self.losses['Dis'](scores_disc_real, scores_disc_fake) 

        self.scaler.scale(loss_data['discriminator']['loss']).backward()
        self.scaler.unscale_(self.optims['Dis'])

        # clipping
        if 'max_grad_norm' in self.conf.keys() and self.conf.max_grad_norm > 0:
            if hasattr(self.optims['Dis'],'clip_grad_norm'):
                # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping - copied from transformers.trainer.py
                self.optims['Dis'].clip_grad_norm(self.conf.max_grad_norm)

            elif hasattr(self.models['Dis'], 'clip_grad_norm_'):
                # Some models (like FullyShardedDDP) have a specific way to do gradient clipping - copied from transformers.trainer.py
                self.models['Dis'].clip_grad_norm_(self.conf.max_grad_norm)

            else:
                nn.utils.clip_grad_norm_(self.models['Dis'].parameters(), self.conf.max_grad_norm)
        self.scaler.step(self.optims['Dis'])

        
        # generator update
        with torch.cuda.amp.autocast(enabled=self.conf.fp16.use_amp):
            mel_slice, mel_slice_hat, scores_disc_fake, feats_disc_fake, feats_disc_real = self.forward_model(batch, 1)
        
        logs_data['mel_slice_gt'] = mel_slice.detach().cpu()
        logs_data['mel_slice_pred'] = mel_slice_hat.detach().cpu()
        
        loss_data['generator'] = self.losses['Gen'](
                                                    mel_slice_hat=mel_slice.float(),
                                                    mel_slice=mel_slice_hat.float(),
                                                    z_p=self.model_outputs_cache["z_p"].float(),
                                                    logs_q=self.model_outputs_cache["logs_q"].float(),
                                                    m_p=self.model_outputs_cache["m_p"].float(),
                                                    logs_p=self.model_outputs_cache["logs_p"].float(),
                                                    z_len=batch["spec_lengths"],
                                                    scores_disc_fake=scores_disc_fake,
                                                    feats_disc_fake=feats_disc_fake,
                                                    feats_disc_real=feats_disc_real,
                                                    use_speaker_encoder_as_loss=self.conf.models.Gen.use_speaker_encoder_as_loss,
                                                    gt_spk_emb=self.model_outputs_cache["gt_spk_emb"],
                                                    syn_spk_emb=self.model_outputs_cache["syn_spk_emb"],
                                                    )
        
        self.scaler.scale(loss_data['generator']['loss']).backward() 
        self.scaler.unscale_(self.optims['Gen'])

        # clipping
        if 'max_grad_norm' in self.conf.keys() and self.conf.max_grad_norm > 0:
            if hasattr(self.optims['Gen'],'clip_grad_norm'):
                # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping - copied from transformers.trainer.py
                self.optims['Gen'].clip_grad_norm(self.conf.max_grad_norm)

            elif hasattr(self.models['Gen'], 'clip_grad_norm_'):
                # Some models (like FullyShardedDDP) have a specific way to do gradient clipping - copied from transformers.trainer.py
                self.models['Gen'].clip_grad_norm_(self.conf.max_grad_norm)

            else:
                nn.utils.clip_grad_norm_(self.models['Gen'].parameters(), self.conf.max_grad_norm)
        self.scaler.step(self.optims['Gen'])

        # scaler update
        self.scaler.update()

        # scheduler update
        scale_after = self.scaler.get_scale()
        optimizer_was_run = scale_before <= scale_after
        if optimizer_was_run:
            for model_key in self.optims.keys():
                if self.conf.models[model_key].optim.scheduler.use_scheduler:
                    self.schedulers[model_key].step()
        
        for model_key in self.optims.keys():
            self.optims[model_key].zero_grad()


        return_loss_data = {}
        for k,v in loss_data['discriminator'].items():
            return_loss_data['dis_{}'.format(k)] = v
        for k,v in loss_data['generator'].items():
            return_loss_data['gen_{}'.format(k)] = v

        self.global_step += 1
        
        if self.global_step == self.conf.total_steps + 1:
            raise EndException # Finish training

        if self.global_step % self.conf.logging.save_freq_step == 0: 
            self.save(epoch_eval_loss=None)
        
        return return_loss_data, logs_data

    def eval_step(self, batch: Dict, i : int) -> Tuple[Dict, Dict]:
        loss_data = {}    
        logs_data = {} 
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                logs_data[key] = value.detach() 
                batch[key] = value.to('cuda:{}'.format(self.rank)) ## ensure data on 'device'
            elif isinstance(value, dict):
                logs_copy_value = {}
                for k, v in value.items():
                    if isinstance(v, torch.Tensor):
                        logs_copy_value[k] = v.detach()
                        batch[key][k] = v.to('cuda:{}'.format(self.rank))
                    else:
                        logs_copy_value[k] = v
                logs_data[key] = logs_copy_value
            else:
                logs_data[key] = value

        for key in self.models.keys():
            self.models[key].eval()

        with torch.cuda.amp.autocast(enabled=self.conf.fp16.use_amp): 
            scores_disc_fake, scores_disc_real, wav_pred_seg, wav_gt_seg = self.forward_model(batch, 0) 


        logs_data['wav_pred_seg'] = wav_pred_seg.detach().cpu()
        logs_data['wav_gt_seg'] =wav_gt_seg.detach().cpu()
        
        # Discriminator update
        loss_data['discriminator'] = self.losses['Dis'](scores_disc_real, scores_disc_fake)
        
        with torch.cuda.amp.autocast(enabled=self.conf.fp16.use_amp):
            mel_slice, mel_slice_hat, scores_disc_fake, feats_disc_fake, feats_disc_real = self.forward_model(batch, 1)
        
        
        logs_data['mel_slice_gt'] = mel_slice.detach().cpu()
        logs_data['mel_slice_pred'] = mel_slice_hat.detach().cpu()
        
        loss_data['generator'] = self.losses['Gen'](
                                                    mel_slice_hat=mel_slice.float(),
                                                    mel_slice=mel_slice_hat.float(),
                                                    z_p=self.model_outputs_cache["z_p"].float(),
                                                    logs_q=self.model_outputs_cache["logs_q"].float(),
                                                    m_p=self.model_outputs_cache["m_p"].float(),
                                                    logs_p=self.model_outputs_cache["logs_p"].float(),
                                                    z_len=batch["spec_lengths"],
                                                    scores_disc_fake=scores_disc_fake,
                                                    feats_disc_fake=feats_disc_fake,
                                                    feats_disc_real=feats_disc_real,
                                                    use_speaker_encoder_as_loss=self.conf.models.Gen.use_speaker_encoder_as_loss,
                                                    gt_spk_emb=self.model_outputs_cache["gt_spk_emb"],
                                                    syn_spk_emb=self.model_outputs_cache["syn_spk_emb"],
                                                    )
        

        return_loss_data = {}
        for k,v in loss_data['discriminator'].items():
            return_loss_data['dis_{}'.format(k)] = v
        for k,v in loss_data['generator'].items():
            return_loss_data['gen_{}'.format(k)] = v

        return return_loss_data, logs_data

    def forward_model(self, batch, i):
        # when i = 0 , for discriminator loss , when i =1 , for generator loss

        if i == 0: # for discriminator
         
            outputs = self.models['Gen'](x = batch['tokens'], x_lengths = batch['token_lengths'], y = batch['specs'],
                                    y_lengths = batch['spec_lengths'], waveform = batch['waveform'],
                                    cropped_waveform = batch['cropped_waveform'], aux_input = batch['aux_input'])
            
            self.model_outputs_cache = outputs
            
            scores_disc_fake, _, scores_disc_real, _ = self.models['Dis'](outputs['model_outputs'].detach(), outputs['waveform_seg'])
            return scores_disc_fake, scores_disc_real, outputs['model_outputs'], outputs['waveform_seg']

        elif i ==1: # for generator
            mel = batch["mel"]

            with torch.cuda.amp.autocast(enabled=False):

                if self.conf.models.Gen.encoder_sample_rate:
                    spec_segment_size = self.conf.models.Gen.spec_segment_size * int(self.conf.models.Gen.interpolate_factor)
                else:
                    spec_segment_size = self.conf.models.Gen.spec_segment_size

                mel_slice = segment(
                    mel.float(), self.model_outputs_cache["slice_ids"], spec_segment_size, pad_short=True
                )
                mel_slice_hat = wav_to_mel(
                    y=self.model_outputs_cache["model_outputs"].float(),
                    n_fft=self.conf.losses.Gen.audio.fft_size,
                    sample_rate=self.conf.losses.Gen.audio.sample_rate,
                    num_mels=self.conf.losses.Gen.audio.num_mels,
                    hop_length=self.conf.losses.Gen.audio.hop_length,
                    win_length=self.conf.losses.Gen.audio.win_length,
                    fmin=self.conf.losses.Gen.audio.mel_fmin,
                    fmax=self.conf.losses.Gen.audio.mel_fmax,
                    center=False,
                ) 

            # compute discriminator scores and features
            scores_disc_fake, feats_disc_fake, _, feats_disc_real = self.models['Dis'](
                self.model_outputs_cache["model_outputs"], self.model_outputs_cache["waveform_seg"]
            )
            return mel_slice, mel_slice_hat, scores_disc_fake, feats_disc_fake, feats_disc_real

        else:
            raise NotImplementedError

    def wandb_log(self, loss_train_data, log_train_data, mode: str): 

        assert loss_train_data != None and log_train_data != None
        if not self.conf.wandb_enable:
            return

        if mode == 'train':
            if (self.is_ddp and self.rank == 0) or (not self.is_ddp):
                train_log_dict = {}
                for key in loss_train_data.keys():
                    if isinstance(loss_train_data,torch.Tensor):
                        train_log_dict['train_{}'.format(key)] = loss_train_data[key].detach().cpu().item()
                    else:
                        train_log_dict['train_{}'.format(key)] = loss_train_data[key]
                    if self.conf.loss_type == 'mean':
                        train_log_dict['train_{}'.format(key)] /= self.conf.gradient_accumulation_steps
                
                # mel spectrogram
                train_mel_gt = log_train_data['mel_slice_gt'][0].numpy()
                fig = plt.Figure()
                ax1 = fig.add_subplot(1,2,1)
                p = librosa.display.specshow(train_mel_gt,sr=16000,hop_length=256,n_fft=1024,win_length=1024,fmin=0,fmax=None,x_axis='s',y_axis='mel', ax= ax1)
                ax1.set(title='gt_mel')

                train_mel_gen = log_train_data['mel_slice_pred'][0].numpy()
                ax2 = fig.add_subplot(1,2,2)
                p = librosa.display.specshow(train_mel_gen,sr=16000,hop_length=256,n_fft=1024,win_length=1024,fmin=0,fmax=None,x_axis='s',y_axis='mel', ax= ax2)
                ax2.set(title='gen_mel')
                
                fig.tight_layout()
                fig.savefig(os.path.join(self.conf.logging.log_dir, 'train_mel.jpg'))
                train_log_dict['train_mel'] = wandb.Image(os.path.join(self.conf.logging.log_dir, 'train_mel.jpg'))

                #audio
                gt_audio_seg = log_train_data['wav_gt_seg'][0][0].numpy()
                train_log_dict['train_gt_audio_seg'] = wandb.Audio(gt_audio_seg, sample_rate=16000)

                gen_audio_seg = log_train_data['wav_pred_seg'][0][0].numpy()
                train_log_dict['train_gen_audio_seg'] = wandb.Audio(gen_audio_seg, sample_rate= 16000)

                wandb.log(train_log_dict, step = self.global_step)
            else:
                pass

        elif mode == 'eval':
            if (self.is_ddp and self.rank == 0) or (not self.is_ddp):
                eval_log_dict = {}
                for key in loss_train_data.keys():
                    if isinstance(loss_train_data,torch.Tensor):
                        eval_log_dict['eval_{}'.format(key)] = loss_train_data[key].detach().cpu().item()
                    else:
                        eval_log_dict['eval_{}'.format(key)] = loss_train_data[key]
                    if self.conf.loss_type == 'mean':
                        eval_log_dict['eval_{}'.format(key)] /= self.conf.gradient_accumulation_steps
                
                # mel spectrogram
                eval_mel_gt = log_train_data['mel_slice_gt'][0].numpy()
                fig = plt.Figure()
                ax1 = fig.add_subplot(1,2,1)
                p = librosa.display.specshow(eval_mel_gt,sr=16000,hop_length=256,n_fft=1024,win_length=1024,fmin=0,fmax=None,x_axis='s',y_axis='mel', ax= ax1)
                ax1.set(title='gt_mel')

                eval_gen_mel = log_train_data['mel_slice_pred'][0].numpy() 
                ax2 = fig.add_subplot(1,2,2)
                p = librosa.display.specshow(eval_gen_mel,sr=16000,hop_length=256,n_fft=1024,win_length=1024,fmin=0,fmax=None,x_axis='s',y_axis='mel', ax= ax2)
                ax2.set(title='gen_mel')

                fig.tight_layout()
                fig.savefig(os.path.join(self.conf.logging.log_dir, 'eval_mel.jpg'))
                eval_log_dict['eval_mel'] = wandb.Image(os.path.join(self.conf.logging.log_dir, 'eval_mel.jpg'))

                #audio
                gt_audio_seg = log_train_data['wav_gt_seg'][0][0].numpy()
                eval_log_dict['eval_gt_audio_seg'] = wandb.Audio(gt_audio_seg, sample_rate=16000)

                gen_audio_seg = log_train_data['wav_pred_seg'][0][0].numpy()
                eval_log_dict['eval_gen_audio_seg'] = wandb.Audio(gen_audio_seg, sample_rate= 16000)


                wandb.log(eval_log_dict, step = self.global_step)
            else:
                pass
        else:
            raise NotImplementedError     

        if self.is_ddp:
            dist.barrier()

    def save(self, epoch_eval_loss):
        
        if (not self.is_ddp) or (self.is_ddp and self.rank == 0):
            data = {}
            data['scaler'] = self.scaler.state_dict()

            for model_key in self.models.keys():
                data[model_key] = {}
                data[model_key]['model_state_dict'] = self.models[model_key].state_dict()
                if model_key in self.optims.keys():
                    data[model_key]['optimizer_state_dict'] = self.optims[model_key].state_dict()
                if model_key in self.schedulers.keys():
                    data[model_key]['scheduler_state_dict'] = self.schedulers[model_key].state_dict() # # works for 'LambdaLR of HuggingFace' (not been checked for other schedulers)

            if epoch_eval_loss == None:
                pass
            elif epoch_eval_loss < self.best_eval_loss:
                self.best_eval_loss = epoch_eval_loss
                dir_save_best = os.path.join(self.conf.logging.log_dir, 'best_checkpoint')
                os.makedirs(dir_save_best, exist_ok= True)
                path_save_best = os.path.join(dir_save_best, 'step_{}.pth'.format(self.global_step))
                past_best_ckpt_file = glob(os.path.join(dir_save_best,'*.pth'))
                
                assert len(past_best_ckpt_file) <= 1
                if len(past_best_ckpt_file) == 1:
                    past_best_ckpt_file = past_best_ckpt_file[0]
                    os.remove(past_best_ckpt_file) 

                torch.save(data, path_save_best)
                                
            dir_save = os.path.join(self.conf.logging.log_dir, 'checkpoint')
            os.makedirs(dir_save, exist_ok=True)
            path_save = os.path.join(dir_save, 'step_{}.pth'.format(self.global_step)) 
            torch.save(data, path_save)
            
            trainer_data = {}
            trainer_data['accumulation_step'] = self.accumulation_step
            trainer_data['global_step'] = self.global_step
            trainer_data['global_epoch'] = self.global_epoch
            trainer_data['run_id'] = self.run_id
            trainer_data['best_eval_loss'] = self.best_eval_loss
            
            dict_to_json(trainer_data, path_save.replace('.pth','.json'))

            # ensure the number of checkpoint files in the log directory
            if self.conf.logging.latest_checkpoint_num != None:
                assert type(self.conf.logging.latest_checkpoint_num) == int
                ckpt_list = glob(os.path.join(dir_save, '*.pth'))

                if len(ckpt_list) > self.conf.logging.latest_checkpoint_num:
                    oldest_ckpt = min(ckpt_list, key=os.path.getctime)
                    oldest_json = oldest_ckpt.replace('.pth','.json')

                    os.remove(oldest_ckpt)
                    os.remove(oldest_json)
        else:
            pass