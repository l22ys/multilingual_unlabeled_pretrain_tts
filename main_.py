import argparse
from omegaconf import OmegaConf, DictConfig, listconfig
import os
import importlib
import datetime
import glob


def parse_args() -> argparse.Namespace:
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='config file path')
    parser.add_argument('--train', action='store_true', help='')
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()

    if args.train and args.test: 
        raise Exception()
    
    return args

def is_conf_complete(conf: DictConfig):
    pass
    # if type(conf.gpu) == listconfig:
    #     number_of_gpu = len(conf.gpu)
    # else:
    #     number_of_gpu = 1

    # assert conf.for_wandb.config.batch_size == conf.datasets.train.conf_dataloader.batch_size * number_of_gpu * conf.for_wandb.config.gradient_accumulation_steps

def make_environment_before_run(conf : DictConfig):
    
    if os.path.exists('restart_status/restart.txt'):
        os.remove('restart_status/restart.txt')

    conf.logging['log_dir'] = os.path.join(conf.logging['log_dir'], str(conf.logging['seed']))
    os.makedirs(conf.logging['log_dir'], exist_ok=True)


def train(conf : DictConfig):
     
    module, cls = conf.trainer.rsplit('.',1)
    Trainer = getattr(importlib.import_module(module, package=None),cls)
    trainer = Trainer(conf) 
    
    trainer.train()


def test(conf):
    module, cls = conf.trainer.rsplit('.',1)
    Trainer = getattr(importlib.import_module(module, package=None),cls)
    trainer = Trainer(conf) 
    
    trainer.test()

def make_restart_flag():
    with open('restart_status/restart.txt','w') as f:
        f.write('restart')

def main():
    args = parse_args()
    
    conf = OmegaConf.load(args.config)
    is_conf_complete(conf)

    make_environment_before_run(conf)
    
    if conf.debug:
        if args.train:
            train(conf)
        elif args.test:
            test(conf)
        else:
            raise NotImplementedError()
    else:

        def handle_exception():
            # edit the checkpoint file
            list_of_checkpoint_files = glob.glob(os.path.join(conf.logging['log_dir'], 'checkpoint/') + '*.json')
            if len(list_of_checkpoint_files) != 0:
                latest_checkpoint_json = max(list_of_checkpoint_files, key=os.path.getctime)
                conf['trainer_data_path'] = latest_checkpoint_json
                for model_key in conf['models'].keys():
                    conf['models'][model_key]['ckpt'] = latest_checkpoint_json.replace('json', 'pth')
                    conf['models'][model_key]['optim']['ckpt'] = latest_checkpoint_json.replace('json', 'pth')
                conf.logging['log_dir'] = conf.logging['log_dir'].rsplit('/',1)[0] # to maintain the log directory
                # reflect revisions in config yaml file
                OmegaConf.save(conf, args.config)

        try:
            if args.train:
                train(conf)
            elif args.test:
                test(conf)
            else:
                raise NotImplementedError()
        
        except KeyboardInterrupt as e:
            print('keyboardinterrupt!')
            handle_exception()
            with open(os.path.join(conf.logging['log_dir'], conf.logging['error_log']),'a') as f:
                now = datetime.datetime.now()
                f.write('{} \nStop program due to an KeyboardInterrupt : {}\n\n'.format(str(now), str(e)))
                        
        # For restarting regardless of whether using DDP or not
        except RuntimeError as e:
            if str(e)[:18]=='CUDA out of memory':
                handle_exception()
                with open(os.path.join(conf.logging['log_dir'], conf.logging['error_log']),'a') as f:
                    now = datetime.datetime.now()
                    f.write('{} \nRelaunch program due to an error : {}\n\n'.format(str(now), str(e)))
                make_restart_flag()
            else:
                with open(os.path.join(conf.logging['log_dir'], conf.logging['error_log']),'a') as f:
                    now = datetime.datetime.now()
                    f.write('{} \Stop program due to an error : {}\n\n'.format(str(now), str(e)))
        
        except Exception as e:
            if 'CUDA out of memory' in str(e):
                handle_exception()
                with open(os.path.join(conf.logging['log_dir'], conf.logging['error_log']),'a') as f:
                    now = datetime.datetime.now()
                    f.write('{} \nRelaunch program due to an error : {}\n\n'.format(str(now), str(e)))

                make_restart_flag()
            else:
                with open(os.path.join(conf.logging['log_dir'], conf.logging['error_log']),'a') as f:
                    now = datetime.datetime.now()
                    f.write('{} \Stop program due to an error : {}\n\n'.format(str(now), str(e)))

if __name__ == '__main__':
    main()