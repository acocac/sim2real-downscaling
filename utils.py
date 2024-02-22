import datetime
from random import randrange
import os
#import wandb
import yaml
from pathlib import Path
import torch
from deepsensor.model.convnp import ConvNP


def random_day_in_interval(start, end):
    if isinstance(start, str):
        start = datetime.datetime.strptime(start, "%Y-%m-%d").date()
    if isinstance(end, str):
        end = datetime.datetime.strptime(end, "%Y-%m-%d").date()
    delta = end - start
    int_delta = delta.days
    random_day = randrange(int_delta)
    return start + datetime.timedelta(days=random_day)

def download_wandb_file(run_ref, file):
    # run_ref: 'entity/project/run_id'
    run_id = run_ref.split("/")[-1]
    api = wandb.Api()
    therun = api.run(run_ref)

    therun.file(file).download(root=os.path.join('wandb', f'run-{run_id}'), exist_ok=True)
    
    ckpt_file_path = str(os.path.join('wandb', f'run-{run_id}', file))
    return ckpt_file_path

def download_wandb_all_files(run_ref):
    run_id = run_ref.split("/")[-1]
    api = wandb.Api()
    therun = api.run(run_ref)
    
    local_folder = f'wandb/wandb/run-{run_id}/files'
    for file in therun.files():
        file.download(root=local_folder)

    return local_folder

def load_wandb_model(config_path, weights_path, data_processor, task_loader, encoder_scales=None):
    # Load config
    config = yaml.safe_load(Path(config_path).read_text())

    # Build model from config
    model_args = dict(
        unet_channels = config['unet_channels']['value'],
        unet_kernels = config['unet_kernels']['value'],
        likelihood = config['likelihood']['value'],
    )

    if encoder_scales is not None:
        model_args['encoder_scales'] = encoder_scales

    model = ConvNP(
        data_processor,
        task_loader,
        **model_args
    )

    map_location = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load model weights
    model.model.load_state_dict(torch.load(weights_path, map_location=map_location))

    return model