import os
from typing import List
from dataclasses import dataclass, field

""" 
    Definintions for directory paths and training parameters
    Train Config will be stored on disk in the config-files/ directory
    for later evalution runs.
"""
@dataclass
class TrainConfig:
    
    scratch_path: str = '/home/philipp/Data/cmip-gan'
    tensorboard_path: str = f'{scratch_path}/tensorboard/'
    checkpoint_path: str = f'{scratch_path}/checkpoints/'
    config_path: str = f'{scratch_path}/config-files/'
    fname_gfdl: str = f'{scratch_path}/datasets/gfdl_historical_v3.nc'
    fname_w5e5: str = f'{scratch_path}/datasets/W5E5v2_1979_2019_regridded_no_leap.nc'
    results_path: str = f'{scratch_path}/results/'
    projection_path: str = None

    train_start: int = 1979
    train_end: int = 1980  
    #train_end: int = 2000
    valid_start: int = 2001
    valid_end: int = 2001
    #valid_end: int = 2003
    valid_end: int = 2003
    test_start: int = 2004
    test_end: int = 2014

    model_name: str = 'gfdl_gan'
    comment: str = ''

    epochs: int = 250
    train_batch_size: int = 1
    test_batch_size: int = 8
    prefetch_queue_depth: int = 3
    num_resnet_layer: int = 7
    discriminator_layer: int = 3
    transforms: List = field(default_factory=lambda: ['log', 'normalize_minus1_to_plus1'])
    rescale: bool = False
    epsilon: float = 0.0001
    log_every_n_steps: int = 10
    norm_output: bool = True

    cmip_historical_log_max: str = 3.7
    w5e5_historical_log_max = 3.94 

""" Further definitions for model evalution """
@dataclass
class HistoricalConfig(TrainConfig):
    
    test_start: int = '2004'
    test_end: int = '2014'
    test_period = (test_start, test_end)

    scratch_path: str = '/p/tmp/hess/scratch/cmip-gan'
    fname_gfdl_isimip = f'{scratch_path}/datasets/isimip_gfdl_historical.nc'
    fname_w5e5 = f'{scratch_path}/datasets/W5E5v2_1979_2019_regridded_no_leap.nc'
    fname_w5e5_gan = '/p/tmp/hess/scratch/cmip-gan/results/gan_constrained_train.nc'
    fname_w5e5_gan_unconstrained = '/p/tmp/hess/scratch/cmip-gan/results/gan_unconstrained_train.nc'
    fname_w5e5_gan_isimip= f'{scratch_path}/datasets/isimip_input/gan_constrained_isimip_future.nc'