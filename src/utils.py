from unittest.mock import NonCallableMagicMock
from uuid import uuid1
from datetime import datetime
import os
import time
import pandas as pd
from IPython.display import display, HTML

import pytorch_lightning as pl
import numpy as np
import matplotlib as mpl, matplotlib.pyplot as plt
from pathlib import Path


def make_dict(path: str, data: dict):
    dir_names = path.split('/')[7:]
    if  len(dir_names) > 2:
        creation_time = os.path.getctime(path)
        creation_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(creation_time))
        model = dir_names[0]
        execution_date = dir_names[1]
        uuid = dir_names[2]
        
        data['Creation Date'].append(creation_time)
        data['Execution Date'].append(execution_date)
        data['UUID'].append(uuid)
        data['Path'].append(path)
        data['Model name'].append(model)
        
    return data


def show_checkpoints(path, model_name=None):
    paths = [x[0] for x in os.walk(path)]
    
    data =  {'Creation Date': [], 'Execution Date': [], 'UUID': [], 'Path': [], 'Model name': []}
    
    for path in paths:
        data = make_dict(path, data)
        
    df = pd.DataFrame(data=data)
    if model_name is not None:
        display(df.loc[df['Model name']==model_name].sort_values(by=['Creation Date'],ascending=False))
    else:
        display(df.sort_values(by=['Creation Date'],ascending=False))
    
    return df


def get_config(uuid: str, path=None):
    import json

    fname = f'{path}/config_model_{uuid}.json'

    with open(fname) as json_file:
        data = json.load(json_file)
    print(json.dumps(data, indent=4, sort_keys=True))
    return data


def get_uuid_from_path(path: str):
        import re
        uuid4hex = re.compile('[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}', re.I)
        uuid = uuid4hex.search(path).group(0)
        return uuid


def get_version():

    model_id = str(uuid1())
    date = datetime.now().date().strftime("%Y_%m_%d")
    version = f'{date}/{model_id}'

    return version


def get_checkpoint_path(config, version):

    model_name = config.model_name    
    checkpoint_path = config.checkpoint_path
    uuid_legth = 36
    date_legth = 10
    path = f'{checkpoint_path[:-1]}/{model_name}/{version[:date_legth]}/{version[len(version)-uuid_legth:]}'

    Path(path).mkdir(parents=True, exist_ok=True)

    return path


def save_config(config, version):
    import json
    uuid_legth = 36
    Path(config.config_path).mkdir(parents=True, exist_ok=True)
    fname = f'{config.config_path}config_model_{version[len(version)-uuid_legth:]}.json'
    with open(fname, 'w') as file:
        file.write(json.dumps(vars(config))) 


def config_from_file(file_name):
    import json
    with open(file_name) as json_file:
        data = json.load(json_file)
    config = ClassFromDict(data)
    return config


def config_dict_from_file(file_name):
    import json
    with open(file_name) as json_file:
        data = json.load(json_file)
    return data
        

class ClassFromDict:
    def __init__(self, dictionary):
        for k, v in dictionary.items():
            setattr(self, k, v)
        setattr(self, 'flag', None)


def set_environment():

    pl.seed_everything(42)
    mpl.rcParams["axes.grid"     ] = False
    mpl.rcParams["figure.figsize"] = (8, 4)


def log_transform(x, epsilon):
    return np.log(x + epsilon) - np.log(epsilon)


def inv_log_transform(x, epsilon):
    return np.exp(x + np.log(epsilon)) - epsilon


def norm_transform(x, x_ref):
    return (x - x_ref.min())/(x_ref.max() - x_ref.min())


def inv_norm_transform(x, x_ref):
    return x * (x_ref.max() - x_ref.min()) + x_ref.min()


def norm_minus1_to_plus1_transform(x, x_ref=None, x_ref_min=None, x_ref_max=None):
    if x_ref is not None:
        x_ref_max = x_ref.max()
        x_ref_min = x_ref.min()
    elif x_ref is None and (x_ref_max is None or x_ref_min is None):
        raise ValueError('Neither training set reference data nor min/max are defined.')
        
    results = (x - x_ref_min)/(x_ref_max - x_ref_min)
    results = results*2 - 1
    return results


def inv_norm_minus1_to_plus1_transform(x, x_ref):
    x = (x + 1)/2
    results = x * (x_ref.max() - x_ref.min()) + x_ref.min()
    return results


def make_seasonal(data, season):
    return data.where(data['time'].dt.season==season, drop=True)


def spatial_bias(target, prediction):
    bias = prediction.mean('time') - target.mean('time') 
    bias = abs(bias).mean().compute().values
    return f'bias {bias:2.3f}'

    
def compute_mean(data):
    print(data.w5e5.mean().compute())
    print(data.w5e5_gan_qm.mean().compute())
    print(data.w5e5.mean().compute())
    print(data.w5e5_gan_qm.mean().compute())
    print(data.isimip.mean().compute())

    
def compute_seasonal_bias(data, bias_function, season:str):
    print(f"Target | Model  | Bias")
    bias = bias_function(make_seasonal(data.w5e5, season),
                         make_seasonal(data.gfdl, season))
    print(f"W5E5   | GFDL   | {bias}")
    
    bias = bias_function(make_seasonal(data.w5e5, season),
                         make_seasonal(data.w5e5_gan, season))
    print(f"W5E5   | GAN | {bias}")
    
    bias = bias_function(make_seasonal(data.w5e5, season),
                         make_seasonal(data.w5e5_gan_unconstrained, season))
    print(f"W5E5   | GAN  unconst. | {bias}")
    
    bias = bias_function(make_seasonal(data.w5e5, season),
                         make_seasonal(data.w5e5_gan_qm, season))
    print(f"W5E5   | GAN-QM | {bias}")
    
    bias = bias_function(make_seasonal(data.w5e5, season),
                         make_seasonal(data.w5e5_gan_isimip, season))
    print(f"W5E5   | GAN-ISIMIP | {bias}")
    
    bias = bias_function(make_seasonal(data.w5e5, season),
                         make_seasonal(data.custom_isimip, season))
    print(f"W5E5   | QM | {bias}")
    

def compute_bias(data, bias_function):
    print(f"Target | Model          | Bias")
    bias = bias_function(data.w5e5, data.gfdl)
    print(f"W5E5   | GFDL           | {bias[5:]}")
    
    bias = bias_function(data.w5e5, data.w5e5_gan)
    print(f"W5E5   | GAN            | {bias[5:]}")
    
    bias = bias_function(data.w5e5, data.w5e5_gan_unconstrained)
    print(f"W5E5   | GAN unconst.   | {bias[5:]}")
    
    bias = bias_function(data.w5e5, data.w5e5_gan_qm)
    print(f"W5E5   | GAN-QM     | {bias[5:]}")
    
    bias = bias_function(data.w5e5, data.w5e5_gan_isimip)
    print(f"W5E5   | GAN-ISIMIP     | {bias[5:]}")
    
    bias = bias_function(data.w5e5, data.custom_isimip)
    print(f"W5E5   | QM  | {bias[5:]}")