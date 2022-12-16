import torch
import xarray as xr
import numpy as np
from dataclasses import dataclass
from src.utils import log_transform, norm_transform,  norm_minus1_to_plus1_transform
import cftime

import src.xarray_utils as xu


""" Main class for loading and preprocessing the data for training """
class CycleDataset(torch.utils.data.Dataset):
    
    def __init__(self, stage, config, verbose=True, epsilon=0.0001):

        self.transforms = config.transforms
        self.verbose = verbose
        self.epsilon = epsilon
        self.config = config
        self.w5e5 = None
        self.w5e5_tensor = None
        self.climate_model = None
        self.climate_model_tensor = None

        if hasattr(self.config, 'w5e5_historical_log_max'):
            self.w5e5_historical_log_max = config.w5e5_historical_log_max
        else:
            self.w5e5_historical_log_max = 3.94 # W5E5

        if hasattr(self.config, 'cmip_historical_log_max'):
            self.cmip_historical_log_max = config.cmip_historical_log_max
        else:
            self.cmip_historical_log_max = 3.7

        self.splits = {
                "train": [str(config.train_start), str(config.train_end)],
                "valid": [str(config.valid_start), str(config.valid_end)],
                "test":  [str(config.test_start), str(config.test_end)],
        }
        self.num_samples = None
        self.pepare_datasets(stage)


    def load_climate_model_data(self):
        """ Y-domain samples """

        if self.verbose: print(f'loading cmip from {self.config.fname_gfdl}')


        with xr.open_dataset(self.config.fname_gfdl) as climate_model:
            climate_model =  climate_model.precipitation
            climate_model = climate_model.chunk({'time': 50})
        climate_model = climate_model.transpose('time', 'latitude', 'longitude')
        return climate_model


    def load_w5e5_data(self):
        """ X-domain samples """
        if self.verbose: print(f'loading w5e5 from {self.config.fname_w5e5}')
        if self.config.fname_w5e5[43:] == 'zarr': 
            w5e5 = xr.open_zarr(self.config.fname_w5e5)
            w5e5 = w5e5.precipitation
        else:
            with xr.open_dataset(self.config.fname_w5e5) as w5e5:
                w5e5 = w5e5.chunk({'time': 50})
                w5e5 = w5e5.precipitation
                w5e5 = xu.shift_longitudes(w5e5)
                w5e5 = w5e5.transpose('time', 'latitude', 'longitude')
                w5e5 = w5e5.astype(float)
                #w5e5 = w5e5.load()
        return w5e5


    def pepare_datasets(self, stage:str):

        if self.verbose: print(f'\n STAGE {stage} \n')

        self.w5e5 = self.load_w5e5_data()
        self.climate_model = self.load_climate_model_data()

        if self.verbose: print('applying Target transforms..')

        self.w5e5 = self.w5e5.sel(time=slice(self.splits[stage][0],
                                 self.splits[stage][1]))

        self.w5e5 = self.w5e5.isel(latitude=slice(0,180))

        self.w5e5 = self.apply_transforms(self.w5e5, data_ref=None,
                                           x_ref_log_min=0,
                                           x_ref_log_max=self.w5e5_historical_log_max)
        self.w5e5_tensor = torch.from_numpy(self.w5e5.values).float()

        if self.verbose: print('applying CMIP transforms..')
        self.climate_model = self.climate_model.sel(time=slice(self.splits[stage][0],
                                                         self.splits[stage][1]))

        self.climate_model = self.climate_model.isel(latitude=slice(0,180))
        self.climate_model = self.apply_transforms(self.climate_model,
                                              data_ref=None,
                                              x_ref_log_min=0,
                                              x_ref_log_max=self.cmip_historical_log_max)

        self.climate_model_tensor = torch.from_numpy(self.climate_model.values).float()
        if self.verbose: print('finished.')

        self.num_samples = len(self.w5e5_tensor)

        np.testing.assert_array_equal(self.w5e5_tensor.shape, self.climate_model_tensor.shape,
                                      'w5e5 and climate model datasets dont have matching shapes.')


    def apply_transforms(self, data, data_ref=None, x_ref_log_min=None, x_ref_log_max=None):

        if 'log' in self.transforms:
            data = log_transform(data, self.epsilon)
            if data_ref is not None:
                data_ref = log_transform(data_ref, self.epsilon)

        if 'normalize' in self.transforms:
            if data_ref is not None:
                data = norm_transform(data, data_ref)

        if 'normalize_minus1_to_plus1' in self.transforms:
            if data_ref is not None:
                data = norm_minus1_to_plus1_transform(data, x_ref=data_ref)
            else: 
                data = norm_minus1_to_plus1_transform(data,
                                                      x_ref=None,
                                                      x_ref_min=x_ref_log_min,
                                                      x_ref_max=x_ref_log_max)
        
        return data


    def __getitem__(self, index):

        x = self.w5e5_tensor[index].unsqueeze(0)
        y = self.climate_model_tensor[index].unsqueeze(0)

        sample = {'A': x, 'B': y}
        
        return sample

    def __len__(self):
        return self.num_samples


class Transforms():
    
    def __init__(self):
        self.epsilon = 0.0001
        self.min_ref = 0
        self.max_ref = 4

    def crop(self, x):
        return x[:-1,:]
    
    def log(self, x):
        return np.log(x + self.epsilon) - np.log(self.epsilon)
    
    def inverse_log(self, x):
        return np.exp(x + np.log(self.epsilon)) - self.epsilon

    def normalize(self, x):
        """ normalize to [-1, 1] """
        results = (x - self.min_ref)/(self.max_ref - self.min_ref)
        results = results*2 - 1
        return results
    
    def inverse_normalize(self, x):
        x = (x + 1)/2
        results = x * (self.max_ref - self.min_ref) + self.min_ref
        return results
    
    def test(self):
        x = np.ones((10,10))
        x_ref = x
        x = self.log(x)
        x = self.normalize(x)
        x = self.inverse_normalize(x)
        x = self.inverse_log(x)
        np.testing.assert_array_equal(x, x_ref)


@dataclass
class TestData():
    
    target: xr.DataArray = None
    w5e5: xr.DataArray = None
    gan: xr.DataArray = None
    uuid: str = None
    model = None
    time_start = '2001'
    time_end = '2014'


    def model_name_definition(self, key):
        dict = {
            'target': 'Target',
            'w5e5': 'W5E5v2',
            'gan': 'GAN',
        }
        return dict[key]


    def colors(self, key):
        dict = {
            'target': 'k',
            'w5e5': 'b',
            'gan': 'r',
        }
        return dict[key]


    def convert_units(self):
        """ from mm/s to mm/d"""

        if self.target is not None:
            self.target = self.target*3600*24

        if self.w5e5 is not None:
            self.w5e5 = self.w5e5*3600*24

        if self.gan is not None:
            self.gan = self.gan*3600*24

    
    def crop_test_period(self):
        print('')
        print(f'Test set period: {self.time_start} - {self.time_end}')
        self.w5e5= self.w5e5.sel(time=slice(self.time_start, self.time_end))

        
    def show_mean(self):
        print('')
        print(f'Mean [mm/d]:')
        print(f'target: {self.w5e5.mean().values:2.3f}')
        print(f'GAN:  {self.gan.mean().values:2.3f}')


@dataclass
class CombinedData():
    
    w5e5: xr.DataArray = None
    w5e5_gan: xr.DataArray = None
    w5e5_gan_isimip: xr.DataArray = None
    w5e5_gan_unconstrained: xr.DataArray = None
    gfdl: xr.DataArray = None
    custom_isimip: xr.DataArray = None

    def model_name_definition(self, key):
        dict = {
            'w5e5': 'w5e5',
            'w5e5': 'W5E5v2',
            'w5e5_5': r'W5E5v2 (regridded to $(2^\circ)$)',
            'w5e5_gan': 'GAN',
            'w5e5_gan_isimip': 'GAN-ISIMIP3BASD',
            'w5e5_gan_unconstrained': 'GAN (unconstrained)',
            'gfdl': 'GFDL-ESM4',
            'custom_isimip': 'ISIMIP3BASD',
        }
        return dict[key]


    def colors(self, key):
        dict = {
            'w5e5': 'green',
            'w5e5': 'black',
            'w5e5_5': 'grey',
            'w5e5_gan': 'cyan',
            'w5e5_gan_unconstrained': 'orange',
            'w5e5_gan_isimip': 'blue',
            'gfdl': 'red',
            'custom_isimip': 'magenta',
        }
        return dict[key]
        

def load_cmip6_model(fname: str, historical_test_period: list) -> xr.DataArray:
    data = xr.open_dataset(fname).precipitation
    return data 


def slice_time(dataset, year_start, year_end):
    year_start = np.where(dataset.time.values == cftime.DatetimeNoLeap(year_start,1,1,12))[0][0]
    year_end = np.where(dataset.time.values == cftime.DatetimeNoLeap(year_end,12,31,12))[0][0]

    dataset = dataset.isel(time=slice(year_start, year_end))
    return dataset