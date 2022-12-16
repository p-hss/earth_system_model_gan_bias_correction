import sys
sys.path.append("..")

import src.xarray_utils as xu
from src.configuration import TrainConfig
import pandas as pd
import xarray as xr
import time
import subprocess
import numpy as np


def load_w5e5(config, verbose=True):
    if verbose: print(f'loading w5e5 from {config.fname_w5e5}')
    if config.fname_w5e5[43:] == 'zarr': 
        w5e5 = xr.open_zarr(config.fname_w5e5)

    else:
        rename = None
        with xr.open_dataset(config.fname_w5e5) as w5e5:
            w5e5 = w5e5.chunk({'time': 50})
    w5e5 = xu.shift_longitudes(w5e5)
    w5e5 = w5e5.transpose('time', 'latitude', 'longitude')
    # convert from m/day to mm/s
    w5e5 = w5e5.astype(float)
    w5e5 = w5e5.isel(latitude=slice(0,180))
    
    return w5e5


def load_cmip(config, verbose=True):
    
    if verbose: print(f'loading cmip from {config.fname_gfdl}')
    cmip = xu.load(config.fname_gfdl,
                   chunks={'time': 50},
                   rename=None,
                   multi_files=False)
    cmip = cmip.chunk({'time': 1})
    cmip = cmip.isel(latitude=slice(0,180))

    return cmip


def add_land_sea_mask(data: xr.DataArray):
    mask = xr.open_dataset('/p/tmp/hess/land_sea_mask/w5e5_land_sea_mask_regridded.nc')
    data.coords['mask'] = (('latitude', 'longitude'), mask.isel(time=0, latitude=slice(0,180)).lsm)


def remove_leap_year(data):
    data = data.sel(time=~((data.time.dt.month == 2) & (data.time.dt.day == 29)))
    return data 


def prepare_isimip_correction():

    start_hist = 1979
    end_hist = 2003
    start_future = 2004
    end_future = 2014

    scratch_path = '/path/to/data'
    fname_target = f'{scratch_path}/W5E5v2_1979_2019_regridded_no_leap.nc'
    fname_model = f'{scratch_path}/gan_unconstrained_train.nc'
    fname_target_historical = f'{scratch_path}/W5E5v2_historical_fixed.nc'
    fname_model_historical = f'{scratch_path}/gan_unconstrained_historical_fixed.nc'
    fname_model_future = f'{scratch_path}/cmip-gan/datasets/isimip_input/gan_unconstrained_future_fixed.nc'

    print('Splitting target and gan dataset into historical and future .nc files.')

    print('inputs:')
    print(fname_target)
    print(fname_model)

    print('output:')

    target = xr.open_dataset(fname_target, chunks={'time': 1}).isel(latitude=slice(0,180)).precipitation
    target = target.load()

    model = xr.open_dataset(fname_model, chunks={'time': 1}).isel(latitude=slice(0,180)).precipitation
    model = model.load()
    
    model = model.transpose('latitude', 'longitude', 'time')
    target = target.transpose('latitude', 'longitude', 'time')

    hist_time = xr.cftime_range(start=str(start_hist),
                                end=str(end_hist+1),
                                calendar='proleptic_gregorian')[:-1]

    future_time = xr.cftime_range(start=str(start_future),
                                  end=str(end_future+1),
                                  calendar='proleptic_gregorian')[:-1]

    model_historical = model.sel(time=slice(str(start_hist), '2004-01-06'))

    np.testing.assert_equal(len(model_historical['time']), len(hist_time))

    model_historical['time'] = hist_time
    xu.write_dataset(model_historical, fname_model_historical)
    print(fname_model_historical)

    target_historical = target.sel(time=slice(str(start_hist), '2004-01-06'))
    target_historical['time'] = hist_time
    xu.write_dataset(target_historical, fname_target_historical)
    print(fname_target_historical)
    
    model_future = model.sel(time=slice('2003-12-29', str(end_future)))
    model_future['time'] = future_time
    xu.write_dataset(model_future, fname_model_future)
    print(fname_model_future)

    print(f'number of historical days {len(target_historical.time)} != 9131')
    print(f'number of future days {len(model_future.time)} != 4018')


if __name__ == "__main__":
    prepare_isimip_correction()