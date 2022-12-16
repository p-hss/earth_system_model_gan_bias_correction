import chunk
import os
import xarray as xr
import torch
import numpy as np
from tqdm import tqdm
from IPython.display import Image, display
import pickle
from pathlib import Path

from src.model_cycle_gan_cmip import CycleGAN, DataModule, ConstrainedGenerator
from src.data import TestData, CycleDataset, load_cmip6_model, CombinedData
from src.plots import PlotAnalysis
from src.utils import log_transform, inv_norm_transform, inv_log_transform, inv_norm_minus1_to_plus1_transform, norm_minus1_to_plus1_transform, config_from_file
from src.configuration import TrainConfig as Config
import src.xarray_utils as xu


""" Interate over model checkpoints and show the test set results. """
class EvaluateCheckpoints():
    
    def __init__(self,
                 checkpoint_path,
                 plot_summary=False,
                 show_plots=False,
                 save_model=False,
                 constrain=False,
                 epoch_index=None,
                 epoch_min=None,
                 run_train_and_test_set=False,
                 max_num_inference_steps=None,
                 projection_path=None,
                 verbose=True,
                 model_fname='gan_train.nc'
                 ):

        self.verbose = verbose
        self.config = Config()
        self.checkpoint_path = checkpoint_path
        self.config_path = Config.config_path
        self.config = self.load_config()
        self.reports_path = f'{Config.results_path}reports/'
        self.plot_summary = plot_summary
        self.uuid = None
        self.show_plots = show_plots
        self.gan_results = None
        self.save_model = save_model
        self.model_fname = model_fname
        self.model = None
        self.test_data = None
        self.run_train_and_test_set = run_train_and_test_set
        self.constrain = constrain
        self.epoch_index = epoch_index
        self.epoch_min = epoch_min
        self.max_num_inference_steps = max_num_inference_steps
        self.dataloader = None
        self.checkpoint_idx = 1
        self.inference = None
        self.dimensions = None
        self.target = None


    def load_config(self):

        path = self.checkpoint_path
        if self.verbose: print(path)
        self.uuid = self.get_uuid_from_path(path)
        config = config_from_file(f'{self.config_path}config_model_{self.uuid}.json')

        return config


    def get_dimensions(self, dataset):

        cmip =  dataset.climate_model
        self.target = dataset.load_w5e5_data()
        self.target = self.target.sel(time=slice(str(self.config.test_start),
                                                 str(self.config.test_end)))
        self.target = self.target.isel(latitude=slice(0,180))

        return cmip.time, cmip.latitude, cmip.longitude 


    def get_uuid_from_path(self, path: str):

        import re
        uuid4hex = re.compile('[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}', re.I)
        uuid = uuid4hex.search(path).group(0)

        return uuid


    def run(self):

        if self.run_train_and_test_set:
            self.config.test_start = '1979'
            self.config.test_end = '2014'
        
        
        files = self.get_files(self.checkpoint_path)

        if self.epoch_index is not None:
            files = [files[self.epoch_index-1]]
        if self.epoch_min is not None:
            files = files[self.epoch_min-1:]

        self.inference = Inference(self.config,
                        constrain=self.constrain,
                        #projection_path=self.projection_path,
                        verbose=self.verbose,
                        max_num_inference_steps=self.max_num_inference_steps)

        dataset = self.inference.dataset
        self.dimensions = self.get_dimensions(dataset)

        self.dataloader = self.inference.get_dataloader()
        self.inference.load_w5e5_cmip_datasets()

        gan_bias = []
        for i, fname in enumerate(files):

            self.num_checkpoints = len(files)
            if self.verbose: print(f'Checkpoint {self.checkpoint_idx} / {self.num_checkpoints}:')
            if self.verbose: print(fname)
            if self.verbose: print('')

            self.run_inference(fname)
            self.read_test_data(self.inference.target)

            if self.plot_summary:
                self.get_plots()

            gan_bias.append(self.evaluate_global_bias(self.test_data.w5e5,
                                                      self.test_data.gan,
                                                      'GAN'))
            self.checkpoint_idx = i+1
        
        return gan_bias

    
    def run_inference(self, path: str):
        
        self.inference.load_model(path)
        self.inference.compute(dataloader=self.dataloader)
        self.gan_results = self.inference.get_netcdf_result(self.dimensions)
        self.model = self.inference.get_model()

        if self.save_model:
            print(f'saving model output to {self.model_fname}')
            self.inference.write(self.model_fname, self.dimensions)

        return self.gan_results 

    
    def evaluate_latitudinal_mean(self, target: xr.DataArray, prediction: xr.DataArray, model_name: str):

        target = target.mean(dim=("longitude", "time"))
        prediction = prediction.mean(dim=("longitude", "time"))
        bias = abs(target-prediction).mean().compute().values
        if self.verbose: print(f'{model_name}, longitudinal bias: {bias:2.3f}')

        return bias


    def evaluate_global_bias(self, target: xr.DataArray, prediction: xr.DataArray, model_name: str):

        bias = prediction.mean('time') - target.mean('time') 
        bias = abs(bias).mean().compute().values
        if self.verbose: print(f'{model_name}, global bias: {bias:2.3f}')

        return bias
        
        
    def get_files(self, path: str):
        
        if os.path.isfile(path):
            files = []
            files.append(path) 
        else:
            files = os.listdir(path)
            for i, f in enumerate(files):
                files[i] = os.path.join(path, f) 

        return files

        
    def read_test_data(self, target):
    
        data = TestData(w5e5=target,
                        gan=self.gan_results.precipitation,
                        )

        data.convert_units()
        data.crop_test_period()
        data.uuid = self.uuid
        data.model = self.model
        self.test_data = data


    def get_test_data(self):
        return self.test_data


    def show_reports(self, uuid):

        path = f'{self.reports_path}{uuid}/'
        files = self.get_files(path)
        for file in files:
            fig = Image(filename=file)
            display(fig)
        
        
    def get_plots(self):

        plot = PlotAnalysis(self.test_data, names=['gan', 'w5e5'])
        new_dir = f'{self.reports_path}{self.uuid}/'
        create_folder(new_dir)
        plot.latitudinal_mean(single_plot=True)


""" Execute model on test data and return output as NetCDF. """
class Inference():
    
    def __init__(self,
                 config,
                 constrain=False,
                 projection_path=None,
                 verbose=True,
                 max_num_inference_steps=None):

        self.config = config
        self.verbose = verbose
        self.constrain = constrain
        self.results_path = config.results_path
        self.target = None
        self.cmip = None
        self.train_start = str(config.train_start)
        self.train_end = str(config.train_end)
        self.test_start = str(config.test_start)
        self.test_end = str(config.test_end)
        self.epsilon = config.epsilon
        self.projection_path = projection_path
        self.model = None
        self.model_output = None
        self.transforms = config.transforms
        self.max_num_inference_steps = max_num_inference_steps
        self.tst_batch_sz = 64
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        datamodule = DataModule(self.config,
                                     verbose=self.verbose,
                                     train_batch_size = 16,
                                     test_batch_size = self.tst_batch_sz)
        datamodule.setup("test")

        self.dataloader = datamodule.test_dataloader()
        self.dataset = datamodule.test


    def load_w5e5_cmip_datasets(self):

        self.target = self.dataset.w5e5
        self.target = self.target.sel(time=slice(str(self.config.test_start),
                                                 str(self.config.test_end)))
        self.target = self.target.isel(latitude=slice(0,180))

        self.cmip =  self.dataset.climate_model
        self.cmip['time'] = self.target.time
        self.test_data_dimesions()


    def test_data_dimesions(self):
        if self.verbose: print('Checking data dimensions')
        np.testing.assert_array_equal(len(self.cmip.time), len(self.target.time),
                                      'Time dimensions dont have matching shapes.') 
        np.testing.assert_array_equal(len(self.cmip.latitude), len(self.target.latitude),
                                    'Latitude dimensions dont have matching shapes.') 
        np.testing.assert_array_equal(len(self.cmip.longitude), len(self.target.longitude),
                                    'Longitude dimensions dont have matching shapes.') 


    def load_model(self, checkpoint_path):
        if hasattr(self.config, 'num_resnet_layer'):
            num_resnet_layer = self.config.num_resnet_layer
        else:
            num_resnet_layer = 9

        if hasattr(self.config, 'discriminator_layer'):
            discriminator_layer = self.config.discriminator_layer
        elif hasattr(self.config, 'num_discriminator_layer'):
            discriminator_layer = self.config.num_discriminator_layer
        else:
            discriminator_layer = 3

        model = CycleGAN.load_from_checkpoint(checkpoint_path=checkpoint_path,
                                              num_resnet_layer=num_resnet_layer,
                                              discriminator_layer=discriminator_layer
                                              )
        model.freeze()
        self.model = model.to(self.device)
        self.model = ConstrainedGenerator(self.model.g_B2A, constrain=self.constrain)


    def get_model(self):
        return self.model 


    def get_w5e5(self):
        return self.target


    def get_cmip(self):
        return self.cmip

        
    def get_dataloader(self):


        return self.dataloader


    def compute(self, dataloader=None):
        """ Use B (ESM) -> A (w5e5) generator for inference """

        test_data = dataloader

        data = []

        if self.verbose: print("Start inference:")
        #for idx, sample in enumerate(tqdm(test_data)):
        for idx, sample in enumerate(tqdm(test_data)):

            sample = sample['B'].to(self.device)
            yhat = self.model(sample)

            data.append(yhat.squeeze().cpu())
            if self.max_num_inference_steps is not None:
                if idx > self.max_num_inference_steps - 1:
                    break
            
        self.model_output = torch.cat(data)


    def get_netcdf_result(self, dimensions):

        time = dimensions[0][:self.model_output.shape[0]]
        latitude = dimensions[1].values
        longitude = dimensions[2].values

        gan_data= xr.DataArray(
            data=self.model_output,
            dims=["time", "latitude", "longitude"],
            coords=dict(
                time=time,
                latitude=latitude,
                longitude=longitude,
            ),
            attrs=dict(
                description="precipitation",
                units="mm/s",
            ))
        
        gan_dataset = gan_data.to_dataset(name="precipitation")
        self.gan_dataset = gan_dataset.transpose('time', 'latitude', 'longitude')

        return self.gan_dataset


    def inv_transform(self, data, reference=None):
        """ The output equals w5e5, therefore it needs to be
            rescaled with respect to it
        """
        if reference is None:
            reference = self.target.sel(time=slice(self.train_start, self.train_end)).values
            

        if 'log' in self.transforms:
            reference = log_transform(reference, self.epsilon)

        if 'normalize' in self.transforms:
            data = inv_norm_transform(data, reference)

        if 'normalize_minus1_to_plus1' in self.transforms:
            data = inv_norm_minus1_to_plus1_transform(data, reference)

        if 'log' in self.transforms:
            data = inv_log_transform(data, self.epsilon)

        return data

    
    def write(self, fname, dimensions):
        
        ds = self.get_netcdf_result(dimensions)
        path  = self.results_path + fname
        xu.write_dataset(ds, path)


def get_best_model(fname):
    '''fname: path to .pkl results'''
    
    with open(fname, 'rb') as handle:
        results = pickle.load(handle)
    
    mins = np.zeros(len(results['bias']))
    for i, bias in enumerate(results['bias']):
        mins[i] = np.min(np.array(bias))
        
    print(f'lowest bias: {np.argmin(mins)}')
    print(f'model uuid: {results["uuid"][np.argmin(mins)]}')
    print(f'model path: {results["path"][np.argmin(mins)]}')


""" Loads test set data from different NetCDF sources for evaluation """
class LoadData():

    def __init__(self,
        config
        ):

        self.config = config
        self.test_period = self.config.test_period
        print("Test period", self.test_period)
        self.unit_conversion = 3600*24
                

    def test_set_crop(self, data):
        data = data.sel(time=slice(self.test_period[0],
                                   self.test_period[1]))
        return data


    def get_w5e5_data(self):
        data = xr.open_dataset(self.config.fname_w5e5, chunks={'time': 1}).precipitation*self.unit_conversion
        data = self.test_set_crop(data)
        data = data.isel(latitude=slice(0,180))
        data = xu.reverse_latitudes(data)
        return data


    def get_w5e5_gan(self, time_axis=None):
        data = xr.open_dataset(self.config.fname_w5e5_gan).precipitation*self.unit_conversion
        data = self.test_set_crop(data)
        data = xu.reverse_latitudes(data)
        if time_axis is not None:
            data['time'] = time_axis
        return data


    def get_w5e5_gan_unconstrained(self):
        data = xr.open_dataset(self.config.fname_w5e5_gan_unconstrained).precipitation*self.unit_conversion
        data = self.test_set_crop(data)
        data = xu.reverse_latitudes(data)
        return data


    def get_w5e5_gan_qm_data(self, time_axis=None):
        data = xr.open_dataset(self.config.fname_w5e5_gan_qm).precipitation*self.unit_conversion
        data = self.test_set_crop(data)
        data = xr.where(data < 0, 0, data)
        data = xu.reverse_latitudes(data)
        data = data.sel(time=slice('2004-01-04', '2014'))
        if time_axis is not None:
            data['time'] = time_axis
        return data


    def get_w5e5_gan_isimip_data(self, time_axis=None):
        data = xr.open_dataset(self.config.fname_w5e5_gan_isimip).precipitation*self.unit_conversion
        data = self.test_set_crop(data)
        data = xr.where(data < 0, 0, data)
        data = xu.reverse_latitudes(data)
        data = data.sel(time=slice('2004-01-04', '2014'))
        if time_axis is not None:
            data['time'] = time_axis
        return data


    def get_custom_isimip_data(self, time_axis=None):
        data = xr.open_dataset(self.config.fname_custom_isimip).precipitation*self.unit_conversion
        data = self.test_set_crop(data)
        data = xr.where(data < 0, 0, data)
        data = xu.reverse_latitudes(data)
        data = data.sel(time=slice('2004-01-04', '2014'))
        if time_axis is not None:
            data['time'] = time_axis
        return data



    def get_cmip6_output(self, time_axis=None):
        data = load_cmip6_model(self.config.fname_gfdl, self.test_period)
        data = self.test_set_crop(data)*self.unit_conversion
        data = data.isel(latitude=slice(0,180))
        if time_axis is not None:
            data['time'] = time_axis
        data = xu.reverse_latitudes(data)
        return data

                
    def collect_historical_data(self):

        w5e5 = self.get_w5e5_data()
        w5e5_gan = self.get_w5e5_gan(time_axis=w5e5.time)
        w5e5_gan_unconstrained = self.get_w5e5_gan_unconstrained()
        w5e5_gan_isimip = self.get_w5e5_gan_isimip_data(time_axis=w5e5.time)
        custom_isimip = self.get_custom_isimip_data(time_axis=w5e5.time)
        gfdl = self.get_cmip6_output(time_axis=w5e5.time)

        test_data = CombinedData(
                             gfdl=gfdl,
                             w5e5=w5e5,
                             w5e5_gan=w5e5_gan,
                             w5e5_gan_isimip=w5e5_gan_isimip,
                             w5e5_gan_unconstrained=w5e5_gan_unconstrained,
                             custom_isimip=custom_isimip,
                             )

        return test_data


    def remove_leap_year(self, data):
        data = data.sel(time=~((data.time.dt.month == 2) & (data.time.dt.day == 29)))
        data = data.sel(time=slice(self.test_period[0], self.test_period[1])).to_dataset()
        return data


def create_folder(path):
    from pathlib import Path
    Path(path).mkdir(parents=True, exist_ok=True)

