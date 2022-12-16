from pysteps.utils.spectral import rapsd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import matplotlib.ticker as ticker
from tqdm.notebook import tqdm



""" Class for computing and plotting power spectral densities of spatial fields """
class SpatialSpectralDensity():
    
    def __init__(self, plot_data, names):
        
        self.time_period = ('2004', '2014')
        self.num_times = None
        self.names = names 
        self.plot_data = plot_data
        self.spectrum = {}
        print('test')
        
    def compute_mean_spectral_density(self, data: xr.DataArray, name=None):

        num_frequencies = np.max((len(data.latitude.values),
                                  len(data.longitude.values)))/2
        mean_spectral_density = np.zeros(int(num_frequencies))
        if self.num_times is None:
            num_times = int(len(data.time))
            
        elif self.timestamp is not None:
            num_times = 1
            
        else:
            num_times = self.num_times
        
        for t in tqdm(range(num_times)):
            if self.timestamp is not None:
                tmp = data.sel(time=self.timestamp).values
            else:
                tmp = data.isel(time=t).values
            psd, freq = rapsd(tmp, return_freq=True, normalize=True, fft_method=np.fft)
            mean_spectral_density += psd
        mean_spectral_density /= num_times
        
        return mean_spectral_density, freq
    
    
    def run(self, num_times=None, timestamp=None):
        
        for i,name in enumerate(self.names):
            data = getattr(self.plot_data, name)
            self.num_times = num_times
            self.timestamp = timestamp
            print(f'Processing {name}')
            self.spectrum[name], self.freq = self.compute_mean_spectral_density(data)

    def plot_distances(self, log=True, axis=None, fname=None, fontsize=None, linewidth=3, names=None):
        
        if names is not None:
            self.names = names

        if axis is None: 
            _, ax = plt.subplots(figsize=(10,8))
        else:
            ax = axis

        plt.rcParams.update({'font.size': 14})
        labels =  self.plot_data.model_name_definition
        colors =  self.plot_data.colors
        x_vals = 1/self.freq*1*111/2

        target = self.spectrum['w5e5']
        self.names.remove('w5e5')

        for i,name in enumerate(self.names):

            data = self.spectrum[name]
            error = abs(data - target)
            
            linewidth = 3
            if name == 'w5e5_gan':
                line_style = '--'
                alpha = 0.8
            if name == 'w5e5_5':
                line_style = '--'
                linewidth = 2
                alpha = 0.8
            elif name == 'w5e5_gan_unconstrained':
                line_style = '-.'
                alpha = 0.8
            elif name == 'w5e5_gan_isimip':
                line_style = ':'
                alpha = 0.7
            else:
                line_style = '-'
                alpha = 1.0
            
            label = f'{self.plot_data.model_name_definition(name)}'

            line, = ax.plot(x_vals, error,
                     label=label,
                     linestyle=line_style,
                     alpha=alpha,
                     color=colors(name),
                     linewidth=linewidth)

            if (name == 'w5e5_gan_unconstrained'):
                line.set_dashes([5, 9])
    
        ax.legend(loc='upper right', fontsize=fontsize)
        ax.set_xlim(x_vals[1]+1024, x_vals[-1]-32)
        if log:
            ax.set_yscale('log', base=10)
        #ax.set_ylim(5e-7, 5e-2)
        ax.set_xscale('log', base=2)
        ax.set_xticks([2**7, 2**8, 2**9, 2**10, 2**11, 2**12, 2**13], fontsize=fontsize)
        ax.tick_params(axis='x', labelsize=fontsize)
        ax.tick_params(axis='y', labelsize=fontsize)
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        ax.grid()
        #ax.set_ylim(4.5e-5, 0.07)
        ax.set_xlabel(r'Wavelength [km]', fontsize=fontsize)
        ax.set_ylabel('PSD absolute distance [a.u]', fontsize=fontsize)
        
        if fname is not None:
            plt.savefig(fname, format='png', bbox_inches='tight')
      
        
    def plot(self, axis=None, fname=None, fontsize=None, linewidth=3, names=None):
        
        if names is not None:
            self.names = names

        if axis is None: 
            _, ax = plt.subplots(figsize=(10,8))
        else:
            ax = axis

        plt.rcParams.update({'font.size': 14})
        labels =  self.plot_data.model_name_definition
        colors =  self.plot_data.colors
        x_vals = 1/self.freq*1*111/2

        for i,name in enumerate(self.names):
            
            linewidth = 3
            if name == 'w5e5_gan':
                line_style = '--'
                alpha = 0.8
            if name == 'w5e5_5':
                line_style = '--'
                linewidth = 2
                alpha = 0.6
            elif name == 'w5e5_gan_unconstrained':
                line_style = '-.'
                alpha = 0.8
            elif name == 'w5e5_gan_isimip':
                line_style = ':'
                alpha = 0.7
            else:
                line_style = '-'
                alpha = 1.0
            
            if name == 'w5e5_5':
                label = r'W5E5v2 ($2^\circ$ resolution)'
            else:
                label = f'{self.plot_data.model_name_definition(name)}'

            line, = ax.plot(x_vals, self.spectrum[name],
                     label=label,
                     linestyle=line_style,
                     alpha=alpha,
                     color=colors(name),
                     linewidth=linewidth)

            if (name == 'w5e5_gan_unconstrained'):
                line.set_dashes([5, 9])
    
        ax.legend(loc='lower left', fontsize=fontsize)
        ax.set_xlim(x_vals[1]+1024, x_vals[-1]-32)
        ax.set_yscale('log', base=10)
        ax.set_ylim(5e-7, 5e-2)
        ax.set_xscale('log', base=2)
        ax.set_xticks([2**7, 2**8, 2**9, 2**10, 2**11, 2**12, 2**13], fontsize=fontsize)
        ax.tick_params(axis='x', labelsize=fontsize)
        ax.tick_params(axis='y', labelsize=fontsize)
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        ax.grid()
        #ax.set_ylim(4.5e-5, 0.07)
        ax.set_xlabel(r'Wavelength [km]', fontsize=fontsize)
        ax.set_ylabel('PSD [a.u]', fontsize=fontsize)
        
        if fname is not None:
            plt.savefig(fname, format='pdf', bbox_inches='tight')