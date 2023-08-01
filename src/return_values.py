import sys 
sys.path.append("..") 
import os
os.environ['PROJ_LIB'] = '/Users/mb/anaconda3/envs/worklab/share/proj'

import numpy as np
import xarray as xr
from tqdm.notebook import tqdm
from scipy.stats import norm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from src.configuration import HistoricalConfig
from src.inference import LoadData
import src.xarray_utils as xu
from src.plots import plot_basemap


def main():

    model_name = 'w5e5'
    #model_name = 'gfdl'
    #model_name = 'w5e5_gan'
    #model_name = 'w5e5_gan_isimip'
    #model_name = 'w5e5_gan_unconstrained'
    #model_name = 'custom_isimip'
    print("processing", model_name)

    fname = f'/p/tmp/hess/scratch/cmip-gan/results/{model_name}_extremes.nc'

    config = HistoricalConfig()
    config.test_period = ('2004', '2014')
    data = LoadData(config).collect_historical_data()

    time_series = getattr(data, model_name).load()

    print("step 1")

    pot_time_series = get_peak_over_threshold(time_series)

    print("step 2")

    gamma, confidence, return_values = get_global_stats(pot_time_series)

    print("step 3")

    gamma = xr.DataArray(
                    data=gamma,
                    name="gamma",
                    dims=["latitude", "longitude"],
                    coords=dict(longitude=pot_time_series.longitude.values,
                                latitude=pot_time_series.latitude.values))

    confidence = xr.DataArray(
                    data=confidence,
                    name="confidence",
                    dims=["latitude", "longitude"],
                    coords=dict(longitude=pot_time_series.longitude.values,
                                latitude=pot_time_series.latitude.values))

    return_values = xr.DataArray(
                    data=return_values,
                    name="return_values",
                    dims=["latitude", "longitude"],
                    coords=dict(longitude=pot_time_series.longitude.values,
                                latitude=pot_time_series.latitude.values))

    dataset = xr.merge([gamma, confidence, return_values])
    xu.write_dataset(dataset, fname)


def get_peak_over_threshold(time_series: xr.DataArray, min_threshold=0.1):
    
    ts_thresholded_times = time_series.where(time_series > min_threshold, 0)
    ts_wet_times = time_series.where(time_series > min_threshold, np.NaN)
    pot_threshold_quantile = 0.95
    pot_threshold = ts_wet_times.load().quantile(pot_threshold_quantile, dim="time", skipna=True)
    ts_pot = ts_thresholded_times.where(ts_thresholded_times > pot_threshold, np.NaN)
    
    return ts_pot


def moment_estimator(data: xr.DataArray):
    """ 
    data :: a 1D array containing extremes over a threshold ordered according
    to their size
    """

    n = len(data)
    k = n-2
    m1 = 0
    m2 = 0

    for i in range(0,k+1):   # i = 0, ..., k
        m1 += np.log(data[n-1-i]) - np.log(data[n-2-k])
        m2 += (np.log(data[n-1-i]) - np.log(data[n-2-k]))**2

    m1 = (1 / (k+1)) * m1 # add 1 to k since it starts at zero
    m2 = (1 / (k+1)) * m2 # add 1 to k since it starts at zero

    gamma = m1 + 1 - 1/2*(1 - m1**2/m2)**-1

    sigma = 0.5*data[k]*m1*(1 - m1**2/m2)**-1

    U_r0 = data[n-2-k]
  
    return gamma, sigma, U_r0


def make_events_unique(time_series: np.ndarray):
    
    events = []
    event_store = 0
    
    for i in range(len(time_series)-1):
        
        event = time_series[i]
        next_event = time_series[i+1]
            
        if event > 0: # is inside event series
            if event > next_event and event_store < event :
                event_store = event
                
            if next_event == 0: # at the end of event series
                events.append(event_store)
                event_store = 0
            
    return events


def dist(r,k,n,gamma,x_k):
    return x_k*(r*k/n)**gamma


def U_func(return_time, U_r0, gamma, sigma, k, n):
    return U_r0 + sigma *((return_time*k/n)**gamma - 1)/gamma


def get_global_stats(pot_time_series: xr.DataArray):
    
    gammas = np.zeros((len(pot_time_series.latitude),  len(pot_time_series.longitude)))
    confidence = np.zeros((len(pot_time_series.latitude),  len(pot_time_series.longitude)))
    return_values = np.zeros((len(pot_time_series.latitude),  len(pot_time_series.longitude)))
    
    for i, lat in enumerate(tqdm(pot_time_series.latitude)):
        for j, lon in enumerate(pot_time_series.longitude):
            events = pot_time_series.sel(latitude=lat, longitude=lon).dropna(dim="time")
            events = events.sortby(events) 
    
            if len(events) > 5:
                gamma, sigma, U_r0 = moment_estimator(events)
                gammas[i, j] = gamma

                alpha = 0.05 # 1 - confidence
                q = norm.ppf(1-alpha/2)
                n = len(events)
                k = n-2
                delta = q * gamma / np.sqrt(k)
                confidence[i, j] = gamma - delta

                return_time = 10
                return_values[i, j] = U_func(return_time, U_r0, gamma, sigma, k, n)


            else: 
                gammas[i, j] = -999
                confidence[i, j] = -999
                return_values[i, j] = -999
    
    return gammas, confidence, return_values


def plot_return_value_statistics(names,
                             path,
                             datasets,
                             return_value_results,
                             figsize,
                             fname_figure,
                            ):
    num_rows = 3
    num_cols = 2
    fig, axs = plt.subplots(3, 2, figsize=(14,16), constrained_layout=True)
    plt.rcParams.update({'font.size': 15, 'axes.titlesize': 16, 'legend.title_fontsize': 16})

    letters = ['a', 'b', 'c', 'd', 'e', 'f']
    alpha = 0.8 
    cmap= 'seismic'
    col = 0
    vmin, vmax =  -300, 300
    index = 0
    
    for row in range(3):
        for col in range(2):
    
            ax = axs[row, col]
            
            ax.annotate(f"{letters[index]}", ha="center", va="center", size=15,
                     xy=(1-0.955, 0.925), xycoords=ax,
                     bbox=dict(boxstyle="square, pad=0.25", fc="white", ec="k", lw=1)) 
            model_name_definition = datasets.model_name_definition(names[index])
            
            if names[index] == 'w5e5':
                cmap = 'viridis_r'
                vmin = 0
                vmax = 250
                alpha = 1.0
                ax.set_title(f'{model_name_definition}')
            else:
                cmap = 'seismic'
                vmin = -300
                vmax = 300
                alpha = 1.0
                global_mae =  return_value_results[names[index]]['global_mae']
                if names[index] == 'w5e5_gan_unconstrained':
                    title = f'{model_name_definition}'
                    mae_title = r"MAE=$\bf{" + f"{global_mae.values.mean():2.3f}"+ "}$"
                    
                else:
                    
                    title = f'{model_name_definition}'
                    mae_title = f'MAE={global_mae.values.mean():2.3f}'
                ax.set_title(title)
                
                ax.annotate(mae_title, ha="center", va="center", size=15,
                             xy=(0.850, 0.925), xycoords=ax,
                             bbox=dict(boxstyle="round, pad=0.35", alpha=0.8,fc="white", ec="k", lw=1)) 
                
            
            if index % 2 == 0:
                parallel_label = [1,0,0,0]
                meridian_label = [0,0,0,0]
                
            if index % 2 != 0:
                parallel_label = [0,0,0,0]
                meridian_label = [0,0,0,0]
                
            if index == num_rows*2-1:
                parallel_label = [0,0,0,0]
                meridian_label = [0,0,0,1] 
                
            if index == num_rows*2-2:
                parallel_label = [1,0,0,0]
                meridian_label = [0,0,0,1] 
                 
            if names[index] == 'w5e5':
                mask = None
            else:
                mask= return_value_results[names[index]]['mask']
                 
            if names[index] == 'w5e5':
                fname = f'{path}/w5e5_extremes.nc'
                result = xr.open_dataset(fname).return_values
            else:
                result = return_value_results[names[index]]['error']
                result = xr.where(mask, result, np.NaN)
                         
            cs = plot_basemap(result, '', vmin, vmax, alpha, cmap,
                         cbar=False,
                         axis=ax,
                         return_cs=True,
                         projection='mill',
                         map_resolution='c',
                         coastline_linewidth=0.25,
                         parallel_label=parallel_label,
                         meridian_label=meridian_label,
                         draw_coordinates=True,
                         plot_mask=True,
                         mask=mask,
                         mask_hatch='///')
            
            if names[index] == 'w5e5':
                cbaxes = inset_axes(ax, width="30%", height="3%", loc=4,
                                     bbox_to_anchor=(-0.05, 0.11, 1, 1), 
                                     bbox_transform=ax.transAxes) 
                cbar = plt.colorbar(cs, cax=cbaxes,
                                 orientation='horizontal',
                                 extend='max',
                                 label="Return values [mm/d]",
                                  ticks=[0,100,200])
                ticklabs = cbar.ax.get_yticklabels()
                cbar.ax.set_yticklabels(ticklabs, fontsize=6)
            index += 1
    fig.colorbar(cs, ax=axs[2, :], shrink=0.4, label="Return value error [mm/d]",
                    orientation='horizontal', extend='both')

    plt.savefig(fname_figure, format='png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()