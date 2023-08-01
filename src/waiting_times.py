import sys 
sys.path.append("..") 
import os
os.environ['PROJ_LIB'] = '/Users/mb/anaconda3/envs/worklab/share/proj'

import xarray as xr
import numpy as np
from scipy.stats import kstest, expon, gamma
import functools
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from src.inference import LoadData
from src.configuration import HistoricalConfig
import src.xarray_utils as xu
from scipy.stats import wasserstein_distance
from src.plots import plot_basemap


def count_waiting_times(time_series: np.ndarray):
    
    counts = []
    count = 1
    
    for i in range(len(time_series)-1):
        
        event = time_series[i]
        next_event = time_series[i+1]
            
        if event == 0: 
            if next_event == 0 :
                count += 1
                
            if next_event == 1: # at the end of event series
                counts.append(count)
                count = 1
            
    return counts


def get_binary_from_threshold(time_series: xr.DataArray, min_threshold=0.1):
    
    ts_thresholded_times = time_series.where(time_series > min_threshold, 0)
    ts_wet_times = time_series.where(time_series > min_threshold, np.NaN)
    pot_threshold_quantile = 0.95
    pot_threshold = ts_wet_times.load().quantile(pot_threshold_quantile, dim="time", skipna=True)
    ts_pot = xr.where(ts_thresholded_times > pot_threshold, 1, 0)
    
    return ts_pot


def get_gamma_pvalues(waiting_times):

    if len(waiting_times)>5:
        theta = np.mean(waiting_times*np.log(waiting_times)) - np.mean(waiting_times)*np.mean(np.log(waiting_times))
        k = np.mean(waiting_times)/theta

        cdf = functools.partial(gamma.cdf, a=k, scale=theta)
        result = kstest(waiting_times, cdf)
        pv = result.pvalue
    else:
        pv = k = theta = -999
    
    return pv, k, theta


def compute_waiting_times_statistics():

    model_name = 'w5e5'
    #model_name = 'gfdl'
    #model_name = 'w5e5_gan'
    #model_name = 'w5e5_gan_isimip'
    #model_name = 'w5e5_gan_unconstrained'
    #model_name = 'custom_isimip'

    print("processing", model_name)

    fname = f'/p/tmp/hess/scratch/cmip-gan/results/{model_name}_waiting_times.nc'

    config = HistoricalConfig()
    config.test_period = ('2004', '2014')
    data = LoadData(config).collect_historical_data()

    model = getattr(data, model_name).load()

    result = np.zeros((4, len(model.latitude), len(model.longitude)))

    for lat in range(len(model.latitude)):
        print(lat)
        for lon in range(len(model.longitude)):
        
            tmp = model.isel(latitude=lat, longitude=lon)
            tmp = get_binary_from_threshold(tmp)
            waiting_times = count_waiting_times(tmp)
            
            result[0, lat, lon], result[1, lat, lon], result[2, lat, lon] = get_gamma_pvalues(waiting_times)
            result[3, lat, lon] = len(waiting_times)

    results_dataset = xr.Dataset(
        data_vars=dict(
            p_values=(["latitude", "longitude"], result[0]),
            k=(["latitude", "longitude"], result[1]),
            theta=(["latitude", "longitude"], result[2]),
            num_events=(["latitude", "longitude"], result[3]),
        ),

        coords=dict(longitude=model.longitude.values,
                    latitude=model.latitude.values))

    xu.write_dataset(results_dataset, fname)


def compute_wasserstein_distance():

    target_name = 'w5e5'

    model_name = 'gfdl'
    #model_name = 'w5e5_gan'
    #model_name = 'w5e5_gan_isimip'
    #model_name = 'w5e5_gan_unconstrained'
    #model_name = 'custom_isimip'

    print("processing", model_name)

    fname = f'/p/tmp/hess/scratch/cmip-gan/results/{model_name}_waiting_times_wasserstein_distance.nc'

    config = HistoricalConfig()
    config.test_period = ('2004', '2014')
    data = LoadData(config).collect_historical_data()

    model = getattr(data, model_name).load()
    target = getattr(data, target_name).load()

    result = np.zeros((1, len(model.latitude), len(model.longitude))) - 999

    for lat in range(len(model.latitude)):
        print(lat)
        for lon in range(len(model.longitude)):
        
            tmp = model.isel(latitude=lat, longitude=lon)
            tmp = get_binary_from_threshold(tmp)
            model_waiting_times = count_waiting_times(tmp)

            tmp = target.isel(latitude=lat, longitude=lon)
            tmp = get_binary_from_threshold(tmp)
            target_waiting_times = count_waiting_times(tmp)

            if len(target_waiting_times) > 5 and len(model_waiting_times) > 5:
                result[0, lat, lon] = wasserstein_distance(np.array(target_waiting_times),
                                                           np.array(model_waiting_times))
                                                     

    results_dataset = xr.Dataset(
        data_vars=dict(
            ws_distance=(["latitude", "longitude"], result[0]),
        ),

        coords=dict(longitude=model.longitude.values,
                    latitude=model.latitude.values))

    xu.write_dataset(results_dataset, fname)


def plot_waiting_times_wassserstein_distance(names,
                                             datasets,
                                             waiting_times_results,
                                             event_threshold,
                                             fname_figure
                                            ):
    num_rows = 3
    num_cols = 2
    fig, axs = plt.subplots(3, 2, figsize=(14,16), constrained_layout=True)
    plt.rcParams.update({'font.size': 15, 'axes.titlesize': 16, 'legend.title_fontsize': 16})
    letters = ['a', 'b', 'c', 'd', 'e', 'f']
    alpha = 0.8 

    index = 0
    
    col = 0
    
    for row in range(3):
        for col in range(2):
    
            ax = axs[row, col]
            
            ax.annotate(f"{letters[index]}", ha="center", va="center", size=15,
                     xy=(1-0.955, 0.925), xycoords=ax,
                     bbox=dict(boxstyle="square, pad=0.25", fc="white", ec="k", lw=1)) 
            model_name_definition = datasets.model_name_definition(names[index])
            
            if names[index] == 'w5e5':
                cmap = 'viridis_r'
                vmin, vmax = 0, 200
                alpha = 1.0
                ax.set_title(f'{model_name_definition}')
            else:
                cmap = 'plasma'
                vmin, vmax = 0, 30
                alpha = 1.0
                global_mae =  xr.where(waiting_times_results['w5e5'] > event_threshold,
                                       waiting_times_results[names[index]],
                                       np.NaN)
                if names[index] in ['w5e5_gan_isimip']:
                    title = f'{model_name_definition}'
                    mae_title = r"MAE=$\bf{" + f"{global_mae.mean(skipna=True).values:2.3f}"+ "}$"
                    
                else:
                    title = f'{model_name_definition}'
                    mae_title = f'MAE={global_mae.mean(skipna=True).values:2.3f}'
                
                ax.annotate(mae_title, ha="center", va="center", size=15,
                             xy=(0.850, 0.925), xycoords=ax,
                             bbox=dict(boxstyle="round, pad=0.35", alpha=0.8,fc="white", ec="k", lw=1)) 
                ax.set_title(title)
            
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
                result = waiting_times_results[names[index]]
            else:
                mask = xr.where(waiting_times_results['w5e5'] > event_threshold, 1, 0) 
                result = xr.where(waiting_times_results['w5e5'] > event_threshold,
                                  waiting_times_results[names[index]],
                                  np.NaN)

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
                         mask=mask,
                         mask_hatch='///')
            
            if names[index] == 'w5e5':
                cbaxes = inset_axes(ax, width="30%", height="3%", loc=4,
                                     bbox_to_anchor=(-0.05, 0.11, 1, 1), 
                                     bbox_transform=ax.transAxes) 
                cbar = plt.colorbar(cs, cax=cbaxes,
                                 orientation='horizontal',
                                 extend='max',
                                   )
                ticklabs = cbar.ax.get_yticklabels()
                cbar.ax.set_yticklabels(ticklabs, fontsize=6)
                cb_color = 'b'
                cbar.set_label('Number of events', color=cb_color)
                cbar.ax.tick_params(color=cb_color, labelcolor=cb_color)
                cbar.outline.set_edgecolor(cb_color)
                plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=cb_color)
                
                
            index += 1
    cbar = fig.colorbar(cs, ax=axs[2, :], shrink=0.4, label="Relative Wasserstein distance [days]",
                    orientation='horizontal', extend='max')
    cbar.ax.locator_params(nbins=7)

    plt.savefig(fname_figure, format='png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    compute_wasserstein_distance()
