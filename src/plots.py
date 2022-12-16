from selectors import EpollSelector
from turtle import position
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D
from src.data import TestData
from src.preproc import add_land_sea_mask
import src.xarray_utils as xu


""" Implements all plotting and analysis function for model evalution """
class PlotAnalysis():
    
    def __init__(self, data: TestData, names: list = None):
        
        self.data = data
        self.names = names

    """ Plot single model output fields for a qualitative comparison """
    def single_frames(self, 
                              vmin=0,
                              vmax=20,
                              time_index=-1,
                              cmap='Blues',
                              mask=False,
                              single_plot=False):
        num_rows = 4
        num_cols = 1
        fig, axs = plt.subplots(num_rows,num_cols,figsize=(7,16),  constrained_layout=True)
        alpha = 1.0 

        #name = ['target', 'gfdl', 'gan', 'quantile_mapping', 'gan_qm',  'gfdl_isimip']
        name = self.names

        letters = ['a', 'b', 'c', 'd', 'e', 'f']

        i = 0
        for row in range(num_rows):
            ax = plt.subplot(num_rows*100+11+i)

            data = abs(getattr(self.data, name[i]).isel(time=time_index))
            if i == 0: print(data.time.values)

            ax.annotate(f"{letters[i]}", ha="center", va="center", size=15,
                     xy=(1-0.955, 0.925), xycoords=ax,
                     bbox=None) 
            plt.title(self.data.model_name_definition(name[i]))

            cbar = False
            cbar_title = ''
                 

            cs = plot_basemap(data, cbar_title, vmin, vmax, alpha, cmap,
                         cbar=cbar,
                         axis=ax,
                         return_cs=True,
                         projection='robin',
                         map_resolution='c',
                         coastline_color='lightgrey',
                         coastline_linewidth=0.25,
                         plot_mask=mask)
            i += 1
                 
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(norm=norm, cmap = cmap)
        sm.set_array([])


    """ Plot single model output fields for a local region and
        qualitative comparison
    """
    def single_frames_local(self, 
                              vmin=0,
                              vmax=20,
                              time_index=-1,
                              cmap='Blues',
                              mask=False,
                              single_plot=False,
                              lat_min=None, 
                              lat_max=None,
                              lon_min=None, 
                              lon_max=None,
                              ):
        num_rows = 2
        fig, axs = plt.subplots(num_rows, 2,figsize=(9.1,9),  constrained_layout=True)
        alpha = 1.0 

        name = self.names

        letters = ['a', 'b', 'c', 'd', 'e', 'f']

        i = 0
        for col in range(2):
            for row in range(num_rows):
                ax = axs[row, col]

                data = abs(getattr(self.data, name[i]).isel(time=time_index))
                data = data.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))
                if i == 0: print(data.time.values)

                ax.annotate(f"{letters[i]}", ha="center", va="center", size=15,
                         xy=(1-0.955, 0.925), xycoords=ax,
                         bbox=None) 
                ax.set(title=self.data.model_name_definition(name[i]))

                cbar = False
                cbar_title = ''

                cs = plot_basemap(data, cbar_title, vmin, vmax, alpha, cmap,
                             cbar=cbar,
                             axis=ax,
                             return_cs=True,
                             projection='mill',
                             draw_coordinates=True,
                             map_resolution='c',
                             draw_countries=True,
                             plot_mask=mask)
                i += 1
        fig.colorbar(cs, ax=axs[:, 1], shrink=0.5, label="Precipitation [mm/d]", extend='max')


    def int2string(self, X):
        return ["%3i" % x for x in X]


    def float2string(self, X):
        strings = []
        decimal_points = [0,1,1,2,2,3,3]
        for i,x in enumerate(X):
            strings.append(f'%.{decimal_points[i]}f' % x)
        return strings

    """ Compute and plot relative frequency histograms """    
    def histograms(self,
                    single_plot=False,
                    bins=200,
                    ax=None,
                    show_legend=True,
                    show_xlabels=False,
                    latitude_bounds=None,
                    longitude_bounds=None,
                    time_bounds=None,
                    annotate=True,
                    xlim=600,
                    ylim=[1e-4, 1],
                    land_masked=False,
                    sea_masked=False,
                    return_histogram=False,
                    min_precipitation_threshold = None,
                    max_precipitation_threshold = None
                    ):

        if single_plot:
            plt.figure(figsize=(6,4))

        ax.set_ylim(ylim[0], ylim[1])
        ax2 = ax.twiny()
        ax2.set_ylim(ylim[0], ylim[1])

        histogram = None
        for name in reversed(self.names):
            print(name)

            if name == 'w5e5' or name == 'w5e5': 
                linestyle = '-' 
                alpha = 1
            elif name == 'w5e5_gan':
                alpha = .7
            else:
                linestyle = '-' 
                alpha = .6

            data = getattr(self.data, name)
            if latitude_bounds is not None:
                data = data.sel(latitude=slice(latitude_bounds[0], latitude_bounds[1]))
            if longitude_bounds is not None:
                data = data.sel(longitude=slice(longitude_bounds[0], longitude_bounds[1]))
            if time_bounds is not None:
                data = data.sel(time=slice(time_bounds[0], time_bounds[1]))
            if land_masked is True:
                add_land_sea_mask(data)
                data = data.where(data.mask==1, drop=True)
            if sea_masked is True:
                add_land_sea_mask(data)
                data = data.where(data.mask==0, drop=True)
            if min_precipitation_threshold is not None:
                data = data.where(data > min_precipitation_threshold, drop=True)
            if max_precipitation_threshold is not None:
                data = data.where(data < max_precipitation_threshold, drop=True)

            data = data.values.flatten()


            label = self.data.model_name_definition(name)
            hist_results = plt.hist(data,
                         bins=bins,
                         histtype='step',
                         log=True,
                         label=label,
                         alpha=alpha,
                         density=True,
                         linewidth=2,
                         linestyle=linestyle,
                         color=self.data.colors(name))
            if name == 'w5e5': 
                histogram = hist_results

        if ax is not None and annotate is True:
           ax.annotate("b", ha="center", va="center", size=15,
                     xy=(1-0.955, 0.925), xycoords=ax,
                     bbox=None) 

        xticks_positions = np.array([0, 25, 50, 75, 100, 125, 150])
        for tic in ax.xaxis.get_major_ticks():
            tic.tick1On = tic.tick2On = False
            tic.tick1line.set_visible(False)
            tic.tick2line.set_visible(False)
            tic.label1.set_visible(False)
            tic.label2.set_visible(False)
        ax.set_xticks(xticks_positions)

        if show_xlabels:
            ax.set_xticks(xticks_positions)
            ax.set_xlabel(r"Precipitation [mm/d]")
            ax.set_xticklabels(self.int2string(xticks_positions)) 
        else:
            ax.set_xticklabels([]) 
        ax.set_xlim(0, xlim)
        ax.set_ylim(ylim[0], ylim[1])
        ax.grid(True)

        percentiles = []
        for value in xticks_positions:
            print(f'computing percetile of {value}')
            percentiles.append(xu.get_percentile_of_value_above_threshold(self.data.w5e5, 0.1, value))

        ax2.set_xticks(xticks_positions)
        ax2.set_xticklabels(self.float2string(percentiles)) 
        ax2.set_xlabel(r"W5E5v2 precipitation percentiles", labelpad = 10)
        ax2.set_xlim(ax.get_xlim())
        ax2.set_ylim(ax.get_ylim())

        handles, labels = ax2.get_legend_handles_labels()
        new_handles = [Line2D([], [], c=h.get_edgecolor()) for h in handles]

        plt.legend(handles=new_handles, labels=labels)
        ax.set_ylabel('Histogram')


        ax.set_ylim(ylim[0], ylim[1])
        ax2.set_ylim(ylim[0], ylim[1])

        if single_plot:
            plt.show()
        if return_histogram is True:
            return histogram


    """ Compute and plot the distance (i.e. error) between 
        relative frequency histograms
    """    
    def histogram_distances(self,
                    log=True,
                    bin_start=0,
                    bin_end=151,
                    latitude_bounds=None,
                    longitude_bounds=None,
                    time_bounds=None,
                    land_masked=False,
                    sea_masked=False,
                    min_precipitation_threshold=None,
                    max_precipitation_threshold=None,
                    season=None):

        bins =  np.arange(bin_start, bin_end)
        data = getattr(self.data, 'w5e5')
        target_bin_values = np.histogram(data, bins=bins, density=True)[0]

        self.names.remove('w5e5')
        for name in reversed(self.names):
            print(name)

            data = getattr(self.data, name)
            if latitude_bounds is not None:
                data = data.sel(latitude=slice(latitude_bounds[0], latitude_bounds[1]))
            if longitude_bounds is not None:
                data = data.sel(longitude=slice(longitude_bounds[0], longitude_bounds[1]))
            if time_bounds is not None:
                data = data.sel(time=slice(time_bounds[0], time_bounds[1]))
            if land_masked is True:
                add_land_sea_mask(data)
                data = data.where(data.mask==1, drop=True)
            if sea_masked is True:
                add_land_sea_mask(data)
                data = data.where(data.mask==0, drop=True)
            if min_precipitation_threshold is not None:
                data = data.where(data > min_precipitation_threshold, drop=True)
            if max_precipitation_threshold is not None:
                data = data.where(data < max_precipitation_threshold, drop=True)
            if season is not None:
                data = data.where(data['time'].dt.season==season, drop=True)

            data = data.values.flatten()

            bin_values = np.histogram(data, bins=bins, density=True)[0]
            differences = abs(target_bin_values - bin_values)
            label = self.data.model_name_definition(name)

            plt.plot(np.arange(0, len(differences)),
                     differences,
                     label=label,
                     color=self.data.colors(name),
                     linewidth=2)

            plt.ylabel('Absolute error')
            plt.xlabel(r"Precipitation [mm/d]")
            plt.xlim(0, 150)
            plt.ylim(1e-8, 1e-1)

            xticks_positions = np.array([0, 25, 50, 75, 100, 125, 150])
            plt.xticks(xticks_positions)
            if log:
                plt.yscale('log')
            #plt.legend()
            plt.grid()


    """ Compute and plot the mean over time and longitudes """    
    def latitudinal_mean(self, single_plot=False, ax=None, show_legend=True, annotate=True):

        if hasattr(self.data, 'target'):
            if self.data.target is not None:
                target = self.data.target
        if hasattr(self.data, 'w5e5'):
            if self.data.w5e5 is not None:
                target = self.data.w5e5
        assert(target is not None)

        
        for name in reversed(self.names):
            alpha = 1
            if name == 'w5e5' or name == 'w5e5': 
                linestyle = '-' 
            else:
                linestyle = '-' 

            data = getattr(self.data, name)
            data = data.mean(dim=("longitude", "time"))
            mae = abs(data - target.mean(dim=("longitude", "time"))).mean().values

            if name == 'w5e5':
                label = f'{self.data.model_name_definition(name)}'
            elif name == 'w5e5_gan_isimip':

                label = f'{self.data.model_name_definition(name)}: '+r"MAE = $\bf{" + f"{mae:2.3f}" + "}$"
            else:
                label = f'{self.data.model_name_definition(name)}: MAE = {mae:2.3f}'

            plt.plot(data.latitude,
                     data,
                     label=label,
                     alpha=alpha,
                     linestyle=linestyle,
                     linewidth=2,
                     color=self.data.colors(name))
        
        if ax is not None and annotate is True:
           ax.annotate("a", ha="center", va="center", size=15,
                     xy=(1-0.955, 0.925), xycoords=ax,
                     bbox=None) 

        plt.ylim(0,8)
        plt.xlim(-90,90)
        plt.xlabel(r'Latitude')

        xn = list(np.arange(-80, 0, 20))
        slabels = [f'{abs(l)}'+r'$^\circ$S' for l in xn]
        xs = list(np.arange(20, 100, 20))
        nlabels = [f'{l}'+r'$^\circ$N' for l in xs]
        x0 = [0]
        zerolabel = [r'0$^\circ$']
        labels = slabels + zerolabel + nlabels
        x = xn + x0 + xs
        plt.xticks(x, labels)
        plt.ylabel('Mean precipitation [mm/d]')
        plt.grid()

        if show_legend:
            plt.legend(loc='upper left')

        if single_plot:
            plt.show()

        
    def latitudinal_mean_and_histograms(self, names=None, single_plot=False):
        if names is not None:
            self.names = names

        ax = None
        self.latitudinal_mean(single_plot=single_plot, ax=ax,
                                          show_legend=False )

        ax = plt.subplot(1,1,1)
        self.histograms(single_plot=single_plot, ax=ax,
                                     show_legend=False)


    """ Compute and plot the mean over time for each grid cell """    
    def bias(self,
              fname,
              season=None):

        fig, axs = plt.subplots(3, 2, figsize=(14,16), constrained_layout=True)
        plt.rcParams.update({'font.size': 15, 'axes.titlesize': 16, 'legend.title_fontsize': 16})


        letters = ['a', 'b', 'c', 'd', 'e', 'f']
        
        i = 0
        for row in range(3):
            for col in range(2):

                ax = axs[row, col]
                # excluding target form self.names
                target = getattr(self.data, 'w5e5')

                name = self.names[i]
                data = getattr(self.data, name)
                label = self.data.model_name_definition(name)

                if season is not None:
                    data = data.where(data['time'].dt.season==season, drop=True)
                    target = target.where(target['time'].dt.season==season, drop=True)

                if name == 'w5e5':
                    bias = data.mean('time')
                    cmap = 'viridis_r'
                    vmin = 0
                    vmax = 10
                    alpha = 0.5

                    title = f'{label} precipitation [mm/d]'
                    ax.set_title(title)
                else:
                    cmap = 'seismic'
                    bias = data.mean('time') - target.mean('time') 
                    vmin = -8.5
                    vmax = 8.5
                    alpha = 0.8
                    if name == 'w5e5_gan_isimip':
                        title = f'{label}: '+r"MAE = $\bf{" + f"{abs(bias).values.mean():2.3f}" + "}$"
                    else:
                        title = f'{label}: MAE = {abs(bias).values.mean():2.3f}'
                    ax.set_title(title)

                print(label,f" \t \t MAE: {abs(bias).values.mean():2.3f} [mm/d]")

                ax.annotate(f"{letters[i]}", ha="center", va="center", size=15,
                     xy=(1-0.955, 0.925), xycoords=ax,
                     bbox=dict(boxstyle="square, pad=0.25", fc="white", ec="k", lw=1)) 

                if i == 0:
                    parallel_label = [1,0,0,0]
                    meridian_label = [0,0,0,0]
                elif i == 1:
                    parallel_label = [0,0,0,0]
                    meridian_label = [0,0,0,0]
                elif i == 2:
                    parallel_label = [1,0,0,0]
                    meridian_label = [0,0,0,0]
                elif i == 3:
                    parallel_label = [0,0,0,0]
                    meridian_label = [0,0,0,0]
                elif i == 4:
                    parallel_label = [1,0,0,0]
                    meridian_label = [0,0,0,1]
                elif i == 5:
                    parallel_label = [0,0,0,0]
                    meridian_label = [0,0,0,1]

                cs = plot_bias_basemap(bias, None, vmin, vmax, alpha, cmap,
                                 cbar=False,
                                 cbar_extend='both',
                                 parallel_label=parallel_label,
                                 meridian_label=meridian_label,
                                 axis=ax)

                if name == 'w5e5':
                    cbaxes = inset_axes(ax, width="30%", height="3%", loc=4,
                                         bbox_to_anchor=(-0.05, 0.06, 1, 1), 
                                         bbox_transform=ax.transAxes) 

                    cbar = plt.colorbar(cs, cax=cbaxes,
                                     orientation='horizontal',
                                     extend='max',
                                     label=None)

                    ticklabs = cbar.ax.get_yticklabels()
                    cbar.ax.set_yticklabels(ticklabs, fontsize=6)

                i += 1

        fig.colorbar(cs, ax=axs[2, :], shrink=0.4, label="Bias [mm/d]",
                    orientation='horizontal', extend='both')

        plt.savefig(fname, format='png', dpi=300, bbox_inches='tight')
        plt.show()
        

    """ Compute and plot the number of events above a threshold value
        for each grid cell
    """    
    def number_of_events_above_threshold(self,
                                        fname:str,
                                        threshold,
                                        vmin=-10,
                                        vmax=10,
                                        cmap="seismic",
                                        single_plot=False,
                                        month: int = None,
                                        season=None):


        fig, axs = plt.subplots(3, 2,figsize=(14,16),  constrained_layout=True)
        plt.rcParams.update({'font.size': 15, 'axes.titlesize': 16, 'legend.title_fontsize': 16})

        alpha = 0.8

        letters = ['a', 'b', 'c', 'd', 'e', 'f']
        
        i = 0
        for row in range(3):
            for col in range(2):

                ax = axs[row, col]

                name = self.names[i]
            
                data = getattr(self.data, name)
                if season is not None:
                    data = data.where(data['time'].dt.season==season, drop=True)

                if month is not None:
                    data = data.where(data['time'].dt.month==month, drop=True)
                data = data.load()

                event_count = xr.where(data > threshold, 1, 0).sum(dim='time')
                print(f'{name} {event_count.max().values}')
                event_count = xr.where(event_count > 0, event_count, np.NaN)

                label = self.data.model_name_definition(name)
                ax.set_title(f'{label}')

                ax.annotate(f"{letters[i]}", ha="center", va="center", size=15,
                     xy=(1-0.955, 0.925), xycoords=ax,
                     bbox=dict(boxstyle="square, pad=0.25", fc="white", ec="k", lw=1)) 

                cbar_title = f'Number of events above {threshold} [mm/d]'
                if i == 0:
                    parallel_label = [1,0,0,0]
                    meridian_label = [0,0,0,0]
                elif i == 1:
                    parallel_label = [0,0,0,0]
                    meridian_label = [0,0,0,0]
                elif i == 2:
                    parallel_label = [1,0,0,0]
                    meridian_label = [0,0,0,0]
                elif i == 3:
                    parallel_label = [0,0,0,0]
                    meridian_label = [0,0,0,0]
                elif i == 4:
                    parallel_label = [1,0,0,0]
                    meridian_label = [0,0,0,1]
                elif i == 5:
                    parallel_label = [0,0,0,0]
                    meridian_label = [0,0,0,1]

                cs = plot_bias_basemap(event_count, cbar_title, vmin, vmax, alpha, cmap,
                             cbar=False, cbar_extend='max',
                             parallel_label=parallel_label,
                             meridian_label=meridian_label,
                              axis=ax)
                i += 1


        fig.colorbar(cs, ax=axs[2, :], shrink=0.4, label="Number of events above 150 [mm/d]",
                    orientation='horizontal', extend='max')

        plt.savefig(fname, format='png', bbox_inches='tight')
        plt.show()


    """ Compute and plot the difference in distribution quantiles
        for each grid cell
    """    
    def extremes_bias(self,
                      target_name: str,
                      quantile: float,
                      fname: str,
                      vmin=-10,
                      vmax=10,
                      cmap="seismic",
                      single_plot=False,
                      month: int = None,
                      season=None):

        fig, axs = plt.subplots(3, 2, figsize=(14,16), constrained_layout=True)
        plt.rcParams.update({'font.size': 15, 'axes.titlesize': 16, 'legend.title_fontsize': 16})


        letters = ['a', 'b', 'c', 'd', 'e', 'f']
        
        i = 0
        for row in range(3):
            for col in range(2):

                ax = axs[row, col]
                target = getattr(self.data, 'w5e5')

                name = self.names[i]
                data = getattr(self.data, name)
                label = self.data.model_name_definition(name)

                data = getattr(self.data, name)
                if season is not None:
                    data = data.where(data['time'].dt.season==season, drop=True)
                    target = target.where(target['time'].dt.season==season, drop=True)

                if month is not None:
                    data = data.where(data['time'].dt.month==month, drop=True)
                    target = target.where(target['time'].dt.month==month, drop=True)

                data = data.load()
                target = target.load()

                if name == 'w5e5':
                    cmap = 'viridis_r'
                    vmin = 0
                    vmax = 50
                    alpha = 0.5
                    ax.set_title("W5E5v2 95th percentile [mm/d]")
                    bias = data.quantile(quantile, dim='time')
                    label = self.data.model_name_definition(name)

                else:
                    cmap = 'seismic'
                    bias = data.mean('time') - target.mean('time') 
                    vmin = -20
                    vmax = 20
                    alpha = 0.8
                    ax.set_title(label)

                    target_extremes = target.quantile(quantile, dim='time')
                    data_extremes = data.quantile(quantile, dim='time')

                    bias = data_extremes - target_extremes

                    global_error = abs(bias).mean().values
                    print(f'{name} {global_error:2.3f}')
                    label = self.data.model_name_definition(name)

                    if name == 'w5e5_gan_isimip':
                        title = f'{label}: '+r"MAE = $\bf{" + f"{global_error:2.3f}" + "}$"
                    else:
                        title = f'{label}: MAE = {global_error:2.3f}'
                    ax.set_title(title)

                ax.annotate(f"{letters[i]}", ha="center", va="center", size=15,
                     xy=(1-0.955, 0.925), xycoords=ax,
                     bbox=dict(boxstyle="square, pad=0.25", fc="white", ec="k", lw=1)) 

                if i == 0:
                    parallel_label = [1,0,0,0]
                    meridian_label = [0,0,0,0]
                elif i == 1:
                    parallel_label = [0,0,0,0]
                    meridian_label = [0,0,0,0]
                elif i == 2:
                    parallel_label = [1,0,0,0]
                    meridian_label = [0,0,0,0]
                elif i == 3:
                    parallel_label = [0,0,0,0]
                    meridian_label = [0,0,0,0]
                elif i == 4:
                    parallel_label = [1,0,0,0]
                    meridian_label = [0,0,0,1]
                elif i == 5:
                    parallel_label = [0,0,0,0]
                    meridian_label = [0,0,0,1]


                cbar_title = f'Differences in the {int(quantile*100)}th percentile [mm/d]'

                cs = plot_bias_basemap(bias, cbar_title, vmin, vmax, alpha, cmap,
                                 cbar=False,
                                 cbar_extend='max',
                                 parallel_label=parallel_label,
                                 meridian_label=meridian_label,
                                  axis=ax)

                i += 1

                if name == 'w5e5':
                    cbaxes = inset_axes(ax, width="30%", height="3%", loc=4,
                                         bbox_to_anchor=(-0.05, 0.06, 1, 1), 
                                         bbox_transform=ax.transAxes) 

                    cbar = plt.colorbar(cs, cax=cbaxes,
                                     orientation='horizontal',
                                     extend='max',
                                     label=None)

                    ticklabs = cbar.ax.get_yticklabels()
                    cbar.ax.set_yticklabels(ticklabs, fontsize=6)

        fig.colorbar(cs, ax=axs[2, :], shrink=0.4, label=cbar_title,
                    orientation='horizontal', extend='both')
        plt.savefig(fname, format='png', dpi=300, bbox_inches='tight')
        plt.show()


    """ Compute and plot the distribution quantiles for each grid cell """    
    def spatial_quantiles(self,
                        target_name: str,
                        quantile,
                        vmin=-10,
                        vmax=10,
                        cmap="seismic",
                        single_plot=False,
                        month: int = None,
                        season=None):

        alpha = 0.8

        letters = ['a', 'b', 'c', 'd']
        target = getattr(self.data, target_name)
        
        count = 1
        for i,name in enumerate(self.names[:]):
            
            if name != 'gan':

                data = getattr(self.data, name)
                if season is not None:
                    data = data.where(data['time'].dt.season==season, drop=True)
                    target = target.where(target['time'].dt.season==season, drop=True)

                if month is not None:
                    data = data.where(data['time'].dt.month==month, drop=True)
                    target = target.where(target['time'].dt.month==month, drop=True)
                data = data.load()
                target = target.load()

                #bias = xr.where(data > threshold, 1, 0).sum(dim='time') \
                #      - xr.where(target > threshold, 1, 0).sum(dim='time')
                bias = data.quantile(quantile, dim='time')

                ax = plt.subplot(3,2,count)
                count += 1
                label = self.data.model_name_definition(name)
                plt.title(f'{label}')

                ax.annotate(f"{letters[i]}", ha="center", va="center", size=15,
                     xy=(1-0.955, 0.925), xycoords=ax,
                     bbox=dict(boxstyle="square, pad=0.25", fc="white", ec="k", lw=1)) 

                cbar_title = f'99th percentile [mm/d]'
                if i == 0:
                    parallel_label = [1,0,0,0]
                    meridian_label = [0,0,0,0]
                elif i == 1:
                    parallel_label = [0,0,0,0]
                    meridian_label = [0,0,0,0]
                elif i == 2:
                    parallel_label = [1,0,0,0]
                    meridian_label = [0,0,0,1]
                elif i == 3:
                    parallel_label = [0,0,0,0]
                    meridian_label = [0,0,0,1]

                if i == 0  or i == 2:
                    plot_bias_basemap(bias, cbar_title, vmin, vmax, alpha, cmap,
                                 cbar=False,
                                 cbar_extend='max',
                                 parallel_label=parallel_label,
                                 meridian_label=meridian_label,
                                  axis=ax)
                else:
                    plot_bias_basemap(bias, cbar_title, vmin, vmax, alpha, cmap,
                                 cbar=True, cbar_extend='max',
                                 parallel_label=parallel_label,
                                 meridian_label=meridian_label,
                                  axis=ax)


""" Basemap configuration for bias plotting """    
def plot_bias_basemap(data: xr.DataArray,
                cbar_title: str,
                vmin: float,
                vmax: float,
                alpha: float,
                cmap: str,
                cbar=True,
                cbar_position='right',
                cbar_extend='max',
                parallel_label=[1, 0, 0, 0],
                meridian_label=[0, 0, 0, 1],
                axis=None,
                contours=None,
                projection='mill',
                map_resolution='l'
                ):


        if axis is not None:
            import matplotlib.pyplot as plt
            cbar_plt = plt
            plt = axis

        lats = data.latitude
        lons = data.longitude

        lon_0 = 0

        m = Basemap(llcrnrlon=lons[0], llcrnrlat=lats[0],
                    urcrnrlon=lons[-1], urcrnrlat=lats[-1],
                    projection=projection, lon_0=lon_0, 
                    resolution=map_resolution, ax=axis)

        m.drawcoastlines()

        par = m.drawparallels(
                              [ -60, 0, 60 ],
                              #[-90, -45, 0, 45, 90],
                              linewidth=1.0,
                              labels=parallel_label,
                              color='grey')

        merid = m.drawmeridians(
                                #[ -90, 0, 90 ],
                                [-120, -60, 0, 60, 120],
                                linewidth=1.0,
                                labels=meridian_label,
                                color='grey')
    
        Lon, Lat = np.meshgrid(lons, lats)
    
        x, y = m(Lon, Lat)

        cs = m.pcolormesh(x, y, data, vmin=vmin, vmax=vmax,
                        alpha=alpha, cmap=cmap,
                        linewidth=0, shading='auto')

        m.drawcoastlines()

        if axis is None:
            ax = plt.gca()
            divider = make_axes_locatable(ax)
            cax = divider.append_axes(cbar_position, size="1.5%", pad=0.4)

        if cbar:
            ax = cbar_plt.gca()
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('bottom', size="2.5%", pad=0.2)
            cbar = cbar_plt.colorbar(cs, cax=cax,  label=cbar_title, extend=cbar_extend)
            #cbar.solids.set(alpha=1)

        return cs


""" Basemap configuration for plotting geographical fields """    
def plot_basemap(data: xr.DataArray,
                cbar_title: str,
                vmin: float,
                vmax: float,
                alpha: float,
                cmap: str,
                cbar=True,
                cbar_extend='max',
                cbar_position='right',
                return_cs=False,
                axis=None,
                plot_mask=False,
                draw_coordinates=False,
                parallel_label=[1, 0, 0, 0],
                meridian_label=[0, 0, 0, 1],
                contours=None,
                fig=None,
                projection='mill',
                contourf=False,
                map_resolution='i',
                draw_countries=False,
                vmin_contours=0.15,
                vmax_contours=0.40,
                coastline_color='k',
                coastline_linewidth=1.5,
                mask_threshold=1):

    import matplotlib.pyplot as plt

    if axis is not None:
        cbar_plt = plt
        plt = axis

    lats = data.latitude
    lons = data.longitude

    if projection == 'mill':
        lon_0 = 0
    else:
        lon_0 = 0


    m = Basemap(llcrnrlon=lons[0], llcrnrlat=lats[0],
                urcrnrlon=lons[-1], urcrnrlat=lats[-1],
               projection=projection, lon_0=lon_0,
                resolution=map_resolution, ax=axis)

   

    m.drawcoastlines()
    if draw_countries:
        m.drawcountries(color='grey', linewidth=1)
                    
    if draw_coordinates:
        par = m.drawparallels(
                              [-90, -75, -50, -25, 0, 25, 60, 90],
                              #[-90, -60, 0, 60, 90],
                              #[-90, -45, 0, 45, 90],
                              linewidth=1.0,
                              labels=parallel_label,
                              color='grey')

        merid = m.drawmeridians(
                                [-100, -75, -50, -25, 0, 25, 90, 180],
                                #[ -90, 0, 90, 180],
                                #[-120, -60, 0, 60, 120, 180],
                                linewidth=1.0,
                                labels=meridian_label,
                                color='grey')
    
    Lon, Lat = np.meshgrid(lons, lats)
    
    x, y = m(Lon, Lat)
                    
    if contourf:
        cs = plt.contourf(x, y, data, 500, vmin=vmin, vmax=vmax,
                        alpha=alpha, cmap=cmap,
                        linewidth=0, shading='auto', extend='max')
    else:
        cs = plt.pcolormesh(x, y, data, vmin=vmin, vmax=vmax,
                        alpha=alpha, cmap=cmap,
                        linewidth=0, shading='auto')

    m.drawcoastlines(color=coastline_color, linewidth=coastline_linewidth)

    if plot_mask is True:
        mask = np.ma.masked_where(data > mask_threshold, data)
        plt.pcolormesh(x,y, mask, vmin=-1, vmax=-1, alpha=1.0, cmap='Greys',shading='auto')

    if contours is not None:
        cs2 = plt.contour(x, y, abs(contours), 8, 
                            alpha=1.0, cmap='YlOrRd',
                            linewidth=4.0, shading='auto')
    if axis is None:
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(cbar_position, size="1.5%", pad=0.4)


    if cbar:

        ax = cbar_plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(cbar_position, size="2.5%", pad=0.2)

        cbar = cbar_plt.colorbar(cs, cax=cax,  label=cbar_title, extend=cbar_extend)
        cbar.solids.set(alpha=1)
        if cbar_position == 'left':
            cbar.ax.yaxis.set_ticks_position('left')
            cbar.ax.yaxis.set_label_position('left')

        if contours is not None:
            norm = matplotlib.colors.Normalize(vmin=cs2.cvalues.min(), vmax=cs2.cvalues.max())
            sm = plt.cm.ScalarMappable(norm=norm, cmap = cs2.cmap)
            sm.set_array([])

            cax = divider.append_axes('right', size="1.5%", pad=0.2)
            fig.colorbar(sm,
                         ticks=cs2.levels,
                         cax=cax,
                         #orientation="horizontal",
                         orientation="vertical",
                         extend='max',
                         label='Feature importance [a.u.]')

    else:
        if axis is None:
            cax.set_visible(False)
    if return_cs:
        return cs