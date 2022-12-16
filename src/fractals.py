import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.configuration import HistoricalConfig
from src.inference import LoadData

# adapted from https://gist.github.com/rougier/e5eafc276a4e54f516ed5559df4242c0
class Fractals():

    def __init__(self):

        self.lat_shift = 50
        self.lon_shift = 20
        self.names = ['w5e5', 'gfdl', 'w5e5_gan', 'w5e5_gan_unconstrained', 'w5e5_gan_isimip']
        self.names_dict = {}
        self.quantiles = np.arange(0.3, 1, 0.05)
        config = HistoricalConfig()
        config.test_period = ('2004', '2014')
        self.combined_data = LoadData(config).collect_historical_data()
        self.data_fname = '/p/tmp/hess/scratch/cmip-gan/results/fractal_dim_global.pkl'


    def compute_dimension(self):

        for name in self.names:
            print(name)
            data = getattr(self.combined_data, name)
            quantiles_dict = {}
            for quantile in self.quantiles:
                print(quantile)
                dims = []
                threshold = np.quantile(data, quantile)
                for time_index in range(len(data.time)):
                    local_data = data.isel(time=time_index)

                    dim = self.get_fractal_dimension_from_binary(local_data, threshold)
                    dims.append(dim)
                quantiles_dict[quantile] = np.mean(dims), np.std(dims)
            self.names_dict[name] = quantiles_dict

        with open(self.data_fname, 'wb') as f:
            pickle.dump(self.names_dict, f)


    def get_fractal_dimension_from_binary(self, data, threshold):
        result = self.fractal_dimension(data.values, threshold)
        return result[0]


    def fractal_dimension(self, Z, threshold):

        assert(len(Z.shape) == 2)

        Z = (Z < threshold)
        p = min(Z.shape)
        n = 2**np.floor(np.log(p)/np.log(2))
        n = int(np.log(n)/np.log(2))
        sizes = 2**np.arange(n, 1, -1)

        counts = []
        for size in sizes:
            counts.append(self.boxcount(Z, size))

        coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
        return -coeffs[0], sizes, counts


    def boxcount(self, Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                               np.arange(0, Z.shape[1], k), axis=1)

        return len(np.where((S > 0) & (S < k*k))[0])


    def plot(self, fname):
        fig = plt.figure(figsize=(10, 7))
        plt.rcParams.update({'font.size': 14})
        ax = fig.add_subplot(111)


        file = open(self.data_fname, "rb")
        self.names_dict = pickle.load(file)

        target_mean = np.array([data[0] for data in list(self.names_dict['w5e5'].values())[2:]])

        for name in self.names:
            qs = list(self.names_dict[name].keys())[2:]
            mean = np.array([data[0] for data in list(self.names_dict[name].values())[2:]])
            std = np.array([data[1] for data in list(self.names_dict[name].values())[2:]]) 

            error = abs(mean - target_mean).mean()

            if name == 'w5e5_gan':
                line_style = '-'
                dashes = None
                alpha = 0.7
            elif name == 'w5e5_gan_unconstrained':
                line_style = '--'
                alpha = 1.0
            else:
                line_style = '-'
                alpha = 1.0
            
            if name == 'w5e5':
                label = f'{self.combined_data.model_name_definition(name)}'
            elif name == 'w5e5_gan' or name == 'w5e5_gan_unconstrained':
                label = f'{self.combined_data.model_name_definition(name)}: '+r"MAE = $\bf{" + f"{error:2.3f}" + "}$"
            else: 
                label = f'{self.combined_data.model_name_definition(name)}: MAE = {error:2.3f}'

            line, = ax.plot(qs, mean, line_style,
                     label=label,
                     color=self.combined_data.colors(name),
                     linewidth=2, 
                     alpha=alpha)

            if name == 'w5e5_gan_unconstrained':
                line.set_dashes([5, 9])


        plt.ylabel('Fractal dimension')
        plt.xlabel('Quantile')

        plt.legend()

        plt.grid(True)
        plt.savefig(fname, format='pdf', bbox_inches='tight')
        plt.show()
        print(fname)