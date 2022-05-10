import matplotlib.pyplot as plt
import numpy as np
from DeepProteinDocking2D.DatasetGeneration.ProteinPool import ProteinPool, Protein
import seaborn as sea
sea.set_style("whitegrid")
from collections import Counter
from tqdm import tqdm
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})


class ShapeDistributions:
    def __init__(self, protein_pool, dataset_name, show=False):
        self.protein_pool = protein_pool
        self.dataset_name = dataset_name
        self.show = show

    def get_counts(self, counts):
        counter = Counter(counts)
        print('counter', counter)
        unique = np.array(list(counter.keys()))
        inds = unique.argsort()
        counts = np.array(list(counter.values()))[inds]
        return unique, counts

    def get_dict_counts(self, shape_params):
        alpha_counts = []
        numpoints_counts = []
        params_list = []
        for dict in shape_params:
            items = list(dict.items())
            alpha = items[0][1]
            numpoints = items[1][1]
            alpha_counts.append(alpha)
            numpoints_counts.append(numpoints)
            params_list.append([alpha, numpoints])

        return alpha_counts, numpoints_counts, params_list

    def get_unique_fracs(self, counts, dataname):
        unique, counts = self.get_counts(counts)
        fracs = np.array(counts)/sum(counts)
        print('\n'+dataname+':\n')
        print('unique values',unique)
        print('counts', counts)
        print('fractions', fracs)
        barwidth = (unique[-1] - unique[0])/len(unique)

        return unique, fracs, barwidth

    def check_missing_examples(self, combination_list, found_list, protein_shapes, params_list):
        missing_list = []
        missing_indices = []
        for i in range(len(combination_list)):
            if combination_list[i] not in found_list:
                missing_list.append(combination_list[i])
                missing_indices.append(i)

        print('\tMissing examples of [alpha, number of points]:', missing_list)
        print('Regenerating missing examples...')
        for i in tqdm(range(len(missing_list))):
            alpha = missing_list[i][0]
            num_points = missing_list[i][1]
            prot = Protein.generateConcave(size=50, alpha=alpha, num_points=num_points)
            protein_shapes.append(prot.bulk)
            params_list.append([alpha, num_points])

        indices = []
        found_list = []
        for i in combination_list:
            for j in range(len(params_list)):
                if i == params_list[j] and i not in found_list:
                    found_list.append(i)
                    indices.append(j)

        # print('missing indices', missing_indices)
        # print('missing_list', missing_list)
        # print('foundlist', found_list)

        return indices

    def get_shape_distributions(self, data, alpha_counts, numpoints_counts, params_list, debug=False):
        protein_shapes = data.proteins
        shape_params = data.params

        alphas_unique, alphas_fracs, alphas_barwidth = self.get_unique_fracs(alpha_counts, 'alpha values')
        numpoints_unique, numpoints_fracs, numpoints_barwidth = self.get_unique_fracs(numpoints_counts, 'number of points')

        combination_list = [[i,j] for i in alphas_unique for j in numpoints_unique]

        indices = []
        found_list = []
        for i in combination_list:
            for j in range(len(params_list)):
                if i == params_list[j] and i not in found_list:
                    found_list.append(i)
                    indices.append(j)

        if len(found_list) < len(combination_list):
            print('ERROR: Missing combination of "alpha" and "number of points" in protein pool')
            print('\tTry increasing probability parameter or dataset size to increase encounter frequencies')
            indices = self.check_missing_examples(combination_list, found_list, protein_shapes, params_list)

        num_rows = len(alphas_unique)
        num_cols = len(numpoints_unique)

        plot_ranges = []
        for i in range(num_rows):
            cur_range = [*range(i*num_cols, (i+1)*num_cols)]
            plot_ranges.append(cur_range)

        plot_rows = []
        for i in plot_ranges[::-1]:
            cur_indices = np.array(indices)[i]
            cur_row = np.array(protein_shapes)[cur_indices]
            plot_rows.append(np.hstack(cur_row))

        shapes_plot = np.vstack(plot_rows)

        if debug:
            for i in indices:
                print('params', shape_params[i])
                plt.imshow(protein_shapes[i])
                plt.show()

        alphas_packed = alphas_unique, alphas_fracs, alphas_barwidth
        numpoints_packed = numpoints_unique, numpoints_fracs, numpoints_barwidth

        return shapes_plot, alphas_packed, numpoints_packed

    def plot_shapes_and_params(self):
        data = ProteinPool.load(self.protein_pool)

        alpha_counts, numpoints_counts, params_list = self.get_dict_counts(data.params)

        shapes_plot, alphas_packed, numpoints_packed = self.get_shape_distributions(data, alpha_counts, numpoints_counts, params_list)

        alphas_unique,  alphas_fracs, alphas_barwidth = alphas_packed
        numpoints_unique, numpoints_fracs, numpoints_barwidth = numpoints_packed

        num_rows = len(alphas_unique)
        num_cols = len(numpoints_unique)
        plot_lenx = shapes_plot.shape[1]
        plot_leny = shapes_plot.shape[0]

        plt.figure(figsize=(num_rows*2,num_cols*2))
        gs = gridspec.GridSpec(4, 4)
        # gs.update(wspace=0.05, hspace=0.05)
        ax0 = plt.subplot(gs[0, 0:-1])
        ax1 = plt.subplot(gs[1:, 0:-1])
        ax2 = plt.subplot(gs[1:, -1])
        ax3 = plt.subplot(gs[0, -1])
        ax3.set_axis_off()

        numpoints_unique_strs = [str(i) for i in sorted(numpoints_unique)]
        alphas_unique_strs = [str(i)+'0' if len(str(i)) < 4 else str(i) for i in sorted(alphas_unique)]

        ax0.bar(numpoints_unique_strs, numpoints_fracs)
        ax0.grid(False)
        ax0.set_ylabel('fraction')
        plt.setp(ax0.get_xticklabels(), visible=False)
        ax0.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        ax1.imshow(shapes_plot, cmap=plt.get_cmap('binary'))
        ax1.grid(False)
        ax1.set_xlabel('number of points')
        ax1.set_ylabel('alphas')
        ax1.xaxis.set_major_formatter(FormatStrFormatter('%.2g'))
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2g'))
        ax1.set_xticks(np.linspace((plot_lenx/num_cols)/2, plot_lenx-(plot_lenx/num_cols)/2, num_cols))
        ax1.set_xticklabels(numpoints_unique_strs)
        ax1.set_yticks(np.linspace((plot_leny/num_rows)/2, plot_leny-(plot_leny/num_rows)/2, num_rows))
        ax1.set_yticklabels(alphas_unique_strs[::-1])
        ax1.autoscale(False)

        ax2.barh(alphas_unique_strs, alphas_fracs)
        ax2.grid(False)
        ax2.set_xlabel('fraction')
        ax2.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.setp(ax2.get_yticklabels(), visible=False)
        plt.savefig('Figs/ShapeDistributions/'+self.dataset_name + str(len(data.proteins)) + 'pool_combined_shapes_params')

        if self.show:
            plt.show()


if __name__ == "__main__":

    data_path = '../DatasetGeneration/PoolData/'

    num_proteins = 100
    trainvalidset_protein_pool = data_path+'trainvalidset_protein_pool' + str(num_proteins) + '.pkl'

    ShapeDistributions(trainvalidset_protein_pool, 'trainset', show=True).plot_shapes_and_params()

    num_proteins = 100
    testset_protein_pool = data_path+'testset_protein_pool' + str(num_proteins) + '.pkl'

    ShapeDistributions(testset_protein_pool, 'testset', show=True).plot_shapes_and_params()