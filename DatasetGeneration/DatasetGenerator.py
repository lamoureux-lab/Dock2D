import torch
import matplotlib.pyplot as plt
import numpy as np

from collections import Counter
from tqdm import tqdm
from os.path import exists
from Dock2D.DatasetGeneration.ProteinPool import ProteinPool, ParamDistribution
from Dock2D.Utility.TorchDockingFFT import TorchDockingFFT
from Dock2D.Utility.UtilityFunctions import UtilityFunctions
from Dock2D.Utility.PlotterFI import PlotterFI
from Dock2D.Tests.check_shape_distributions import ShapeDistributions

from matplotlib import rcParams


class DatasetGenerator:

    def __init__(self):
        r"""
        Initialize and modify all dataset generation parameters here.
        Creates both training set and testing set protein pools if they do not already exist in the specified ``pool_savepath``.

        .. code-block:: python3

            ### Initializations START
            self.plotting = True
            self.plot_freq = 100
            self.show = True
            self.trainset_pool_stats = None
            self.testset_pool_stats = None

            ## Paths
            self.pool_savepath = 'PoolData/'
            self.poolstats_savepath = 'PoolData/stats/'
            self.data_savepath = '../Datasets/'
            self.datastats_savepath = '../Datasets/stats/'
            self.log_savepath = 'Log/losses/'

            ## initialize FFT
            self.FFT = TorchDockingFFT()

            ## number of unique protein shapes to generate in pool
            self.trainpool_num_proteins = 10
            self.testpool_num_proteins = 10

            ## proportion of training set kept for validation
            self.validation_set_cutoff = 0.8

            ## shape feature scoring coefficients
            self.weight_bound, self.weight_crossterm, self.weight_bulk = 10, 20, 200

            ## energy cutoff for deciding if a shape interacts or not
            self.docking_decision_threshold = -90
            self.interaction_decision_threshold = -90

            ## string of scoring coefficients for plot titles and filenames
            self.weight_string = str(self.weight_bound) + ',' + str(self.weight_crossterm1) + ','\
                                 + str(self.weight_crossterm2) + ',' + str(self.weight_bulk)

            ## training and testing set pool filenames
            self.trainvalidset_protein_pool = 'trainvalidset_protein_pool' + str(self.trainpool_num_proteins) + '.pkl'
            self.testset_protein_pool = 'testset_protein_pool' + str(self.testpool_num_proteins) + '.pkl'

            ### Generate training/validation set protein pool
            ## dataset parameters (value, relative frequency)
            self.train_alpha = [(0.80, 1), (0.85, 2), (0.90, 1)] # concavity level [0-1)
            self.train_num_points = [(60, 1), (80, 2), (100, 1)] # number of points for shape generation [0-1)
            self.train_params = ParamDistribution(alpha=self.train_alpha, num_points=self.train_num_points)

            ### Generate testing set protein pool
            ## dataset parameters (value, relative frequency)
            self.test_alpha = [(0.70, 1), (0.80, 4), (0.90, 6), (0.95, 4), (0.98, 1)]
            self.test_num_points = [(40, 1), (60, 3), (80, 3), (100, 1)]
            self.test_params = ParamDistribution(alpha=self.test_alpha, num_points=self.test_num_points)
            ### Initializations END
        """
        ### Initializations START
        self.plotting = True
        self.plot_freq = 10
        self.show = True
        self.plot_pub = True
        if self.plot_pub:
            self.format = 'pdf'
        else:
            self.format = 'png'
        self.trainset_pool_stats = None
        self.testset_pool_stats = None

        self.plot_accepted_rejected_examples = False
        self.plot_interacting_examples = True
        self.gaussian_blur_bulk = False

        ## Paths
        self.pool_savepath = 'PoolData/'
        self.poolstats_savepath = 'PoolData/stats/'
        self.data_savepath = '../Datasets/'
        self.datastats_savepath = '../Datasets/stats/'
        self.datastats_figs_savepath = '../Datasets/stats/figs/'
        self.log_savepath = 'Log/losses/'

        ## initialize FFT
        padded_dim = 100
        num_angles = 360
        self.FFT = TorchDockingFFT(padded_dim=padded_dim, num_angles=num_angles)
        self.UtilityFuncs = UtilityFunctions()
        ## number of unique protein shapes to generate in pool
        self.trainpool_num_proteins = 50
        self.testpool_num_proteins = 50

        ## proportion of training set kept for validation
        self.validation_set_cutoff = 0.8

        ## shape feature scoring coefficients
        # 10, 5, 500
        # self.weight_bound, self.weight_crossterm, self.weight_bulk = 10, 10, 100

        self.weight_bulk, self.weight_crossterm, self.weight_bound = 100, -10, -10

        ## energy cutoff for deciding if a shape interacts or not
        self.LSEvolume = torch.logsumexp(torch.zeros(num_angles, padded_dim, padded_dim), dim=(0,1,2))

        docking_threshold = torch.tensor(-100)
        self.docking_decision_threshold = docking_threshold
        self.interaction_decision_threshold = docking_threshold

        ## string of scoring coefficients for plot titles and filenames
        self.weight_string = str(self.weight_bound) + ',' + str(self.weight_crossterm) + ',' + str(self.weight_bulk)

        ## training and testing set pool filenames
        self.trainvalidset_protein_pool = 'trainvalidset_protein_pool' + str(self.trainpool_num_proteins) + '.pkl'
        self.testset_protein_pool = 'testset_protein_pool' + str(self.testpool_num_proteins) + '.pkl'

        ### Generate training/validation set protein pool
        ## dataset parameters (value, relative frequency)
        self.train_alpha = [(0.80, 1), (0.85, 2), (0.90, 1)]  # concavity level [0-1)
        self.train_num_points = [(60, 1), (80, 2), (100, 1)]  # number of points for shape generation [0-1)
        self.train_params = ParamDistribution(alpha=self.train_alpha, num_points=self.train_num_points)

        ### Generate testing set protein pool
        ## dataset parameters (value, relative frequency)
        self.test_alpha = [(0.70, 1), (0.80, 4), (0.90, 6), (0.95, 4), (0.98, 1)]
        self.test_num_points = [(40, 1), (60, 3), (80, 3), (100, 1)]
        self.test_params = ParamDistribution(alpha=self.test_alpha, num_points=self.test_num_points)
        ### Initializations END

        ## check if protein pools already exist
        if exists(self.pool_savepath+self.trainvalidset_protein_pool):
            self.trainset_exists = True
            print('\n' + self.trainvalidset_protein_pool, 'already exists!')
            print('This training/validation protein shape pool will be loaded for dataset generation..')
        else:
            self.trainset_exists = False
            print('\n' + self.trainvalidset_protein_pool, 'does not exist yet...')
            print('Generating pool of', str(self.trainpool_num_proteins), 'protein shapes for training/validation set...')
            self.trainset_pool_stats = self.generate_pool(self.train_params, self.trainvalidset_protein_pool, self.trainpool_num_proteins)

        if exists(self.pool_savepath + self.testset_protein_pool):
            self.testset_exists = True
            print('\n' + self.testset_protein_pool, 'already exists!')
            print('This testing protein shape pool will be loaded for dataset generation..')
        else:
            self.testset_exists = False
            print('\n' + self.testset_protein_pool, 'does not exist yet...')
            print('Generating pool of', str(self.testset_protein_pool), 'protein shapes for testing set...')
            self.testset_pool_stats = self.generate_pool(self.test_params, self.testset_protein_pool, self.testpool_num_proteins)

    def generate_pool(self, params, pool_savefile, num_proteins):
        r"""
        Generate the protein pool using parameters for concavity and number of points.


        :param params: Pool generation parameters as a list of tuples for `alpha` and `number of points` as (value, relative freq).

            .. code-block::

                shape_alpha = [(0.70, 1), (0.80, 4), (0.90, 6), (0.95, 4), (0.98, 1)]
                num_points = [(40, 1), (60, 3), (80, 3), (100, 1)]

        :param pool_savefile: protein pool .pkl filename
        :param num_proteins: number of unique protein shapes to make
        :return: ``stats``  observed sampling of alphas and num_points used in protein pool creation
        """
        pool, stats = ProteinPool.generate(num_proteins=num_proteins, params=params)
        pool.save(self.pool_savepath+pool_savefile)
        return stats

    def generate_interactions(self, receptor, ligand):
        r"""
        Generate pairwise interactions through FFT scoring of shape bulk and boundary.

        :param receptor: a shape assigned as ``receptor`` from protein pool
        :param ligand: a shape assigned as ``ligand`` from protein pool
        :return: ``receptor, ligand,`` and their ``fft_score``
        """
        receptor = torch.tensor(receptor, dtype=torch.float).cuda()
        ligand = torch.tensor(ligand, dtype=torch.float).cuda()
        receptor_stack = self.UtilityFuncs.make_boundary(receptor, gaussian_blur_bulk=self.gaussian_blur_bulk)
        ligand_stack = self.UtilityFuncs.make_boundary(ligand, gaussian_blur_bulk=self.gaussian_blur_bulk)
        angle = None
        fft_score = self.FFT.dock_rotations(receptor_stack, ligand_stack, angle,
                                            self.weight_bulk, self.weight_crossterm, self.weight_bound)

        return receptor, ligand, fft_score

    def generate_datasets(self, protein_pool, num_proteins=None):
        r"""
        Generate docking and interaction dataset based on protein pool.

        :param protein_pool: protein pool .pkl filename
        :param num_proteins: can specify size of protein pool to use in generating pairwise interactions, ``None`` uses the entire protein pool.
        :return: ``energies_list`` used only in plotting, the ``docking_set`` is the docking dataset (IP) as a list of lists `[receptor, ligand, rot, trans]`,
                 and ``interaction_set`` a list of `[protein_shapes, indices_list, labels_list]`
        """
        data = ProteinPool.load(self.pool_savepath+protein_pool)

        protein_pool_prefix = protein_pool.split('.')[0]

        protein_shapes = data.proteins
        energies_list = []
        free_energies_list = []
        transformations_list = []

        docking_set = []
        indices_list = []
        labels_list = []
        plot_count = 0

        gt_rotations = []
        interaction_rotations = []
        homodimer_count = 0
        heterodimer_count = 0

        shape_status = None
        freeE_logfile = self.log_savepath+'log_rawdata_FI_'+protein_pool_prefix+'.txt'
        with open(freeE_logfile, 'w') as fout:
            fout.write('F\tF_0\tLabel\n')

        for i in tqdm(range(num_proteins)):
            for j in tqdm(range(i, num_proteins)):
                plot_count += 1
                receptor, ligand = protein_shapes[i], protein_shapes[j]
                receptor, ligand, fft_score = self.generate_interactions(receptor, ligand)

                rot, trans = self.FFT.extract_transform(fft_score)
                energies = fft_score
                deg_index_rot = ((rot * 180.0 / np.pi) + 180.0).type(torch.long)
                minimum_energy = energies[deg_index_rot, trans[0], trans[1]]

                ## picking docking shapes
                if minimum_energy < self.docking_decision_threshold:
                    docking_set.append([receptor, ligand, rot, trans])
                    gt_rotations.append(rot.item())
                    if i == j:
                        homodimer_count += 1
                        shape_status = 'HOMODIMER'
                    else:
                        heterodimer_count += 1
                        shape_status = 'HETERODIMER'

                ## picking interaction shapes
                free_energy = -(torch.logsumexp(-energies, dim=(0, 1, 2)))
                if free_energy < self.interaction_decision_threshold:
                    interaction = torch.tensor(1)
                    interaction_rotations.append(rot.item())
                else:
                    interaction = torch.tensor(0)

                with open(freeE_logfile, 'a') as fout:
                    fout.write('%f\t%s\t%d\n' % (free_energy.item(), self.interaction_decision_threshold.item(), interaction.item()))

                energies_list.append(minimum_energy.item())
                free_energies_list.append(free_energy.item())
                indices_list.append([i, j])
                labels_list.append(interaction)
                transformations_list.append([rot, trans])

                if self.plot_accepted_rejected_examples:
                    self.plot_accepted_rejected_shapes(receptor, ligand, rot, trans, minimum_energy, free_energy, fft_score,
                                                  protein_pool_prefix+shape_status, plot_count)

        dimertype_counts = (homodimer_count, heterodimer_count)

        interaction_set = [protein_shapes, indices_list, labels_list]

        if self.plot_interacting_examples:
            self.plot_interaction_dataset_examples(interaction_set, free_energies_list, transformations_list, protein_pool_prefix)

        return energies_list, free_energies_list, protein_pool_prefix, docking_set, interaction_set, dimertype_counts, gt_rotations, interaction_rotations

    def plot_interaction_dataset_examples(self, interaction_set, free_energies_list, transformations_list, protein_pool_prefix):

        protein_shapes, indices_list, labels_list = interaction_set

        temp = list(zip(indices_list, labels_list, free_energies_list, transformations_list))
        np.random.shuffle(temp)
        shuffled_indices, shuffled_labels, shuffled_free_energies, shuffled_transformations = zip(*temp)
        indices_list, labels_list, free_energies_list, transformations_list = list(shuffled_indices), list(shuffled_labels), list(shuffled_free_energies), list(shuffled_transformations)

        examples_to_plot = 5

        plt.figure(figsize=(examples_to_plot*4, examples_to_plot*2))

        plot_data_interacting = []
        interacting_FE = []
        plot_data_noninteracting = []
        noninteracting_FE = []
        visited_indices = []
        pair = None

        min_FE = min(free_energies_list)
        max_FE = max(free_energies_list)

        for i in range(len(labels_list)):
            receptor_index = indices_list[i][0]
            ligand_index = indices_list[i][1]
            free_energy = np.around(free_energies_list[i], decimals=2)
            rot, trans = transformations_list[i]
            receptor = protein_shapes[receptor_index]
            ligand = protein_shapes[ligand_index]
            label = labels_list[i]

            if label == 1:# and free_energy < min_FE + 20:
                if len(plot_data_interacting) < examples_to_plot and receptor_index not in visited_indices:
                    # print('interaction found', free_energy)
                    pair = UtilityFunctions().plot_assembly(receptor,
                                                            ligand,
                                                            rot.detach().cpu(),
                                                            trans.detach().cpu(),
                                                            interaction_fact=True)
                    pair = pair[20:130, 20:130]
                    plot_data_interacting.append(pair)
                    interacting_FE.append(free_energy)
                    visited_indices.append(receptor_index)
            if label == 0 and free_energy > max_FE - 5:
                if len(plot_data_noninteracting) < examples_to_plot and receptor_index not in visited_indices:
                    # print('non-interaction found', free_energy)
                    pair = UtilityFunctions().plot_assembly(receptor,
                                                            ligand,
                                                            rot.detach().cpu(),
                                                            trans.detach().cpu(),
                                                            interaction_fact=True)
                    pair = pair[20:130, 20:130]
                    plot_data_noninteracting.append(pair)
                    noninteracting_FE.append(free_energy)
                    visited_indices.append(receptor_index)

            if len(plot_data_interacting) > examples_to_plot and len(plot_data_noninteracting) > examples_to_plot:
                break

        interacting = np.hstack((plot_data_interacting))
        noninteracting = np.hstack((plot_data_noninteracting))

        plot = np.vstack((interacting, noninteracting))

        pair_dim = pair.shape[-1]
        spacer = pair_dim
        offset = spacer*(2/3)
        midpoint = (examples_to_plot*spacer)//2
        font = {'weight': 'bold',
                'size': 24,}
        plt.text(midpoint-2.1*len(' interacting '), -0.1*spacer, 'interacting', fontdict=font)
        plt.text(midpoint-2.5*len(' non-interacting '), 0.9*spacer, 'non-interacting', fontdict=font)
        # plt.text(0, 0, s=''.join(str(interacting_FE).split(',')[1:-1]))
        # plt.text(0, 100, s=''.join(str(noninteracting_FE).split(',')[1:-1]))

        font = {'weight': 'normal',
                'size': 24,}
        for i in range(examples_to_plot):
            plt.text((i+1)*spacer-offset, 0.05*spacer, str(interacting_FE[i]), fontdict=font)
            plt.text((i+1)*spacer-offset, 1.05*spacer, str(noninteracting_FE[i]), fontdict=font)

        cmap = 'gist_heat_r'
        plt.imshow(plot, cmap=cmap)
        plt.grid(False)
        plt.axis('off')
        # plt.show()
        # protein_pool_prefix_title = ' '.join(protein_pool_prefix.split('_'))
        plt.savefig(self.datastats_figs_savepath + protein_pool_prefix+'_interactionsVSnon-interactions.'+self.format, format=self.format)
        if self.show:
            plt.show()

    def plot_gt_rotation_distributions(self, gt_rotations, interaction_rotations,  protein_pool_prefix):
        plt.close()

        plt.figure(figsize=(8,6))
        plt.hist(gt_rotations, alpha=0.33, bins=60)
        # plt.hist(interaction_rotations, alpha=0.33, bins=60)
        plt.xlim([-np.pi, np.pi])
        plt.xticks(np.round(np.linspace(-np.pi, np.pi, 5, endpoint=True), decimals=2))
        protein_pool_prefix_title = ' '.join(protein_pool_prefix.split('_'))
        if not self.plot_pub:
            plt.title('Accepted rotations '+protein_pool_prefix_title+', docking<'+str(self.docking_decision_threshold.item())[:6])
        plt.xlabel(r'rotation $(\mathrm{\phi})$')
        plt.ylabel('count')
        # plt.legend([r'$E(\mathrm{\phi})$'])
        plt.grid(None)
        plt.margins(x=None)
        plt.savefig(self.datastats_figs_savepath + protein_pool_prefix+'_groundtruth_rotation_distribution.'+self.format, format=self.format)
        if self.show:
            plt.show()

    def plot_energy_distributions(self, energies_list, free_energies, protein_pool_prefix):
        r"""
        Plot histograms of all pairwise energies and free energies, within training and testing set.

        :param energies_list: all pairwise energies (E = fft_scores)
        :param free_energies: all pairwise energies (logsumexp(-E))
        :param protein_pool_prefix: used in title and filename
        """
        plt.close()
        plt.figure(figsize=(8,6))
        protein_pool_prefix_title = ' '.join(protein_pool_prefix.split('_'))
        if not self.plot_pub:
            plt.title('Energies '+protein_pool_prefix_title)
        y1, x1, _ = plt.hist(energies_list, alpha=0.33, bins=60)
        y2, x2, _ = plt.hist(free_energies, alpha=0.33, bins=60)
        plt.xlabel('energy')
        plt.ylabel('count')
        ymax = max(max(y1.max(), y2.max()), max(y1.max(), y2.max()))+1
        docking_value = str(self.docking_decision_threshold.item())[:6]
        docking_label = 'docking < '+docking_value
        interaction_value = str(self.interaction_decision_threshold.item())[:6]
        interaction_label = 'interaction < '+interaction_value
        plt.vlines(self.docking_decision_threshold, ymin=0, ymax=max(y1.max(), y2.max())+1, linestyles='dashed', colors='k',
                   # label=docking_label
                   label='_nolegend_'
                   )
        plt.vlines(self.interaction_decision_threshold, ymin=0, ymax=ymax, linestyles='solid', colors='k',
                   # label=interaction_label
                   label='_nolegend_'
                   )
        plt.legend(['Energy minima', 'Free energies'], prop={'size': 12}, loc='upper left')
        plt.grid(None)
        plt.margins(x=None)
        plt.xlim([-250, 0])

        plt.savefig(self.datastats_figs_savepath + protein_pool_prefix+'_MinEnergyandFreeEnergy_distribution.'+self.format, format=self.format)
        if self.show:
            plt.show()

    def plot_accepted_rejected_shapes(self, receptor, ligand, rot, trans, minimum_energy, free_energy, fft_score, protein_pool_prefix, plot_count):
        r"""
        Plot examples of accepted and rejected shape pairs, based on docking and interaction decision thresholds set.

        :param receptor: receptor shape image
        :param ligand: ligand shape image
        :param rot: rotation to apply to ligand
        :param trans: translation to apply to ligand
        :param minimum_energy: minimum energy from FFT score
        :param fft_score: over all FFT score
        :param protein_pool_prefix: filename prefix
        :param plot_count: used as index in plotting
        """
        if plot_count % self.plot_freq == 0:
            plt.close()
            pair = UtilityFunctions().plot_assembly(receptor.cpu(), ligand.cpu(), rot.cpu(), trans.cpu())
            plt.imshow(pair.transpose())
            docking_cond = minimum_energy < self.docking_decision_threshold
            interaction_cond = free_energy < self.interaction_decision_threshold
            if docking_cond and interaction_cond:
                acc_or_rej = 'ACCEPTED_dockingANDinteraction'
            elif docking_cond and not interaction_cond:
                acc_or_rej = 'ACCEPTED_docking_REJECTED_interaction'
            elif not docking_cond and interaction_cond:
                acc_or_rej = 'ACCEPTED_interaction_REJECTED_docking'
            else:
                acc_or_rej = 'REJECTED'
            title = acc_or_rej + '_energy' + str(minimum_energy.item()) + '_docking' + str(self.docking_decision_threshold) + '_interaction' + str(self.interaction_decision_threshold)
            plt.title(title)
            plt.savefig('Figs/AcceptRejectExamples/' + title + '.png') # not included in manuscript
            protein_pool_prefix = ' '.join(protein_pool_prefix.split('_'))

            ### plot corresponding 2d energy surface best scoring translation energy vs rotation angle
            UtilityFunctions().plot_rotation_energysurface(fft_score, trans,
                                                           stream_name=acc_or_rej + '_datasetgen_' + protein_pool_prefix,
                                                           plot_count=plot_count)

    def run_generator(self):
        r"""
        Generates the training, validation, and testing sets for both docking (IP) and interaction (FI) from current protein pool.
        Write all datasets to .pkl files. Saves all metrics to file. Prints IP and FI dataset stats.
        If ``self.plotting=True`` plot and save dataset generation plots.
        Specify ``self.show=True`` to show each plot in a new window (does not affect saving).

        Links to plotting methods:

            |    :func:`~plot_energy_distributions`
            |    :func:`~Utility.UtilityFunctions.plot_rotation_energysurface`
            |    :func:`~plot_accepted_rejected_shapes`
            |    :func:`~Dock2D.Utility.PlotterFI.plot_deltaF_distribution`
            |    :func:`~ShapeDistributions.plot_shapes_and_params`

        """

        ### Generate training/validation set
        train_energies_list, train_free_energies_list, train_protein_pool_prefix, train_docking_set, train_interaction_set, train_dimer_count, train_gt_rotations, train_interaction_rotations = self.generate_datasets(
            self.trainvalidset_protein_pool, self.trainpool_num_proteins)
        ### Generate testing set
        test_energies_list, test_free_energies_list, test_protein_pool_prefix, test_docking_set, test_interaction_set, test_dimer_count, test_gt_rotations, test_interaction_rotations = self.generate_datasets(
            self.testset_protein_pool, self.testpool_num_proteins)

        ## Slice validation set out of shuffled training docking set
        valid_docking_cutoff_index = int(len(train_docking_set) * self.validation_set_cutoff)
        np.random.shuffle(train_docking_set)
        shuffled_docking_set = train_docking_set
        train_docking_set = shuffled_docking_set[:valid_docking_cutoff_index]
        valid_docking_set = shuffled_docking_set[valid_docking_cutoff_index:]

        ## Slice validation set out of simultaneously shuffled training interaction set
        valid_interaction_cutoff_index = int(len(train_interaction_set[-1]) * self.validation_set_cutoff)
        training_pool_shapes = train_interaction_set[0]
        train_interaction_set_indices = train_interaction_set[1]
        train_interaction_set_labels = train_interaction_set[2]

        temp = list(zip(train_interaction_set_indices, train_interaction_set_labels))
        np.random.shuffle(temp)
        shuffled_indices, shuffled_labels = zip(*temp)
        shuffled_indices_list, shuffled_labels_list = list(shuffled_indices), list(shuffled_labels)

        train_interaction_set = [training_pool_shapes,
                                 shuffled_indices_list[:valid_interaction_cutoff_index],
                                 shuffled_labels_list[:valid_interaction_cutoff_index]]

        valid_interaction_set = [training_pool_shapes,
                                 shuffled_indices_list[valid_interaction_cutoff_index:],
                                 shuffled_labels_list[valid_interaction_cutoff_index:]]

        ## Interaction set stats for train, valid, and test
        train_interaction_set_labels = train_interaction_set[2]
        number_of_positive_train_interactions = sum(train_interaction_set_labels)
        fraction_positive_train_interactions = number_of_positive_train_interactions/len(train_interaction_set_labels)

        valid_interaction_set_labels = valid_interaction_set[2]
        number_of_positive_valid_interactions = sum(valid_interaction_set_labels)
        fraction_positive_valid_interactions = number_of_positive_valid_interactions/len(valid_interaction_set_labels)

        test_interaction_set_labels = test_interaction_set[2]
        number_of_positive_test_interactions = sum(test_interaction_set_labels)
        fraction_positive_test_interactions = number_of_positive_test_interactions/len(test_interaction_set_labels)

        # ### Print dataset stats
        # print('\nProtein Pool:', self.trainpool_num_proteins)
        # print('Docking decision threshold ', self.docking_decision_threshold)
        # print('Interaction decision threshold ', self.interaction_decision_threshold)
        #
        # print('\nRaw Training set:')
        # print('Docking set length', len(train_docking_set))
        # print('Interaction set length', len(train_interaction_set[-1]))
        #
        # print('\nRaw Validation set:')
        # print('Docking set length', len(valid_docking_set))
        # print('Interaction set length', len(valid_interaction_set[-1]))
        #
        # print('\nRaw Testing set:')
        # print('Docking set length', len(test_docking_set))
        # print('Interaction set length', len(test_interaction_set[-1]))
        #
        # ## Write protein pool summary statistics to file
        # if not self.trainset_exists:
        #     with open(self.poolstats_savepath + 'protein_trainpool_stats_' + str(self.trainpool_num_proteins) + 'pool.txt',
        #               'w') as fout:
        #         fout.write('TRAIN/VALIDATION SET PROTEIN POOL STATS')
        #         fout.write('\nProtein Pool size=' + str(self.trainpool_num_proteins) + ':')
        #         fout.write(
        #             '\nTRAIN set params (alpha, number of points):\n' + str(self.train_alpha) + '\n' + str(self.train_num_points))
        #         fout.write('\nTRAIN set probabilities: ' + '\nalphas:' + str(self.trainset_pool_stats[0]) +
        #                    '\nnumber of points' + str(self.trainset_pool_stats[1]))
        #
        # if not self.testset_exists:
        #     with open(self.poolstats_savepath + 'protein_testpool_stats_' + str(self.testpool_num_proteins) + 'pool.txt',
        #               'w') as fout:
        #         fout.write('TEST SET PROTEIN POOL STATS')
        #         fout.write('\nProtein Pool size=' + str(self.testpool_num_proteins) + ':')
        #         fout.write(
        #             '\n\nTEST set params  (alpha, number of points):\n' + str(self.test_alpha) + '\n' + str(self.test_num_points))
        #         fout.write('\nTEST set probabilities: ' + '\nalphas:' + str(self.testset_pool_stats[0]) +
        #                    '\nnumber of points' + str(self.testset_pool_stats[1]))
        #
        # ## Write dataset summary statistics to file
        # with open(self.datastats_savepath + 'trainvalid_dataset_stats_' + str(self.trainpool_num_proteins) + 'pool.txt',
        #           'w') as fout:
        #     fout.write('TRAIN DATASET STATS')
        #     fout.write('\nProtein Pool size=' + str(self.trainpool_num_proteins) + ':')
        #     fout.write('\nScoring Weights: Bound,Crossterm,Bulk ' + self.weight_string)
        #     fout.write('\nDocking decision threshold ' + str(self.docking_decision_threshold))
        #     fout.write('\nInteraction decision threshold ' + str(self.interaction_decision_threshold))
        #
        #     fout.write('\nDocking set homodimer vs heterodimer counts:')
        #     fout.write('\nHomodimer count ' + str(train_dimer_count[0]))
        #     fout.write('\nHeterodimer count ' + str(train_dimer_count[1]))
        #
        #     unique = Counter(np.around(np.array(train_gt_rotations), decimals=1))
        #     fout.write('\nUnique rotations count:\n' + str(unique))
        #
        #     fout.write('\n\nRaw Training set:')
        #     fout.write('\nDocking set length ' + str(len(train_docking_set)))
        #     fout.write('\nInteraction set length ' + str(len(train_interaction_set[-1])))
        #     fout.write('\nInteraction set positive example count ' + str(number_of_positive_train_interactions))
        #     fout.write('\nInteraction set positive example fraction ' + str(fraction_positive_train_interactions))
        #
        #     fout.write('\n\nRaw Validation set:')
        #     fout.write('\nDocking set length ' + str(len(valid_docking_set)))
        #     fout.write('\nInteraction set length ' + str(len(valid_interaction_set[-1])))
        #     fout.write('\nInteraction set positive example count ' + str(number_of_positive_valid_interactions))
        #     fout.write('\nInteraction set positive example fraction ' + str(fraction_positive_valid_interactions))
        #
        # with open(self.datastats_savepath + 'testset_dataset_stats_' + str(self.testpool_num_proteins) + 'pool.txt', 'w') as fout:
        #     fout.write('TEST DATASET STATS')
        #     fout.write('\nProtein Pool size=' + str(self.testpool_num_proteins) + ':')
        #     fout.write('\nScoring Weights: Bound,Crossterm,Bulk ' + self.weight_string)
        #     fout.write('\nDocking decision threshold ' + str(self.docking_decision_threshold))
        #     fout.write('\nInteraction decision threshold ' + str(self.interaction_decision_threshold))
        #
        #     fout.write('\nDocking set homodimer vs heterodimer counts:')
        #     fout.write('\nHomodimer count ' + str(test_dimer_count[0]))
        #     fout.write('\nHeterodimer count ' + str(test_dimer_count[1]))\
        #
        #     unique = Counter(np.around(np.array(test_gt_rotations), decimals=1))
        #     fout.write('\nUnique rotations count:\n' + str(unique))
        #
        #     fout.write('\n\nRaw Testing set:')
        #     fout.write('\nDocking set length ' + str(len(test_docking_set)))
        #     fout.write('\nInteraction set length ' + str(len(test_interaction_set[-1])))
        #     fout.write('\nInteraction set positive example count ' + str(number_of_positive_test_interactions))
        #     fout.write('\nInteraction set positive example fraction ' + str(fraction_positive_test_interactions))
        #
        # ## Save training sets
        # docking_train_file = self.data_savepath + 'docking_train_' + str(self.trainpool_num_proteins) + 'pool.pkl'
        # interaction_train_file = self.data_savepath + 'interaction_train_' + str(self.trainpool_num_proteins) + 'pool.pkl'
        # UtilityFunctions().write_pkl(data=train_docking_set, filename=docking_train_file)
        # UtilityFunctions().write_pkl(data=train_interaction_set, filename=interaction_train_file)
        #
        # ## Save validation sets
        # docking_valid_file =self.data_savepath + 'docking_valid_' + str(self.trainpool_num_proteins) + 'pool.pkl'
        # interaction_valid_file = self.data_savepath + 'interaction_valid_' + str(self.trainpool_num_proteins) + 'pool.pkl'
        # UtilityFunctions().write_pkl(data=valid_docking_set, filename=docking_valid_file)
        # UtilityFunctions().write_pkl(data=valid_interaction_set, filename=interaction_valid_file)
        #
        # ## Save testing sets
        # docking_test_file = self.data_savepath + 'docking_test_' + str(self.testpool_num_proteins) + 'pool.pkl'
        # interaction_test_file = self.data_savepath + 'interaction_test_' + str(self.testpool_num_proteins) + 'pool.pkl'
        # UtilityFunctions().write_pkl(data=test_docking_set, filename=docking_test_file)
        # UtilityFunctions().write_pkl(data=test_interaction_set, filename=interaction_test_file)

        if self.plotting:
            training_set_name = self.trainvalidset_protein_pool.split('.')[0]
            testing_set_name = self.testset_protein_pool.split('.')[0]

            ## Dataset shape pair docking energies distributions
            # self.plot_energy_distributions(train_energies_list, test_energies_list, show=self.show)
            self.plot_energy_distributions(train_energies_list, train_free_energies_list, train_protein_pool_prefix)
            self.plot_energy_distributions(test_energies_list, test_free_energies_list, test_protein_pool_prefix)
            self.plot_gt_rotation_distributions(train_gt_rotations, train_interaction_rotations, train_protein_pool_prefix)
            self.plot_gt_rotation_distributions(test_gt_rotations, train_interaction_rotations, test_protein_pool_prefix)

            ## Dataset free energy distributions
            ## Plot interaction training/validation set
            training_filename = self.log_savepath + 'log_rawdata_FI_' + training_set_name + '.txt'
            PlotterFI(training_set_name).plot_deltaF_distribution(filename=training_filename, binwidth=2,
                                                                                show=self.show, plot_pub=self.plot_pub)

            ## Plot interaction testing set
            testing_filename = self.log_savepath + 'log_rawdata_FI_' + testing_set_name + '.txt'
            PlotterFI(testing_set_name).plot_deltaF_distribution(filename=testing_filename, binwidth=2,
                                                                          show=self.show, plot_pub=self.plot_pub)

            ## Plot protein pool distribution summary
            ShapeDistributions(self.pool_savepath + self.trainvalidset_protein_pool, 'trainset',
                               show=self.show).plot_shapes_and_params(plot_pub=self.plot_pub)
            ShapeDistributions(self.pool_savepath + self.testset_protein_pool, 'testset',
                               show=self.show).plot_shapes_and_params(plot_pub=self.plot_pub)


if __name__ == '__main__':
    rcParams.update({'font.size': 15})
    DatasetGenerator = DatasetGenerator()
    DatasetGenerator.run_generator()
