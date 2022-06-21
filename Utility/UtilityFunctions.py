import scipy.ndimage as ndimage
import _pickle as pkl
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import torch
from torch.nn import functional as F
from Dock2D.Utility.ValidationMetrics import RMSD
from matplotlib import rcParams
# rcParams.update({'figure.autolayout': True})
rcParams.update({'font.size': 14})


class UtilityFunctions():
    def __init__(self, experiment=None):
        self.experiment = experiment

    def write_pkl(self, data, filename):
        '''
        :param data: to write to .pkl  file
        :param filename: specify `filename.pkl`
        '''
        print('\nwriting '+filename+' to .pkl\n')
        with open(filename, 'wb') as fout:
            pkl.dump(data, fout)
        fout.close()

    def read_pkl(self, filename):
        '''
        :param filename: `filename.pkl` to load
        :return: data
        '''
        print('\nreading '+filename+'\n')
        with open(filename, 'rb') as fin:
            data = pkl.load(fin)
        fin.close()
        return data

    def check_model_gradients(self, model):
        '''
        Check current model parameters and gradients in-place.
        Specifically if weights are frozen or updating
        '''
        for n, p in model.named_parameters():
            if p.requires_grad:
                print('name', n, 'param', p, 'gradient', p.grad)

    def weights_init(self, model):
        '''
        Initialize weights for SE(2)-equivariant convolutional network.
        Generally unused for SE(2) network, as e2nn library has its own Kaiming He weight initialization.
        '''
        if isinstance(model, torch.nn.Conv2d):
            print('updating convnet weights to kaiming uniform initialization')
            torch.nn.init.kaiming_uniform_(model.weight)
            # torch.nn.init.kaiming_normal_(model.weight)

    def swap_quadrants(self, input_volume):
        """
        FFT returns features centered with the origin at the center of the image, not at the top left corner.

        :param input_volume: FFT output array
        """
        num_features = input_volume.size(0)
        L = input_volume.size(-1)
        L2 = int(L / 2)
        output_volume = torch.zeros(num_features, L, L, device=input_volume.device, dtype=input_volume.dtype)

        output_volume[:, :L2, :L2] = input_volume[:, L2:L, L2:L]
        output_volume[:, L2:L, L2:L] = input_volume[:, :L2, :L2]

        output_volume[:, L2:L, :L2] = input_volume[:, :L2, L2:L]
        output_volume[:, :L2, L2:L] = input_volume[:, L2:L, :L2]

        output_volume[:, L2:L, L2:L] = input_volume[:, :L2, :L2]
        output_volume[:, :L2, :L2] = input_volume[:, L2:L, L2:L]

        return output_volume

    @staticmethod
    def make_boundary(grid_shape):
        """
        Create the boundary feature for data generation and unit testing.

        :param grid_shape: input shape grid image
        :return: features stack with original shape as "bulk" and created "boundary"
        """
        grid_shape = grid_shape.unsqueeze(0).unsqueeze(0)
        epsilon = 1e-5
        sobel_top = torch.tensor([[[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]], dtype=torch.float).cuda()
        sobel_left = sobel_top[0, 0, :, :].t().view(1, 1, 3, 3)

        feat_top = F.conv2d(grid_shape, weight=sobel_top, padding=1)
        feat_left = F.conv2d(grid_shape, weight=sobel_left, padding=1)

        top = feat_top
        right = feat_left
        boundary = torch.sqrt(top ** 2 + right ** 2)
        feat_stack = torch.cat([grid_shape, boundary], dim=1)

        return feat_stack.squeeze()

    def rotate(self, repr, angle):
        """
        Rotate a grid image using 2D rotation matrix.

        :param repr: input grid image
        :param angle: angle in radians
        :return: rotated grid image
        """
        alpha = angle.detach()
        T0 = torch.stack([torch.cos(alpha), -torch.sin(alpha), torch.zeros_like(alpha)], dim=1)
        T1 = torch.stack([torch.sin(alpha), torch.cos(alpha), torch.zeros_like(alpha)], dim=1)
        R = torch.stack([T0, T1], dim=1)
        curr_grid = F.affine_grid(R, size=repr.size(), align_corners=True).type(torch.float)
        return F.grid_sample(repr, curr_grid, align_corners=True)

    def plot_rotation_energysurface(self, fft_score, pred_txy, num_angles=360, stream_name=None, plot_count=0, format='png'):
        """
        Plot the lowest energy translation index from `fft_score` per rotation angle as an energy surface curve.

        :param fft_score: FFT scores generated using a docker method.
        :param pred_txy: predicted translation `[x, y]`
        :param num_angles: number of angles used to generate `fft_score`
        :param stream_name: data stream name
        :param plot_count: plotting index used in titles and filename
        """
        plt.close()
        mintxy_energies = []
        if num_angles == 1:
            minimumEnergy = -fft_score[pred_txy[0], pred_txy[1]].detach().cpu()
            mintxy_energies.append(minimumEnergy)
        else:
            for i in range(num_angles):
                minimumEnergy = -fft_score[i, pred_txy[0], pred_txy[1]].detach().cpu()
                mintxy_energies.append(minimumEnergy)

        fig = plt.figure(figsize=(8,5))
        plt.rcParams['axes.xmargin'] = 0
        plt.rcParams['axes.ymargin'] = 0.05
        rcParams.update({'font.size': 14})

        axarr = fig.add_subplot(1,1,1)
        xrange = np.arange(-np.pi, np.pi, 2 * np.pi / num_angles)
        plt.xticks(np.linspace(-np.pi, np.pi, 5, endpoint=True))
        plt.xlim([-np.pi, np.pi])
        plt.plot(xrange, mintxy_energies)
        hardmin_minEnergies = stream_name + '_energysurface' + '_example' + str(plot_count)
        plt.title('Best Scoring Translation Energy Surface')
        plt.ylabel('Energy')
        plt.xlabel('Rotation (rads)')
        plt.savefig('Figs/EnergySurfaces/' + hardmin_minEnergies + '.'+format, format=format)

    def plot_MCsampled_energysurface(self, free_energies_visited_indices, accumulated_free_energies, acceptance_rate, stream_name=None, interaction=None, plot_count=0, epoch=0):
        """
        Plot the accumulated sample buffer free energies from Monte Carlo sampling.

        :param free_energies_visited_indices: unique indices of free energies already visted
        :param accumulated_free_energies: visited free energies recomputed per epoch
        :param acceptance_rate: MC sampling acceptance rate in plot title
        :param stream_name: data stream name
        :param interaction: current example interaction label (1 or 0)
        :param plot_count: plotting index used in titles and filename
        :param epoch: epoch
        """
        plt.close()
        plt.figure(figsize=(15,10))

        if len(free_energies_visited_indices[0]) > 1:
            # print('free_energies_visited_indices', free_energies_visited_indices)
            # print('accumulated_free_energies', accumulated_free_energies)

            free_energies_visited_indices = free_energies_visited_indices.squeeze().detach().cpu()
            accumulated_free_energies = accumulated_free_energies.squeeze().detach().cpu()

            free_energies_indices = free_energies_visited_indices.numpy().sort()
            inds = np.array(free_energies_visited_indices).argsort()
            freeEnergies_argsort = np.array(accumulated_free_energies)[inds]

            # print(free_energies_visited_indices)
            # print(free_energies_indices)
            # print(inds)
            # print(freeEnergies_argsort)

            mcsampled_energies_name = 'example' + str(plot_count)+'_epoch'+str(epoch) + stream_name + self.experiment + '_energysurface'
            # plt.plot(free_energies_visited_indices, freeEnergies_argsort)
            plt.scatter(free_energies_visited_indices, freeEnergies_argsort)
            plt.ylim([min(freeEnergies_argsort), 1])
            plt.hlines(y=0, xmin=0, xmax=360, linestyles='dashed', label='zero energy', colors='k')
            plt.suptitle('MonteCarlo sampled energy surface, acceptance rate='+str(acceptance_rate)+', interaction='+str(interaction))
            plt.title(mcsampled_energies_name)
            plt.ylabel('Energy')
            plt.xlabel('Rotation indices')
            plt.savefig('Figs/EnergySurfaces/' + mcsampled_energies_name + '.png')

    def plot_assembly(self, receptor, ligand, gt_rot, gt_txy, pred_rot=None, pred_txy=None, tiling=False, interaction_fact=False):
        """
        Plot the predicting docking pose for the IP task. From left to right, plots the ground truth docking pose,
        the pose passed into the model, and the predicted docking pose.

        :param receptor: receptor shape grid image
        :param ligand: ligand shape grid image
        :param gt_rot: ground truth rotation
        :param gt_txy: ground truth translation `[x, y]`
        :param pred_rot: predicted rotation
        :param pred_txy: predicted translation `[x, y]`
        :return: plotting object with specified poses
        """
        box_size = receptor.shape[-1]
        if tiling:
            receptor_copy = receptor
            ligand_copy = ligand
        elif interaction_fact:
            receptor_copy = receptor * 2
            ligand_copy = ligand
            padding = box_size
            receptor_copy = np.pad(receptor_copy, ((padding, padding), (padding, padding)), 'constant', constant_values=0)
            ligand_copy = np.pad(ligand_copy, ((padding, padding), (padding, padding)), 'constant', constant_values=0)
        else:
            receptor_copy = receptor * -100
            ligand_copy = ligand * 200

        padding = box_size//2
        if box_size < 100:
            receptor_copy = np.pad(receptor_copy, ((padding, padding), (padding, padding)), 'constant', constant_values=0)
            ligand_copy = np.pad(ligand_copy, ((padding, padding), (padding, padding)), 'constant', constant_values=0)

        inputshapes = receptor_copy + ligand_copy

        gt_rot = (gt_rot * 180.0/np.pi)
        gt_transformlig = self.rotate_gridligand(ligand_copy, gt_rot)
        gt_transformlig = self.translate_gridligand(gt_transformlig, gt_txy[0], gt_txy[1])
        gt_transformlig += receptor_copy

        if pred_txy is not None and pred_rot is not None:
            pred_rot = (pred_rot * 180.0 / np.pi)
            transformligand = self.rotate_gridligand(ligand_copy, pred_rot)
            transformligand = self.translate_gridligand(transformligand, pred_txy[0], pred_txy[1])
            transformligand += receptor_copy

            pair = np.vstack((gt_transformlig, inputshapes, transformligand))
        elif tiling:
            pair, ligand_copy = abs(gt_transformlig), abs(ligand_copy)
            return pair, ligand_copy
        elif interaction_fact:
            pair = np.clip(gt_transformlig, a_min=0, a_max=2)
            return pair
        else:
            pair = np.vstack((gt_transformlig, inputshapes))

        return pair

    def rotate_gridligand(self, ligand, rotation_angle):
        """
        Rotate grid image in degrees using `scipy.ndimage.rotate()` for :func:`plot_assembly()`

        :param ligand: grid image of ligand
        :param rotation_angle: angle in degrees
        :return: rotated ligand
        """
        ligand = ndimage.rotate(ligand, rotation_angle, reshape=False, order=3, mode='nearest', cval=0.0)
        return ligand

    def translate_gridligand(self, ligand, tx, ty):
        """
        Translate grid image using `scipy.ndimage.shift()` for :func:`plot_assembly()`

        :param ligand: grid image of ligand
        :param tx: x dimension translation
        :param ty: y dimension translation
        :return: translated ligand
        """
        ligand = ndimage.shift(ligand, [tx, ty], mode='wrap', order=3, cval=0.0)
        return ligand

    def plot_predicted_pose(self, receptor, ligand, gt_rot, gt_txy, pred_rot, pred_txy, plot_count, stream_name):
        """
        Plotting helper function for :func:`plot_assembly()`.

        :param receptor: receptor shape grid image
        :param ligand: ligand shape grid image
        :param gt_rot: ground truth rotation
        :param gt_txy: ground truth translation `[x, y]`
        :param pred_rot: predicted rotation
        :param pred_txy: predicted translation `[x, y]`
        :param plot_count: plotting index used in titles and filename
        :param stream_name: data stream name
        """
        plt.close()
        plt.figure(figsize=(8, 8))
        # pred_rot, pred_txy = self.dockingFFT.extract_transform(fft_scores)
        print('extracted predicted indices', pred_rot, pred_txy)
        print('gt indices', gt_rot, gt_txy)
        rmsd_out = RMSD(ligand, gt_rot, gt_txy, pred_rot, pred_txy).calc_rmsd()
        print('RMSD', rmsd_out.item())

        pair = self.plot_assembly(receptor.squeeze().detach().cpu().numpy(), ligand.squeeze().detach().cpu().numpy(),
                                            pred_rot.detach().cpu().numpy(),
                                            (pred_txy[0].detach().cpu().numpy(), pred_txy[1].detach().cpu().numpy()),
                                            gt_rot.squeeze().detach().cpu().numpy(), gt_txy.squeeze().detach().cpu().numpy())

        plt.imshow(pair.transpose())
        plt.title('Ground truth', loc='left')
        plt.title('Input')
        plt.title('Predicted pose', loc='right')
        plt.text(225, 110, "RMSD = " + str(rmsd_out.item())[:5], backgroundcolor='w')
        plt.grid(False)
        plt.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)  # labels along the bottom
        plt.tick_params(
            axis='y',
            which='both',
            left=False,
            right=False,
            labelleft=False)
        plt.savefig('Figs/Features_and_poses/'+stream_name+'_docking_pose_example' + str(plot_count) + '_RMSD' + str(rmsd_out.item())[:4] + '.png')
        # plt.show()

    def orthogonalize_feats(self, scoring_weights, feat_stack):
        """
        Orthogonalize learned shape features for single shape.

        :param scoring_weights: learned scoring coefficients from scoring function
        :param feat_stack: feature stack for one shape [bulk, boundary]
        :return: orthogonalized feature stack
        """
        boundW, crosstermW, bulkW = scoring_weights
        A = torch.tensor([[boundW, crosstermW],[crosstermW, bulkW]])
        eigvals, V = torch.linalg.eig(A)
        V = V.real
        rv11 = V[0, 0] * feat_stack[0, :, :] + V[1, 0] * feat_stack[1, :, :]
        rv12 = V[0, 1] * feat_stack[0, :, :] + V[1, 1] * feat_stack[1, :, :]
        orth_feats = torch.stack([rv11, rv12], dim=0).unsqueeze(dim=0).detach()
        return orth_feats

    def plot_features(self, rec_feat, lig_feat, receptor, ligand, scoring_weights, plot_count=0, stream_name='trainset'):
        """
        Plot the learned shape pair features (bulk and boundary) from the docking model from `Docking` in `model_docking.py`.

        :param rec_feat: receptor feature stack
        :param lig_feat: ligand feature stack
        :param receptor: receptor shape grid image
        :param ligand: ligand shape grid image
        :param scoring_weights: learned scoring coefficients used in scoring function
        :param plot_count: plotting index used in titles and filename
        :param stream_name: data stream name
        """
        rec_feat = self.orthogonalize_feats(scoring_weights, rec_feat).squeeze()
        lig_feat = self.orthogonalize_feats(scoring_weights, lig_feat).squeeze()

        boundW, crosstermW, bulkW = scoring_weights
        if plot_count == 0:
            print('\nLearned scoring coefficients')
            print('bound', str(boundW.item())[:6])
            print('crossterm', str(crosstermW.item())[:6])
            print('bulk', str(bulkW.item())[:6])
        plt.close()
        # plt.figure(figsize=(8, 8))
        # if rec_feat.shape[-1] < receptor.shape[-1]:
        #     pad_size = (receptor.shape[-1] - rec_feat.shape[-1]) // 2
        #     if rec_feat.shape[-1] % 2 == 0:
        #         rec_feat = F.pad(rec_feat, pad=([pad_size, pad_size, pad_size, pad_size]), mode='constant', value=0)
        #         lig_feat = F.pad(lig_feat, pad=([pad_size, pad_size, pad_size, pad_size]), mode='constant', value=0)
        #     else:
        #         rec_feat = F.pad(rec_feat, pad=([pad_size, pad_size + 1, pad_size, pad_size + 1]), mode='constant',
        #                          value=0)
        #         lig_feat = F.pad(lig_feat, pad=([pad_size, pad_size + 1, pad_size, pad_size + 1]), mode='constant',
        #                          value=0)

        # pad_size = (receptor.shape[-1]) // 2
        # receptor = F.pad(receptor, pad=([pad_size, pad_size, pad_size, pad_size]), mode='constant', value=0)
        # ligand = F.pad(ligand, pad=([pad_size, pad_size, pad_size, pad_size]), mode='constant', value=0)
        # rec_feat = F.pad(rec_feat, pad=([pad_size, pad_size, pad_size, pad_size]), mode='constant', value=0)
        # lig_feat = F.pad(lig_feat, pad=([pad_size, pad_size, pad_size, pad_size]), mode='constant', value=0)

        # pad_size3x = pad_size*3
        # receptor = receptor[:, pad_size:pad_size3x, pad_size:pad_size3x]
        # ligand = ligand[:, pad_size:pad_size3x, pad_size:pad_size3x]
        # rec_feat = rec_feat[:, pad_size:pad_size3x, pad_size:pad_size3x]
        # lig_feat = lig_feat[:, pad_size:pad_size3x, pad_size:pad_size3x]

        # rec_input_plot = np.hstack((receptor.squeeze().t().detach().cpu(), rec_bound.squeeze().t().detach().cpu()))
        # lig_input_plot = np.hstack((ligand.squeeze().t().detach().cpu(), lig_bound.squeeze().t().detach().cpu()))

        # plt.imshow(rec_input_plot, cmap='gist_heat_r')
        # plt.colorbar()
        # plt.show()

        # rec_feat_plot = np.hstack((rec_input_plot,
        #                       rec_feat[0].squeeze().t().detach().cpu(),
        #                       rec_feat[1].squeeze().t().detach().cpu()))
        # lig_feat_plot = np.hstack((lig_input_plot,
        #                       lig_feat[0].squeeze().t().detach().cpu(),
        #                       lig_feat[1].squeeze().t().detach().cpu()))
        #
        # norm = colors.CenteredNorm(vcenter=0.0) # center normalized color scale
        # stacked_image = np.vstack((rec_feat_plot, lig_feat_plot))

        rec_bulk, rec_bound = self.make_boundary(receptor.view(50,50))
        lig_bulk, lig_bound = self.make_boundary(ligand.view(50,50))
        rec_bulk, rec_bound = rec_bulk.squeeze().t().detach().cpu(), rec_bound.squeeze().t().detach().cpu()
        lig_bulk, lig_bound = lig_bulk.squeeze().t().detach().cpu(), lig_bound.squeeze().t().detach().cpu()

        rec_feat_bulk, rec_feat_bound = rec_feat[0].squeeze().t().detach().cpu(), rec_feat[1].squeeze().t().detach().cpu()
        lig_feat_bulk, lig_feat_bound = lig_feat[0].squeeze().t().detach().cpu(), lig_feat[1].squeeze().t().detach().cpu()

        # print(rec_bulk.shape, lig_bulk.shape)
        # print(rec_bound.shape, lig_bound.shape)
        # print(rec_feat_bulk.shape, lig_feat_bulk.shape)
        # print(rec_feat_bound.shape, lig_feat_bound.shape)

        figs_list = [[rec_bulk, rec_bound], [rec_feat_bulk, rec_feat_bound], [lig_bulk, lig_bound], [lig_feat_bulk, lig_feat_bound]]
        rows, cols = 2, 2
        fig, ax = plt.subplots(rows, cols, figsize=(8,8))
        plt.subplots_adjust(wspace=0, hspace=0)

        norm = colors.CenteredNorm(vcenter=0.0)  # center normalized color scale
        # norm = plt.colors.DivergingNorm(vcenter=0)
        shrink_bar = 0.8
        # extent = [0, 100, 0, 100]
        extent = None
        aspect = None
        cmap_data = 'binary'
        cmap_feats = 'seismic'
        # cmap_feats = 'binary'
        # cmap_feats = 'Spectral'
        # cmap_feats = 'coolwarm'
        # cmap_feats = 'PiYG'

        # plt.tight_layout()
        for i in range(rows):
            for j in range(cols):
                ax[i,j].grid(b=None)
                ax[i,j].axis('off')

                data = figs_list[i][j]

                if j == 0: #first col
                    # im = ax[i, j].imshow(data, cmap=cmap_data, extent=extent, aspect=aspect)
                    if i == 0: #first row
                        ax[i, j].set_title('input bulk')
                        im = ax[i, j].imshow(data, cmap=cmap_data)
                        tick_list = [0.0,  0.5, 1.0]
                        cb1 = plt.colorbar(im, ax=ax[i, j], shrink=shrink_bar, location='left', ticks=tick_list)
                        cb1.set_ticklabels(list(map(str, tick_list)))
                    if i == 1:
                        ax[i, j].set_title('learned bulk')
                        im = ax[i, j].imshow(data, cmap=cmap_feats, norm=norm)
                        # tick_list = [data.min().round().item(), 0, -data.min().round().item()]
                        cb2 = plt.colorbar(im, ax=ax[i, j], shrink=shrink_bar, location='left')#, ticks=tick_list)
                        # cb2.set_ticklabels(list(map(str, tick_list)))
                else:
                    # im = ax[i, j].imshow(data, cmap=cmap_data, extent=extent, aspect=aspect)
                    if i == 0: #first row
                        ax[i, j].set_title('input boundary')
                        im = ax[i, j].imshow(data/data.max(), cmap=cmap_data)
                        # tick_list = list(np.linspace(-data.max().item(), data.max().item(), 5, endpoint=False))
                        tick_list = [0.0,  0.5, 1.0]
                        cb1 = plt.colorbar(im, ax=ax[i, j], shrink=shrink_bar, location='right', ticks=tick_list)
                        cb1.set_ticklabels(list(map(str, tick_list)))
                        # cb1.set_ticks(list(map(str, tick_list)))
                        # cb1.set_ticks(tick_list)

                    if i == 1:
                        ax[i, j].set_title('learned boundary')
                        im = ax[i, j].imshow(data, cmap=cmap_feats, norm=norm)
                        # tick_list = [data.min().round().item(), 0, -data.min().round().item()]
                        cb2 = plt.colorbar(im, ax=ax[i, j], shrink=shrink_bar, location='right')#, ticks=tick_list)
                        # cb2.set_ticklabels(list(map(str, tick_list)))

                # if j == 0: #left column
                #     loc = 'left'
                #     if i == 0: #first row
                #         ax[i,j].set_title('input')
                #         im = ax[i, j].imshow(data, cmap=cmap_data, extent=extent, aspect=aspect)
                #         cb = plt.colorbar(im, ax=ax[i, j], shrink=shrink_bar, location=loc)
                #     if i == 1:#second row
                #         ax[i,j].set_title('learned bulk')
                #         im = ax[i, j].imshow(data, cmap=cmap_feats, norm=norm, extent=extent, aspect=aspect)
                #         # ticks = np.array([data.min(), data.max()])
                #         cb = plt.colorbar(im, ax=ax[i, j], shrink=shrink_bar, location=loc)
                #
                # else: #right column
                #     loc = 'right'
                #     if i == 0: #first row
                #         ax[i,j].set_title('boundary')
                #         im = ax[i, j].imshow(data, cmap=cmap_data, extent=extent, aspect=aspect)
                #         cb = plt.colorbar(im, ax=ax[i, j], shrink=shrink_bar, location=loc)
                #     if i == 1:#second row
                #         ax[i,j].set_title('learned boundary')
                #         im = ax[i, j].imshow(data, cmap=cmap_feats, norm=norm, extent=extent, aspect=aspect)
                #         # ticks = np.array([data.min(), data.max()])
                #         cb = plt.colorbar(im, ax=ax[i, j], shrink=shrink_bar, location=loc)

        # plt.imshow(stacked_image, cmap='cividis', norm=norm)  # plot scale limits
        # # plt.colorbar()
        # plt.colorbar(shrink=0.5, location='left')
        # plt.title('Input', loc='left')
        # plt.title('F1_bulk')
        # plt.title('F2_bound', loc='right')
        # plt.grid(False)
        # plt.tick_params(
        #     axis='x',  # changes apply to the x-axis
        #     which='both',  # both major and minor ticks are affected
        #     bottom=False,  # ticks along the bottom edge are off
        #     top=False,  # ticks along the top edge are off
        #     labelbottom=False)  # labels along the bottom
        # plt.tick_params(
        #     axis='y',
        #     which='both',
        #     left=False,
        #     right=False,
        #     labelleft=False)

        plt.savefig('Figs/Features_and_poses/'+stream_name+'_docking_feats'+'_example' + str(plot_count)+'.png', format='png')
        # plt.show()


if __name__ == '__main__':
    print('works')
