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
# rcParams.update({'font.size': 14})


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

    def gaussian1D(self, M, mean, sigma, a=1, gaussian_norm=False):
        '''
        Create 1D gaussian vector
        '''
        if gaussian_norm:
            a = 1 / (sigma * np.sqrt(2 * np.pi))
        else:
            a = a
        # x = torch.arange(0, M) - (M - 1.0) / 2.0 ## don't substract by 1
        x = torch.arange(0, M) - (M / 2)
        var = 2 * sigma ** 2
        w = a * torch.exp(-((x - mean) ** 2 / var))
        return w

    def gaussian2D(self, kernlen=50, mean=0, sigma=5.0, a=1, gaussian_norm=False):
        '''
        Use the outer product of two gaussian vectors to create 2D gaussian
        '''
        gkernel1d = self.gaussian1D(kernlen, mean=mean, sigma=sigma, a=a, gaussian_norm=gaussian_norm)
        gkernel2d = torch.outer(gkernel1d, gkernel1d)
        return gkernel2d

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

    def make_boundary(self, grid_shape, gaussian_blur_bulk=False):
        """
        Create the boundary feature for data generation and unit testing.

        :param grid_shape: input shape grid image
        :return: features stack with original shape as "bulk" and created "boundary"
        """
        grid_shape = grid_shape.unsqueeze(0).unsqueeze(0)
        sobel_top = torch.tensor([[[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]], dtype=torch.float).cuda()
        sobel_left = sobel_top[0, 0, :, :].t().view(1, 1, 3, 3)

        feat_top = F.conv2d(grid_shape, weight=sobel_top, padding=1)
        feat_left = F.conv2d(grid_shape, weight=sobel_left, padding=1)

        if gaussian_blur_bulk:
            debug = False
            kernlen=5
            sigma=1
            padding = kernlen//2
            gaussian_filter = self.gaussian2D(kernlen=kernlen, mean=0, sigma=sigma, a=1, gaussian_norm=True).view(1, 1, kernlen, kernlen).cuda()

            gaussian_filter = gaussian_filter/torch.sum(gaussian_filter)
            grid_shape_blurred = F.conv2d(grid_shape, weight=gaussian_filter, padding=padding)

            if debug:
                kernel_sum = torch.sum(gaussian_filter)
                blurred_bulk_plot = grid_shape_blurred.squeeze().detach().cpu()
                raw_bulk_plot = grid_shape.squeeze().detach().cpu()
                plt.title('original VS. blurred; k='+str(kernlen)+'x'+str(kernlen)+' sig='+str(sigma)+
                          '\nkernel_sum='+ str(kernel_sum.item())[:4] +'blurred maxval='+str(grid_shape.max().item())[:4])
                figure = np.hstack((raw_bulk_plot, blurred_bulk_plot))
                plt.imshow(figure)
                plt.colorbar()
                plt.show()

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

    def plot_rotation_energysurface(self, fft_score, pred_txy, num_angles=360, stream_name=None, plot_count=0, plot_pub=False):
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
        # if num_angles == 1:
        #     minimumEnergy = fft_score[pred_txy[0], pred_txy[1]].detach().cpu()
        #     mintxy_energies.append(minimumEnergy)
        # else:
        for i in range(num_angles):
            minimumEnergy = fft_score[i, pred_txy[0], pred_txy[1]].detach().cpu()
            mintxy_energies.append(minimumEnergy)

        fig = plt.figure(figsize=(8,5))
        plt.rcParams['axes.xmargin'] = 0
        plt.rcParams['axes.ymargin'] = 0.05
        # rcParams.update({'font.size': 14})

        axarr = fig.add_subplot(1,1,1)
        xrange = np.arange(-np.pi, np.pi, 2 * np.pi / num_angles)
        plt.xticks(np.round(np.linspace(-np.pi, np.pi, 5, endpoint=True), decimals=2))
        plt.xlim([-np.pi, np.pi])
        plt.plot(xrange, mintxy_energies)
        if plot_pub:
            format = 'pdf'
        else:
            format = 'png'
            plt.title('Best Scoring Translation Energy Surface')

        hardmin_minEnergies = stream_name + '_energysurface' + '_example' + str(plot_count)
        plt.ylabel('Energy')
        plt.xlabel('Rotation (rads)')
        plt.savefig('Figs/EnergySurfaces/' + hardmin_minEnergies + '.'+format, format=format)
        # plt.show()

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
            padding = box_size//2
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

        # gt_rot = (gt_rot * 180.0/np.pi)
        # gt_transformlig = self.rotate_gridligand(ligand_copy, gt_rot)
        gt_transformlig = self.rotate(torch.tensor(ligand_copy).unsqueeze(0).unsqueeze(0), torch.tensor(gt_rot).unsqueeze(0))
        gt_transformlig = np.clip(self.translate_gridligand(gt_transformlig.squeeze().detach().cpu(), gt_txy[0], gt_txy[1]), a_min=0, a_max=1)
        receptor_copy = np.clip(receptor_copy, a_min=0, a_max=2)

        gt_transformlig += receptor_copy
        gt_transformlig[gt_transformlig > 2.05] = 0.5

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
            # pair = np.clip(gt_transformlig, a_min=0, a_max=2)
            return gt_transformlig
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


        rec_bulk, rec_bound = self.make_boundary(receptor.view(50,50))
        lig_bulk, lig_bound = self.make_boundary(ligand.view(50,50))
        rec_bulk, rec_bound = rec_bulk.squeeze().t().detach().cpu(), rec_bound.squeeze().t().detach().cpu()
        lig_bulk, lig_bound = lig_bulk.squeeze().t().detach().cpu(), lig_bound.squeeze().t().detach().cpu()

        rec_feat_bulk, rec_feat_bound = rec_feat[0].squeeze().t().detach().cpu(), rec_feat[1].squeeze().t().detach().cpu()
        lig_feat_bulk, lig_feat_bound = lig_feat[0].squeeze().t().detach().cpu(), lig_feat[1].squeeze().t().detach().cpu()

        figs_list = [[rec_bulk, rec_bound], [rec_feat_bulk, rec_feat_bound], [lig_bulk, lig_bound], [lig_feat_bulk, lig_feat_bound]]
        rows, cols = 2, 2
        fig, ax = plt.subplots(rows, cols, figsize=(8,8))
        plt.subplots_adjust(wspace=0, hspace=0)

        norm = colors.CenteredNorm(vcenter=0.0)  # center normalized color scale
        # norm = plt.colors.DivergingNorm(vcenter=0)
        shrink_bar = 0.8
        # extent = [0, 100, 0, 100]
        # extent = None
        # aspect = None
        cmap_data = 'binary'
        cmap_feats = 'seismic'

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

        plt.savefig('Figs/Features_and_poses/'+stream_name+'_docking_feats'+'_example' + str(plot_count)+'.png', format='png')
        # plt.show()


if __name__ == '__main__':
    print('works')
