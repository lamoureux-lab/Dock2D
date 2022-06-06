import scipy.ndimage as ndimage
import _pickle as pkl
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import torch
from torch.nn import functional as F
from Dock2D.Utility.ValidationMetrics import RMSD


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

    def plot_rotation_energysurface(self, fft_score, pred_txy, num_angles=360, stream_name=None, plot_count=0):
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

        xrange = np.arange(-np.pi, np.pi, 2 * np.pi / num_angles)
        hardmin_minEnergies = stream_name + '_energysurface' + '_example' + str(plot_count)
        plt.plot(xrange, mintxy_energies)
        plt.title('Best Scoring Translation Energy Surface')
        plt.ylabel('Energy')
        plt.xlabel('Rotation (rads)')
        plt.savefig('Figs/EnergySurfaces/' + hardmin_minEnergies + '.png')

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

    def plot_assembly(self, receptor, ligand, gt_rot, gt_txy, pred_rot=None, pred_txy=None):
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
        A = torch.tensor([[bulkW, crosstermW],[crosstermW,boundW]])
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
        plt.figure(figsize=(8, 8))
        if rec_feat.shape[-1] < receptor.shape[-1]:
            pad_size = (receptor.shape[-1] - rec_feat.shape[-1]) // 2
            if rec_feat.shape[-1] % 2 == 0:
                rec_feat = F.pad(rec_feat, pad=([pad_size, pad_size, pad_size, pad_size]), mode='constant', value=0)
                lig_feat = F.pad(lig_feat, pad=([pad_size, pad_size, pad_size, pad_size]), mode='constant', value=0)
            else:
                rec_feat = F.pad(rec_feat, pad=([pad_size, pad_size + 1, pad_size, pad_size + 1]), mode='constant',
                                 value=0)
                lig_feat = F.pad(lig_feat, pad=([pad_size, pad_size + 1, pad_size, pad_size + 1]), mode='constant',
                                 value=0)

        pad_size = (receptor.shape[-1]) // 2
        receptor = F.pad(receptor, pad=([pad_size, pad_size, pad_size, pad_size]), mode='constant', value=0)
        ligand = F.pad(ligand, pad=([pad_size, pad_size, pad_size, pad_size]), mode='constant', value=0)
        rec_feat = F.pad(rec_feat, pad=([pad_size, pad_size, pad_size, pad_size]), mode='constant', value=0)
        lig_feat = F.pad(lig_feat, pad=([pad_size, pad_size, pad_size, pad_size]), mode='constant', value=0)
        rec_plot = np.hstack((receptor.squeeze().t().detach().cpu(),
                              rec_feat[0].squeeze().t().detach().cpu(),
                              rec_feat[1].squeeze().t().detach().cpu()))
        lig_plot = np.hstack((ligand.squeeze().t().detach().cpu(),
                              lig_feat[0].squeeze().t().detach().cpu(),
                              lig_feat[1].squeeze().t().detach().cpu()))

        norm = colors.CenteredNorm(vcenter=0.0) # center normalized color scale
        stacked_image = np.vstack((rec_plot, lig_plot))
        plt.imshow(stacked_image, cmap='coolwarm', norm=norm)  # plot scale limits
        # plt.colorbar()
        plt.colorbar(shrink=0.5, location='left')
        plt.title('Input', loc='left')
        plt.title('F1_bulk')
        plt.title('F2_bound', loc='right')
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
        plt.savefig('Figs/Features_and_poses/'+stream_name+'_docking_feats'+'_example' + str(plot_count)+'.png')
        # plt.show()


if __name__ == '__main__':
    print('works')
