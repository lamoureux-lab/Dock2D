import torch
import numpy as np
from DeepProteinDocking2D.Models.BruteForce.utility_functions import plot_assembly
import matplotlib.pyplot as plt
import pandas as pd

class EBMPlotter:
    def __init__(self, model, experiment=None, plot_freq=1000):
        self.model = model
        self.experiment = experiment
        self.plot_freq = plot_freq

    def plot_deltaF_distribution(self, filename, path, plot_epoch=1, show=False):
        plt.close()
        # print('plotting deltaF distribution')
        # Plot RMSD distribution of all samples across epoch
        train = pd.read_csv(filename+'.txt', sep='\t', header=0, names=['F', 'F_0', 'Label'])

        fig, ax = plt.subplots(figsize=(10, 10))
        plt.suptitle('deltaF distribution: epoch' + str(plot_epoch) + ' ' + self.experiment)
        # print(train)
        labels = train.Label.unique()
        F = train['F']
        binwidth = max(F) - min(F) // 10
        bins = np.arange(min(F), max(F) + binwidth, binwidth)
        y, x, _ = plt.hist([train.loc[train.Label == x, 'F'] for x in labels], label=labels, bins=bins, rwidth=binwidth,
                           color=['g', 'r'], alpha=0.5)
        plt.vlines(train['F_0'].to_numpy()[-1], ymin=-1, ymax=y.max() + 1, linestyles='dashed', label='F_0', colors='k')
        # ax.set_xticks(np.arange(int(x.min()) - 1, int(x.max()) + 1, num_ticks))
        plt.legend(('(+) interaction', ' (-) interaction', 'final F_0'), prop={'size': 10})
        ax.set_ylabel('Training set counts')
        ax.set_xlabel('Free Energy (F)')
        ax.grid(visible=True)

        plt.savefig(path+'/freeE_hist'+str(plot_epoch))
        if show:
            plt.show()

    def FI_energy_vs_F0(self, pos_list, neg_list, pos_ind, neg_ind, F_0, filename, epoch):
        plt.close()
        # plot at epoch level
        # plot scatter of Free energy mins for positive and negative examples
        plot_limit = 100
        plt.title('Epoch'+str(epoch)+' Free Energies and F_0')
        plt.ylabel('F')
        plt.xlabel('Examples')
        plot_min = max(min([min(pos_list), min(neg_list), F_0]), -plot_limit)
        plot_max = min(max([max(pos_list), max(neg_list), F_0]), +plot_limit)
        plt.ylim([plot_min, plot_max])
        plt.scatter(pos_ind, pos_list, c='g', marker='v', alpha=0.50)
        plt.scatter(neg_ind, neg_list, c='r', marker='^', alpha=0.33)
        plt.hlines(F_0, xmin=-5, xmax=max(pos_ind[-1], neg_ind[-1])+5)
        plt.legend(['Interaction','Non-interaction','Learned F_0'], loc='upper right')
        plt.savefig(filename)
        plt.close()

    def plot_pose(self, receptor, ligand, rotation, translation, plot_title, filename, pos_idx, epoch, gt_rot=0,
                  gt_txy=(0, 0), pred_interact=None, gt_interact=None, plot_LD=False, Energyscored=None, LDindex=None):
        plt.close()
        if pos_idx % self.plot_freq == 0:
            pair = plot_assembly(receptor.squeeze().detach().cpu().numpy(),
                                 ligand.squeeze().detach().cpu().numpy(),
                                 rotation.detach().cpu().numpy(),
                                 (translation.squeeze()[0].detach().cpu().numpy(),
                                  translation.squeeze()[1].detach().cpu().numpy()),
                                 gt_rot,
                                 gt_txy)

            header = 'example' + str(pos_idx.item()) + '_epoch' + str(epoch + 1)+ 'EnergyScore=' + str(Energyscored.item())[:4] +\
                     ' x=' + str(translation.squeeze()[0].item())[:4] + 'y=' + str(translation.squeeze()[1].item())[:4] + '\n'
            if plot_LD:
                plt.imshow(pair[100:, :].transpose())
                plt.title('EBM Input', loc='left')
                plt.suptitle(header+'LDindex' + str(LDindex+1))
                plt.title(plot_title, loc='right')

            else:
                plt.imshow(pair[200:, :].transpose())
                # plt.suptitle('example' + str(pos_idx.item()) + '_epoch' + str(epoch + 1))
                plt.suptitle(header)
            if gt_interact is not None and pred_interact is not None:
                plt.title('Interaction: gt=' + str(gt_interact) + ' pred=' + str(pred_interact)[:3])

            plt.savefig(filename)
            plt.close()

    def plot_feats(self, neg_rec_feat, neg_lig_feat, epoch, pos_idx, filename):
        plt.close()
        if pos_idx % self.plot_freq == 0:
            with torch.no_grad():
                # neg_rec_feat = neg_rec_feat.detach().cpu()
                # neg_lig_feat = neg_lig_feat.detach().cpu()
                neg_rec_feat, neg_lig_feat = self.eigenvec_feats(neg_rec_feat.detach().cpu(),
                                                                 neg_lig_feat.detach().cpu())
                neg_rec_bulk, neg_rec_bound = neg_rec_feat.squeeze()[0, :, :], neg_rec_feat.squeeze()[1, :, :]
                neg_lig_bulk, neg_lig_bound = neg_lig_feat.squeeze()[0, :, :], neg_lig_feat.squeeze()[1, :, :]
                lig_plot = np.hstack((neg_lig_bulk, neg_lig_bound))
                rec_plot = np.hstack((neg_rec_bulk, neg_rec_bound))
                neg_plot = np.vstack((rec_plot, lig_plot))
                plt.imshow(neg_plot)
                plt.colorbar()
                plt.title('Bulk', loc='left')
                plt.title('Epoch'+str(epoch)+' Ex' + str(pos_idx.item()))
                plt.title('Boundary', loc='right')

                plt.savefig(filename)
                plt.close()

    def eigenvec_feats(self, neg_rec_feat, neg_lig_feat):
        plt.close()
        A = self.model.scorer[0].weight.view(2, 2).detach().cpu().clone()
        eigvals, V = torch.linalg.eig(A)
        V = V.real
        rv01 = V[0, 0] * neg_rec_feat[:, 0, :, :] + V[1, 0] * neg_rec_feat[:, 1, :, :]
        rv02 = V[0, 1] * neg_rec_feat[:, 0, :, :] + V[1, 1] * neg_rec_feat[:, 1, :, :]
        repr_0 = torch.stack([rv01, rv02], dim=0).unsqueeze(dim=0).detach()
        # print(V)

        rv01 = V[0, 0] * neg_lig_feat[:, 0, :, :] + V[1, 0] * neg_lig_feat[:, 1, :, :]
        rv02 = V[0, 1] * neg_lig_feat[:, 0, :, :] + V[1, 1] * neg_lig_feat[:, 1, :, :]
        repr_1 = torch.stack([rv01, rv02], dim=0).unsqueeze(dim=0).detach()
        # print(V)

        reprs = torch.cat([repr_0, repr_1], dim=0)

        return reprs

    def plot_IP_energy_loss(self, L_p, L_n, epoch, pos_idx, filename):
        plt.close()
        print('L_p, L_n', L_p, L_n)
        f, ax = plt.subplots(figsize=(6, 6))

        axes_lim = (-0.25, 0.25)
        ax.scatter(L_n, L_p, c=".3")
        ax.plot(axes_lim, axes_lim, ls="--", c=".3")
        ax.set(xlim=axes_lim, ylim=axes_lim)
        ax.set_ylabel('L_p')
        ax.set_xlabel('L_n two temp simulation ')
        plt.quiver([0], [0], [L_n], [L_p], angles='xy', scale_units='xy', scale=1)
        plt.quiver([0], [L_p], color=['r'], angles='xy', scale_units='xy', scale=1)
        plt.quiver([L_n], [0], color=['b'], angles='xy', scale_units='xy', scale=1)
        plt.title(
            'IP Loss: Difference in L_p and L_n\n' + 'epoch ' + str(epoch) + ' example number' + str(pos_idx.item()))

        plt.savefig(filename)
        plt.close()

    def plot_energy_and_pose(self, pos_idx, L_p, L_n, epoch, receptor, ligand, pos_alpha, pos_dr, neg_alpha, neg_dr,
                             filename):
        if pos_idx % self.plot_freq == 0:
            print('PLOTTING LOSS')
            self.plot_IP_energy_loss(L_p.detach().cpu().numpy(), L_n.squeeze().detach().cpu().numpy(), epoch, pos_idx,
                                     filename)
            print('PLOTTING PREDICTION')
            self.plot_pose(receptor, ligand, neg_alpha.squeeze(), neg_dr.squeeze(), 'Pose after LD',
                           filename, pos_idx, epoch,
                           pos_alpha.squeeze().detach().cpu(), pos_dr.squeeze().detach().cpu())
