import torch
import torch.nn.functional as F

import matplotlib.pylab as plt
import seaborn as sea
sea.set_style("whitegrid")

from Dock2D.Utility.validation_metrics import RMSD
from Dock2D.Utility.UtilityFunctions import UtilityFunctions
import numpy as np


class TorchDockingFFT:
    def __init__(self, dim=100, num_angles=360, angle=None, swap_plot_quadrants=False, debug=False, normalization='ortho'):
        self.debug = debug
        self.swap_plot_quadrants = swap_plot_quadrants
        self.dim = dim
        self.num_angles = num_angles
        self.angle = angle
        if self.num_angles == 1 and angle:
            self.angles = angle.squeeze().unsqueeze(0).cuda()
        else:
            self.angles = torch.from_numpy(np.linspace(-np.pi, np.pi, num=self.num_angles)).cuda()

        self.norm = normalization
        self.onehot_3Dgrid = torch.zeros([self.num_angles, self.dim, self.dim], dtype=torch.double).cuda()

        self.UtilityFuncs = UtilityFunctions()

    def encode_transform(self, gt_rot, gt_txy):
        '''
        One hot encoded ground truth transformation into a 3D array.
        :param gt_rot: ground truth rotation in radians `[[gt_rot]]`.
        :param gt_txy: ground truth translation `[[x], [y]]`.
        :return: flattened one hot encoded array.
        '''
        deg_index_rot = (((gt_rot * 180.0/np.pi) + 180.0) % self.num_angles).type(torch.long)
        index_txy = gt_txy.type(torch.long)

        if self.num_angles == 1:
            deg_index_rot = 0

        self.onehot_3Dgrid[deg_index_rot, index_txy[0], index_txy[1]] = 1
        target_flatindex = torch.argmax(self.onehot_3Dgrid.flatten()).cuda()
        self.onehot_3Dgrid[deg_index_rot, index_txy[0], index_txy[1]] = 0

        return target_flatindex

    def extract_transform(self, pred_score):
        pred_index = torch.argmax(pred_score)
        pred_rot = (torch.div(pred_index, self.dim ** 2) * torch.div(np.pi, 180)) - np.pi
        XYind = torch.remainder(pred_index, self.dim ** 2)
        if self.swap_plot_quadrants:
            pred_X = torch.div(XYind, self.dim, rounding_mode='floor') - self.dim//2
            pred_Y = torch.fmod(XYind, self.dim) - self.dim//2
        else:
            pred_X = torch.div(XYind, self.dim, rounding_mode='floor')
            pred_Y = torch.fmod(XYind, self.dim)

        # Just to make translation values look nice in terms of + or - signs
        if pred_X > self.dim//2:
            pred_X = pred_X - self.dim
        if pred_Y > self.dim//2:
            pred_Y = pred_Y - self.dim
        return pred_rot, torch.stack((pred_X, pred_Y), dim=0)

    @staticmethod
    def make_boundary(grid_shape):
        grid_shape = grid_shape.unsqueeze(0).unsqueeze(0)
        epsilon = 1e-5
        sobel_top = torch.tensor([[[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]], dtype=torch.float).cuda()
        sobel_left = sobel_top[0,0,:,:].t().view(1,1,3,3)

        feat_top = F.conv2d(grid_shape, weight=sobel_top, padding=1)
        feat_left = F.conv2d(grid_shape, weight=sobel_left, padding=1)

        top = feat_top + epsilon
        right = feat_left + epsilon
        boundary = torch.sqrt(top ** 2 + right ** 2)
        feat_stack = torch.cat([grid_shape, boundary], dim=1)

        return feat_stack.squeeze()

    def dock_global(self, receptor, ligand, weight_bound, weight_crossterm1, weight_crossterm2, weight_bulk):
        initbox_size = receptor.shape[-1]
        pad_size = initbox_size // 2

        f_rec = receptor.unsqueeze(0).repeat(self.num_angles,1,1,1)
        f_lig = ligand.unsqueeze(0).repeat(self.num_angles,1,1,1)
        rot_lig = self.UtilityFuncs.rotate(f_lig, self.angles)

        if initbox_size % 2 == 0:
            f_rec = F.pad(f_rec, pad=([pad_size, pad_size, pad_size, pad_size]), mode='constant', value=0)
            rot_lig = F.pad(rot_lig, pad=([pad_size, pad_size, pad_size, pad_size]), mode='constant', value=0)
        else:
            f_rec = F.pad(f_rec, pad=([pad_size, pad_size+1, pad_size, pad_size+1]), mode='constant', value=0)
            rot_lig = F.pad(rot_lig, pad=([pad_size, pad_size+1, pad_size, pad_size+1]), mode='constant', value=0)

        if self.debug:
            with torch.no_grad():
                step = 30
                for i in range(rot_lig.shape[0]):
                    print(self.angle)
                    if i % step == 0:
                        plt.title('Torch '+str(i)+' degree rotation')
                        plt.imshow(rot_lig[i,0,:,:].detach().cpu())
                        plt.show()

        score = self.dock_translations(f_rec, rot_lig, weight_bound, weight_crossterm1, weight_crossterm2, weight_bulk)

        return score

    def dock_translations(self, receptor, ligand, weight_bound, weight_crossterm1, weight_crossterm2, weight_bulk):
        num_feats_per_shape = 2
        receptor_bulk, receptor_bound = torch.chunk(receptor, chunks=num_feats_per_shape, dim=1)
        ligand_bulk, ligand_bound = torch.chunk(ligand, chunks=num_feats_per_shape, dim=1)
        receptor_bulk = receptor_bulk.squeeze()
        receptor_bound = receptor_bound.squeeze()
        ligand_bulk = ligand_bulk.squeeze()
        ligand_bound = ligand_bound.squeeze()

        # Bulk score
        cplx_rec = torch.fft.rfft2(receptor_bulk, dim=(-2, -1), norm=self.norm)
        cplx_lig = torch.fft.rfft2(ligand_bulk, dim=(-2, -1), norm=self.norm)
        trans_bulk = torch.fft.irfft2(cplx_rec * torch.conj(cplx_lig), dim=(-2, -1), norm=self.norm)

        # Boundary score
        cplx_rec = torch.fft.rfft2(receptor_bound, dim=(-2, -1), norm=self.norm)
        cplx_lig = torch.fft.rfft2(ligand_bound, dim=(-2, -1), norm=self.norm)
        trans_bound = torch.fft.irfft2(cplx_rec * torch.conj(cplx_lig), dim=(-2, -1), norm=self.norm)

        # Boundary - bulk score
        cplx_rec = torch.fft.rfft2(receptor_bound, dim=(-2, -1), norm=self.norm)
        cplx_lig = torch.fft.rfft2(ligand_bulk, dim=(-2, -1), norm=self.norm)
        trans_bulk_bound = torch.fft.irfft2(cplx_rec * torch.conj(cplx_lig), dim=(-2, -1), norm=self.norm)

        # Bulk - boundary score
        cplx_rec = torch.fft.rfft2(receptor_bulk, dim=(-2, -1), norm=self.norm)
        cplx_lig = torch.fft.rfft2(ligand_bound, dim=(-2, -1), norm=self.norm)
        trans_bound_bulk = torch.fft.irfft2(cplx_rec * torch.conj(cplx_lig), dim=(-2, -1), norm=self.norm)

        ## cross-term score maximizing
        score = weight_bound * trans_bound + weight_crossterm1 * trans_bulk_bound + weight_crossterm2 * trans_bound_bulk - weight_bulk * trans_bulk

        if self.swap_plot_quadrants:
            return self.UtilityFuncs.swap_quadrants(score)
        else:
            return score

    def check_FFT_predictions(self, fft_score, receptor, ligand, gt_txy, gt_rot):
        print('\n'+'*'*50)

        pred_rot, pred_txy = self.extract_transform(fft_score)
        energies = -fft_score
        rmsd_out = RMSD(ligand, gt_rot, gt_txy, pred_rot, pred_txy).calc_rmsd()
        print('extracted predicted indices', pred_rot, pred_txy)
        print('gt indices', gt_rot, gt_txy)
        print('RMSD', rmsd_out.item())
        print()
        plt.title('docking energy surface per shape')
        plt.grid(False)
        if self.num_angles == 1:
            plt.imshow(energies.detach().cpu())
            plt.colorbar()
            plt.show()
        else:
            plt.imshow(energies[pred_rot.long(), :, :].detach().cpu())
            plt.colorbar()
            plt.show()

        pair = UtilityFunctions().plot_assembly(receptor.detach().cpu(), ligand.detach().cpu().numpy(),
                                                gt_rot.detach().cpu().numpy(), gt_txy.detach().cpu().numpy(),
                                                pred_rot.detach().cpu().numpy(), pred_txy.detach().cpu().numpy())
        plt.imshow(pair.transpose())
        plt.show()

if __name__ == '__main__':

    from DeepProteinDocking2D.Utility.torchDataLoader import get_docking_stream
    from tqdm import tqdm

    testset = '../Datasets/docking_test_100pool'
    max_size = None
    data_stream = get_docking_stream(testset + '.pkl', batch_size=1, max_size=max_size)

    swap_quadrants = True
    FFT = TorchDockingFFT(swap_plot_quadrants=swap_quadrants, normalization="ortho")

    weight_bound, weight_crossterm1, weight_crossterm2, weight_bulk = 10, 20, 20, 200

    for data in tqdm(data_stream):
        receptor, ligand, gt_rot, gt_txy = data

        receptor = receptor.squeeze()
        ligand = ligand.squeeze()
        gt_rot = gt_rot.squeeze()
        gt_txy = gt_txy.squeeze()

        receptor = receptor.cuda()
        ligand = ligand.cuda()
        gt_rot = gt_rot.cuda()
        gt_txy = gt_txy.cuda()

        receptor_stack = FFT.make_boundary(receptor)
        ligand_stack = FFT.make_boundary(ligand)
        fft_score = FFT.dock_global(receptor_stack, ligand_stack, weight_bound, weight_crossterm1, weight_crossterm2, weight_bulk)
        rot, trans = FFT.extract_transform(fft_score)
        print(rot, trans)
        lowest_energy = -fft_score[rot.long(), trans[0], trans[1]].detach().cpu()
        print('lowest energy', lowest_energy)
        FFT.check_FFT_predictions(fft_score, receptor, ligand, gt_txy, gt_rot)
