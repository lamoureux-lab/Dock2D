import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from Dock2D.Utility.TorchDataLoader import get_docking_stream
from Dock2D.Utility.TorchDockingFFT import TorchDockingFFT
from Dock2D.Utility.UtilityFunctions import UtilityFunctions

if __name__ == "__main__":

    # dataset = 'docking_test_100pool.pkl'
    dataset = 'docking_train_400pool.pkl'
    max_size = None
    data_stream = get_docking_stream(dataset, max_size=max_size)

    FFT = TorchDockingFFT()

    plot_all_poses = False

    homodimer_Elist = []
    heterodimer_Elist = []
    Energy_list = []

    weight_bound, weight_crossterm1, weight_crossterm2, weight_bulk = 10, 20, 20, 200

    for data in tqdm(data_stream):
        receptor, ligand, gt_rot, gt_txy = data

        receptor = receptor.to(device='cuda', dtype=torch.float).squeeze()
        ligand = ligand.to(device='cuda', dtype=torch.float).squeeze()
        gt_rot = gt_rot.to(device='cuda', dtype=torch.float).squeeze()
        gt_txy = gt_txy.to(device='cuda', dtype=torch.float).squeeze()

        receptor_stack = FFT.make_boundary(receptor)
        ligand_stack = FFT.make_boundary(ligand)
        fft_score = FFT.dock_global(receptor_stack, ligand_stack,
                                    weight_bound, weight_crossterm1, weight_crossterm2, weight_bulk)

        rot, trans = FFT.extract_transform(fft_score)
        lowest_energy = -fft_score[rot.long(), trans[0], trans[1]].detach().cpu()
        Energy_list.append(float(lowest_energy.detach().cpu()))

        if plot_all_poses:
            plt.close()
            pair = UtilityFunctions().plot_assembly(receptor.cpu(), ligand.cpu(), gt_rot.cpu(), gt_txy.cpu(), rot.cpu(), trans.cpu())
            plt.imshow(pair.transpose())
            plt.title('Energy'+str(lowest_energy.item()))
            plt.show()

        if torch.sum(receptor - ligand) == 0:
            # print('homodimer found')
            homodimer_Elist.append(lowest_energy)
            # plt.imshow(np.hstack((receptor.detach().cpu(), ligand.detach().cpu())))
            # plt.show()
        else:
            # print('heterodimer...')
            heterodimer_Elist.append(lowest_energy)
            # plt.imshow(np.hstack((receptor.detach().cpu(), ligand.detach().cpu())))
            # plt.show()

    binwidth = 1
    bins = np.arange(min(Energy_list), max(Energy_list) + binwidth, binwidth)

    plt.hist([homodimer_Elist], label=['homodimer'], bins=bins, alpha=0.33)
    plt.hist([heterodimer_Elist], label=['heterodimer'], bins=bins, alpha=0.33)
    plt.legend(['homodimers', 'heterodimers'])

    title = 'homodimer_vs_heterodimer_' + dataset + '_lowest energies'
    plt.xlabel('Energy')
    plt.ylabel('Counts')
    plt.title(title)
    plt.savefig('Figs/' + title + '.png')

    plt.show()

    print('number of homodimers', len(homodimer_Elist))
    print('number of heterodimers', len(heterodimer_Elist))
