import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from Dock2D.Utility.TorchDataLoader import get_docking_stream, get_interaction_stream
from Dock2D.Utility.TorchDockingFFT import TorchDockingFFT
from Dock2D.Utility.UtilityFunctions import UtilityFunctions


def homodimer_vs_heterodimer(setname, interaction=False, plot_all_poses=False):
    dataset = '../Datasets/'+ setname + '_400pool.pkl'

    FFT = TorchDockingFFT(padded_dim=100, num_angles=360)

    homodimer_Elist = []
    # heterodimer_Elist = []
    # Energy_list = []

    weight_bound, weight_crossterm, weight_bulk = 10, 20, 200
    angle = None
    interacting_example = False
    gt_rot, gt_txy = None, None

    if interaction:
        data_stream = get_interaction_stream(dataset, number_of_pairs=None)
    else:
        data_stream = get_docking_stream(dataset, max_size=None)

    for data in tqdm(data_stream):
        if interaction:
            receptor, ligand, gt_interaction = data
            if gt_interaction == 1:
                positive_interaction = True
            else:
                positive_interaction = False
        else:
            receptor, ligand, gt_rot, gt_txy = data
            positive_interaction = True

        receptor = receptor.to(device='cuda', dtype=torch.float).squeeze()
        ligand = ligand.to(device='cuda', dtype=torch.float).squeeze()

        # rec_single_dim_sum = torch.sum(receptor)
        # lig_single_dim_sum = sum(ligand)
        #
        # print(rec_single_dim_sum)
        # print(lig_single_dim_sum)
        # print(rec_single_dim_sum - lig_single_dim_sum)

        shape_diff = abs(receptor.detach().cpu() - ligand.detach().cpu())
        sum_shape_diff = torch.sum(shape_diff)

        if positive_interaction and sum_shape_diff == 0:
            # plt.imshow(shape_diff)
            # plt.colorbar()
            # plt.show()
            receptor_stack = UtilityFunctions().make_boundary(receptor)
            ligand_stack = UtilityFunctions().make_boundary(ligand)
            fft_score = FFT.dock_rotations(receptor_stack, ligand_stack, angle,
                                           weight_bound, weight_crossterm, weight_bulk)

            rot, trans = FFT.extract_transform(fft_score)
            lowest_energy = -fft_score[rot.long(), trans[0], trans[1]].detach().cpu()

            homodimer_Elist.append(lowest_energy.detach().cpu())

            # Energy_list.append(float(lowest_energy.detach().cpu()))

            # if not interaction:
            #     if plot_all_poses:
            #         plt.close()
            #         pair = UtilityFunctions().plot_assembly(receptor.cpu(), ligand.cpu(), gt_rot.cpu(), gt_txy.cpu(), rot.cpu(),
            #                                                 trans.cpu())
            #         plt.imshow(pair.transpose())
            #         plt.title('Energy' + str(lowest_energy.item()))
            #         plt.show()


    # binwidth = 1
    # bins = np.arange(min(Energy_list), max(Energy_list) + binwidth, binwidth)
    #
    # plt.hist([homodimer_Elist], label=['homodimer'], bins=bins, alpha=0.33)
    # plt.hist([heterodimer_Elist], label=['heterodimer'], bins=bins, alpha=0.33)
    # plt.legend(['homodimers', 'heterodimers'])
    #
    # title = 'homodimer_vs_heterodimer_' + dataset + '_lowest energies'
    # plt.xlabel('Energy')
    # plt.ylabel('Counts')
    # plt.title(title)
    # plt.savefig('Figs/' + dataset.split('/')[-1] + '.png')
    #
    # plt.show()

    print(setname + ' set stats:\n')
    print('number of homodimers', len(homodimer_Elist))
    # print('number of heterodimers', len(heterodimer_Elist))
    # print('percent homodimer of examples with E < -90',
    #       len(homodimer_Elist) / (len(homodimer_Elist) + len(heterodimer_Elist)))


if __name__ == "__main__":
    plot_all_poses = False

    homodimer_vs_heterodimer('docking_train', plot_all_poses)
    homodimer_vs_heterodimer('docking_valid', plot_all_poses)
    homodimer_vs_heterodimer('docking_test', plot_all_poses)

    interaction = True
    homodimer_vs_heterodimer('interaction_train', interaction, plot_all_poses)
    homodimer_vs_heterodimer('interaction_valid', interaction, plot_all_poses)
    homodimer_vs_heterodimer('interaction_test', interaction, plot_all_poses)
