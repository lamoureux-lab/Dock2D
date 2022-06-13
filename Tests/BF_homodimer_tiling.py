from Dock2D.Models.TrainerIP import *
import random
from Dock2D.Utility.TorchDataLoader import get_docking_stream
from torch import optim
from Dock2D.Utility.PlotterIP import PlotterIP
from Dock2D.Utility.TorchDockingFFT import TorchDockingFFT
from Dock2D.Models.model_sampling import SamplingModel


if __name__ == '__main__':
    #################################################################################
    # Datasets
    trainset = '../Datasets/docking_train_400pool.pkl'
    validset = '../Datasets/docking_valid_400pool.pkl'
    ### testing set
    testset = '../Datasets/docking_test_400pool.pkl'
    #########################
    #### initialization of random seeds
    random_seed = 42
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(0)
    # torch.autograd.set_detect_anomaly(True)
    ######################
    max_size = 50
    train_stream = get_docking_stream(trainset, max_size=max_size)
    valid_stream = get_docking_stream(validset, max_size=max_size)
    test_stream = get_docking_stream(testset, max_size=max_size)
    sample_buffer_length = max(len(train_stream), len(valid_stream), len(test_stream))
    ######################
    experiment = 'BS_lr-2_30ep_latest_400poolcheck'

    ######################
    train_epochs = 13
    lr = 10 ** -2
    #####################
    padded_dim = 100
    num_angles = 1
    sampledFFT = TorchDockingFFT(padded_dim=padded_dim, num_angles=num_angles)
    model = SamplingModel(sampledFFT, IP=True).to(device=0)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    Trainer = TrainerIP(sampledFFT, model, optimizer, experiment)

    ### Resume training for validation sets
    # Trainer.run_trainer(
    #     train_epochs=1, train_stream=None, valid_stream=valid_stream, #test_stream=valid_stream,
    #     resume_training=True, resume_epoch=train_epochs)

    ## Brute force evaluation and plotting


    import matplotlib.pyplot as plt
    import torch.nn.functional as F
    plotting = True
    show=True
    eval_angles = 360


    stream_name = 'tiling'
    pos_idx = 0
    for data in tqdm(test_stream):
        receptor, ligand, gt_rot, gt_txy = data

        monomer1 = receptor.squeeze().detach().cpu()
        monomer2 = ligand.squeeze().detach().cpu()
        shape_diff = abs(monomer1 - monomer2)
        sum_shape_diff = torch.sum(shape_diff)

        if sum_shape_diff == 0:
            print('homodimer found')
            title = 'homodimer_tiling_ex' + str(pos_idx)
        else:
            print('heterodimer found')
            title = 'heterodimer_tiling_ex' + str(pos_idx)

            # dimer = np.hstack((monomer1, monomer2, shape_diff))
            # plt.imshow(homodimer)
            # plt.colorbar()
            # plt.show()

            num_cycles = 3
            for i in range(num_cycles):
                evalFFT = TorchDockingFFT(padded_dim=200, num_angles=eval_angles)
                eval_model = SamplingModel(evalFFT, IP=True).to(device=0)

                print('receptor.shape', receptor.shape)
                print('ligand.shape', ligand.shape)
                # ligand = pad_to_match_shape(ligand, [receptor.shape[-2], receptor.shape[-1]])
                # receptor = pad_to_match_shape(receptor, [100, 100])

                # print('ligand.shape after padding', ligand.shape)

                # a = np.ones((500, 500))
                # shape = [1000, 1103]
                # if num_cycles > 0:
                #     initbox_size = receptor.shape[-1]
                #     pad_size = initbox_size // 2
                #
                #     if initbox_size % 2 == 0:
                #         ligand = F.pad(ligand, pad=([pad_size, pad_size, pad_size, pad_size]),
                #                                      mode='constant', value=0)
                #     else:
                #         ligand = F.pad(ligand, pad=([pad_size, pad_size + 1, pad_size, pad_size + 1]),
                #                             mode='constant', value=0)

                lowest_energy, pred_rot, pred_txy, fft_score = eval_model(receptor, ligand, plotting=plotting,
                                                                          training=False)

                # UtilityFunctions(stream_name).plot_predicted_pose(monomer1, monomer2, pred_rot, pred_txy, pred_rot.squeeze(),
                #                                      pred_txy.squeeze(), pos_idx, stream_name=stream_name)

                pred_rot = pred_rot.squeeze()
                pred_txy = pred_txy.squeeze()
                pair, ligand = UtilityFunctions(stream_name).plot_assembly(receptor.detach().cpu().numpy(), ligand.detach().cpu().numpy(),
                                                    pred_rot.squeeze().detach().cpu().numpy(),
                                                    (pred_txy[0].detach().cpu().numpy(), pred_txy[1].detach().cpu().numpy()), tiling=True)

                receptor = torch.tensor(pair).cuda().unsqueeze(0)
                ligand = torch.tensor(ligand).cuda().unsqueeze(0)
                receptor = receptor.clamp(min=0, max=1)
                # ligand = receptor
                monomer1 = receptor.squeeze()
                # monomer2 = monomer1
                if i == num_cycles-1:
                    plt.close()
                    plt.imshow(receptor.squeeze().detach().cpu())
                    plt.colorbar()
                    plt.title(title)
                    plt.savefig('Figs/Features_and_poses/'+title)

                    if show:
                        plt.show()

        pos_idx += 1
