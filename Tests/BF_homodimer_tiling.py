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
    max_size = 1000
    train_stream = get_docking_stream(trainset, max_size=max_size)
    valid_stream = get_docking_stream(validset, max_size=max_size)
    test_stream = get_docking_stream(testset, max_size=max_size)
    sample_buffer_length = max(len(train_stream), len(valid_stream), len(test_stream))
    ######################
    # experiment = 'BS_check_code_consolidated_10ep'
    # experiment = 'BS_check_singlecrossterm_10ep'

    # experiment = 'BS_lr-2_30ep'
    # experiment = 'BS_lr-3_30ep'
    # experiment = 'BS_lr-4_30ep'

    # experiment = 'BS_lr-2_10ep_check_NoNormPool'
    # experiment = 'BS_lr-2_10ep_latest_400poolcheck'
    experiment = 'BS_lr-2_30ep_latest_400poolcheck'
    #
    # experiment = 'BF_eval_homodimer_tiling'

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
    plotting = True
    eval_angles = 360
    evalFFT = TorchDockingFFT(padded_dim=padded_dim, num_angles=eval_angles)
    eval_model = SamplingModel(evalFFT, IP=True).to(device=0)
    EvalTrainer = TrainerIP(evalFFT, eval_model, optimizer, experiment,
                            BF_eval=True, plotting=plotting, tiling=True)

    EvalTrainer.run_trainer(train_epochs=1, train_stream=None, valid_stream=valid_stream,
                            test_stream=test_stream,
                            resume_training=True, resume_epoch=train_epochs)
