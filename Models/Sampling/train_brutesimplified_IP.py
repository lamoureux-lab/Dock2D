from Dock2D.Models.TrainerIP import *
import random
from Dock2D.Utility.TorchDataLoader import get_docking_stream
from torch import optim
from Dock2D.Utility.PlotterIP import PlotterIP
from Dock2D.Models.model_sampling import SamplingModel

if __name__ == '__main__':
    #################################################################################
    # Datasets
    trainset = '../../Datasets/docking_train_400pool.pkl'
    validset = '../../Datasets/docking_valid_400pool.pkl'
    ### testing set
    testset = '../../Datasets/docking_test_400pool.pkl'
    #########################
    #### initialization torch settings
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
    experiment = 'BS_check_code_consolidated_10ep'

    ######################
    train_epochs = 10
    lr = 10 ** -2
    plotting = False
    show = True
    #####################
    sampledFFT = TorchDockingFFT(num_angles=1, angle=None)
    model = SamplingModel(sampledFFT, num_angles=1, IP=True).to(device=0)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    ######################
    ### Train model from beginning
    TrainerIP(sampledFFT, model, optimizer, experiment).run_trainer(train_epochs, train_stream=train_stream)

    ### Resume training model at chosen epoch
    # BruteSimplifiedDockingTrainer(dockingFFT, model, optimizer, experiment, plotting=True, debug=debug).run_trainer(
    #     train_epochs=1, train_stream=train_stream, valid_stream=None, test_stream=None,
    #     resume_training=True, resume_epoch=train_epochs)

    ### Resume training for validation sets
    # BruteSimplifiedDockingTrainer(dockingFFT, model, optimizer, experiment, plotting=plotting, debug=debug).run_trainer(
    #     train_epochs=1, train_stream=None, valid_stream=valid_stream, #test_stream=valid_stream,
    #     resume_training=True, resume_epoch=train_epochs)

    ## Brute force evaluation and plotting
    start = train_epochs-1
    stop = train_epochs
    eval_angles = 360
    eval_model = SamplingModel(sampledFFT, num_angles=eval_angles, IP=True).to(device=0)
    for epoch in range(start, stop):
        if stop-1 == epoch:
            plotting = False
            TrainerIP(sampledFFT, eval_model, optimizer, experiment,
                                    BF_eval=True, plotting=plotting, sample_buffer_length=sample_buffer_length).run_trainer(
                                    train_epochs=1, train_stream=None, valid_stream=valid_stream, test_stream=test_stream,
                                    resume_training=True, resume_epoch=epoch)

    ## Plot loss and RMSDs from current experiment
    PlotterIP(experiment).plot_loss(ylim=None)
    PlotterIP(experiment).plot_rmsd_distribution(plot_epoch=train_epochs, show=show)
