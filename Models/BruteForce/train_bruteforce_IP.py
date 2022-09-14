from Dock2D.Models.TrainerIP import *
import random
from Dock2D.Utility.TorchDataLoader import get_docking_stream
from torch import optim
from Dock2D.Utility.PlotterIP import PlotterIP
from Dock2D.Models.model_sampling import SamplingModel
from Dock2D.Utility.TorchDockingFFT import TorchDockingFFT

if __name__ == '__main__':
    #################################################################################
    # Datasets
    trainset = '../../Datasets/docking_train_400pool.pkl'
    validset = '../../Datasets/docking_valid_400pool.pkl'
    ### testing set
    testset = '../../Datasets/docking_test_400pool.pkl'
    #########################
    #### initialization of random seeds
    random_seed = 42
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    # torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(0)
    # torch.autograd.set_detect_anomaly(True)
    ######################
    max_size = 10
    train_stream = get_docking_stream(trainset, max_size=max_size)
    valid_stream = get_docking_stream(validset,  max_size=None, shuffle=False)
    test_stream = get_docking_stream(testset, max_size=None, shuffle=False)
    ######################
    # experiment = 'docking_scratch_BF_FI_finaldataset_100pairs_500ep'
    # experiment = 'docking_scratch_BF_FI_finaldataset_100pairs_1000ep'
    # experiment = 'docking_scratch_MC_FI_finaldataset_100pairs_1000ep_2sample_100steps'

    # experiment = 'BF_IP_finaldataset_100pairs_100ep'
    # experiment = 'BF_IP_finaldataset_1000pairs_100ep'

    experiment = 'BF_IP_finaldataset_10pairs_100ep'

    ######################
    plotting = False

    train_epochs = 100
    learning_rate = 10 ** -3

    padded_dim = 100
    num_angles = 360
    BFdockingFFT = TorchDockingFFT(padded_dim=padded_dim, num_angles=num_angles)
    model = SamplingModel(BFdockingFFT, IP=True).to(device=0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    Trainer = TrainerIP(BFdockingFFT, model, optimizer, experiment, BF_eval=True, plotting=plotting)
    ######################
    ### Train model from beginning, evaluate if valid_stream and/or test_stream passed in
    Trainer.run_trainer(train_epochs=train_epochs, train_stream=train_stream, valid_stream=None, test_stream=None)

    ### Resume training model at chosen epoch
    # Trainer.run_trainer(train_stream=None, valid_stream=valid_stream, test_stream=test_stream,
    #                     resume_training=True, resume_epoch=13, train_epochs=17)

    # ### Evaluate model on chosen dataset only and plot at chosen epoch and dataset frequency
    Trainer.run_trainer(train_stream=None, valid_stream=valid_stream, test_stream=test_stream,
                        resume_training=True, resume_epoch=train_epochs, train_epochs=1)

    ## Plot loss and RMSDs from current experiment
    PlotterIP(experiment).plot_loss(show=True)
    PlotterIP(experiment).plot_rmsd_distribution(plot_epoch=train_epochs+1, show=True)
