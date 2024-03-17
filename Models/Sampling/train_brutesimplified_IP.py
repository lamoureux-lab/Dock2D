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
    trainset = '../../Datasets/docking_train_50pool.pkl'
    validset = '../../Datasets/docking_valid_50pool.pkl'
    ### testing set
    testset = '../../Datasets/docking_test_50pool.pkl'
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
    # max_size = 2
    train_stream = get_docking_stream(trainset, max_size=max_size)
    valid_stream = get_docking_stream(validset, max_size=None, shuffle=False)
    test_stream = get_docking_stream(testset, max_size=None, shuffle=False)
    sample_buffer_length = max(len(train_stream), len(valid_stream), len(test_stream))
    ######################
    # experiment = 'BS_IP_finaldataset'
    # experiment = 'BS_IP_finaldataset_100pairs_100ep'
    # experiment = 'BS_IP_finaldataset_1000pairs_100ep'
    # experiment = 'BS_IP_finaldataset_10pairs_100ep'

    # experiment = 'BS_IP_finaldataset_10pairs_100ep_rep1'
    experiment = 'BS_IP_finaldataset_overfitcheck_1pair_100ep'

    ######################
    train_epochs = 100
    lr = 10 ** -3
    ########################
    model_name = "BS IP"
    debug = False
    plotting = False
    show = True
    ########################
    padded_dim = 100
    num_angles = 1
    sampledFFT = TorchDockingFFT(padded_dim=padded_dim, num_angles=num_angles, model_name=model_name)
    model = SamplingModel(sampledFFT, IP=True).to(device=0)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    Trainer = TrainerIP(sampledFFT, model, optimizer, experiment)
    ######################
    ### Train model from beginning
    Trainer.run_trainer(train_epochs, train_stream=train_stream)

    ### Resume training model at chosen epoch
    # Trainer.run_trainer(
    #     train_epochs=1, train_stream=train_stream, valid_stream=None, test_stream=None,
    #     resume_training=True, resume_epoch=train_epochs)

    ### Resume training for validation sets
    # Trainer.run_trainer(
    #     train_epochs=10, train_stream=None, valid_stream=valid_stream, #test_stream=valid_stream,
    #     resume_training=True, resume_epoch=train_epochs)

    ## Brute force evaluation and plotting
    # plotting = False
    # start = train_epochs-1
    # stop = train_epochs

    # eval_angles = 360
    # evalFFT = TorchDockingFFT(padded_dim=padded_dim, num_angles=eval_angles, model_name=model_name)
    # eval_model = SamplingModel(evalFFT, IP=True).to(device=0)
    # EvalTrainer = TrainerIP(evalFFT, eval_model, optimizer, experiment,
    #                         BF_eval=True, plotting=plotting, sample_buffer_length=sample_buffer_length)
    # # # for epoch in range(start, stop):
    # # #     if stop-1 == epoch:
    # # #         plotting = True
    # # #     EvalTrainer.run_trainer(train_epochs=1, train_stream=None, valid_stream=valid_stream, test_stream=test_stream,
    # # #                             resume_training=True, resume_epoch=epoch)
    # #
    # EvalTrainer.run_trainer(train_epochs=1, train_stream=None, valid_stream=valid_stream, test_stream=test_stream,
    #                         resume_training=True, resume_epoch=train_epochs)

    ## Plot loss and RMSDs from current experiment
    PlotterIP(experiment).plot_loss(ylim=None, show=show)
    from matplotlib import rcParams
    rcParams.update({'font.size': 20})
    PlotterIP(experiment).plot_rmsd_distribution(plot_epoch=train_epochs, show=show)
