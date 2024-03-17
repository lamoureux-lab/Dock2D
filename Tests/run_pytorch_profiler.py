import torch.autograd.profiler as profiler

from Dock2D.Models.TrainerIP import *
import random
from Dock2D.Utility.TorchDataLoader import get_docking_stream
from torch import optim
from Dock2D.Models.model_sampling import SamplingModel
from Dock2D.Utility.TorchDockingFFT import TorchDockingFFT

if __name__ == '__main__':
    #################################################################################
    # Datasets
    trainset = '../Datasets/docking_train_50pool.pkl'
    validset = '../Datasets/docking_valid_50pool.pkl'
    ### testing set
    testset = '../Datasets/docking_test_50pool.pkl'
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
    max_size = 10
    # max_size = 1
    train_stream = get_docking_stream(trainset, max_size=max_size)
    # valid_stream = get_docking_stream(validset,  max_size=None, shuffle=False)
    # test_stream = get_docking_stream(testset, max_size=None, shuffle=False)
    ######################
    # experiment = 'BF_IP_profiler_check'
    experiment = 'BS_IP_profiler_check'

    ######################
    plotting = False

    train_epochs = 100
    # learning_rate = 10 ** -3
    learning_rate = 10 ** -1

    padded_dim = 100
    # num_angles = 360
    num_angles = 1
    BFdockingFFT = TorchDockingFFT(padded_dim=padded_dim, num_angles=num_angles)
    model = SamplingModel(BFdockingFFT, IP=True).to(device=0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    Trainer = TrainerIP(BFdockingFFT, model, optimizer, experiment, BF_eval=True, plotting=plotting)

    ### warm-up
    Trainer.run_trainer(
        train_epochs=train_epochs, train_stream=train_stream, valid_stream=None, test_stream=None)

    # #### run profiler
    # with profiler.profile(with_stack=True, profile_memory=True) as prof:
    #     Trainer.run_trainer(
    #         train_epochs=train_epochs, train_stream=train_stream, valid_stream=None, test_stream=None)
    #
    # print(prof.key_averages().table(sort_by='self_cpu_time_total'))
    # print(prof.total_average())
