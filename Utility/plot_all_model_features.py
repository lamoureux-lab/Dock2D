from Dock2D.Models.TrainerFI import *
import random
from Dock2D.Utility.TorchDataLoader import get_interaction_stream
from torch import optim
from Dock2D.Utility.PlotterFI import PlotterFI
from Dock2D.Models.model_interaction import Interaction
from Dock2D.Models.model_sampling import SamplingModel
from Dock2D.Utility.TorchDockingFFT import TorchDockingFFT


if __name__ == '__main__':
    #################################################################################
    ##Datasets
    trainset = '../Datasets/interaction_train_400pool.pkl'
    validset = '../Datasets/interaction_valid_400pool.pkl'
    # ### testing set
    testset = '../Datasets/interaction_test_400pool.pkl'
    #########################
    #### initialization of random seeds
    random_seed = 42
    randomstate = np.random.RandomState(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    # torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(0)
    # torch.autograd.set_detect_anomaly(True)
    #########################
    ## number_of_pairs provides max_size of interactions: max_size = number_of_pairs*(number_of_pairs + 1)/2
    number_of_pairs = 100
    train_stream = get_interaction_stream(trainset, number_of_pairs=number_of_pairs, randomstate=randomstate)
    valid_stream = get_interaction_stream(validset, number_of_pairs=None)
    test_stream = get_interaction_stream(testset, number_of_pairs=None)
    ######################
    experiment = 'BF_FI_finaldataset_100pairs_1000ep'
    # experiment = 'BF_FI_finaldataset_100pairs_expC_BFIP_1000ex100ep'
    ##################### Load and freeze/unfreeze params (training, no eval)
    ### path to pretrained docking model
    path_pretrain = 'Log/saved_models/IP_saved/BF_IP_finaldataset_1000pairs_100ep100.th'
    # training_case = 'A' # CaseA: train with docking model frozen
    # training_case = 'B' # CaseB: train with docking model unfrozen
    # training_case = 'C' # CaseC: train with docking model SE2 CNN frozen and scoring ("a") coeffs unfrozen
    training_case = 'scratch' # Case scratch: train everything from scratch
    experiment = training_case + '_' + experiment
    #####################
    train_epochs = 100
    lr_interaction = 10 ** -1
    lr_docking = 10 ** -4
    sample_steps = 10
    sample_buffer_length = max(len(train_stream), len(valid_stream), len(test_stream))

    debug = False
    plotting = True
    show = False

    interaction_model = Interaction().to(device=0)
    interaction_optimizer = optim.Adam(interaction_model.parameters(), lr=lr_interaction)

    padded_dim = 100
    num_angles = 360
    dockingFFT = TorchDockingFFT(padded_dim=padded_dim, num_angles=num_angles)
    docking_model = SamplingModel(dockingFFT, sample_steps=sample_steps, FI_BF=True).to(device=0)
    docking_optimizer = optim.Adam(docking_model.parameters(), lr=lr_docking)
    # Trainer = TrainerFI(docking_model, docking_optimizer, interaction_model, interaction_optimizer, experiment,
    #           training_case, path_pretrain,
    #           FI_MC=False,
    #           plotting=plotting,)

    Trainer = TrainerFI(docking_model, docking_optimizer, interaction_model, interaction_optimizer, experiment,
              training_case, path_pretrain,
              FI_MC=False,
              plotting=plotting,)
    ######################
    # Evaluate model at chosen epoch (Brute force evaluation)
    eval_angles = 360
    evalFFT = TorchDockingFFT(padded_dim=padded_dim, num_angles=eval_angles, model_name=model_name)
    eval_model = SamplingModel(evalFFT, FI_MC=True).to(device=0)
    TrainerFI(eval_model, docking_optimizer, interaction_model, interaction_optimizer, experiment, FI_MC=True,
              plotting=plotting,
                                  ).run_trainer(resume_training=True, resume_epoch=1000, train_epochs=1,
                                                train_stream=None, valid_stream=valid_stream, test_stream=test_stream)
