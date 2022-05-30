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
    # Datasets
    trainset = '../../Datasets/interaction_train_400pool.pkl'
    validset = '../../Datasets/interaction_valid_400pool.pkl'
    ### testing set
    testset = '../../Datasets/interaction_test_400pool.pkl'
    #########################
    #### initialization random settings
    random_seed = 42
    randomstate = np.random.RandomState(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(0)
    # torch.autograd.set_detect_anomaly(True)
    #########################
    ## number_of_pairs provides max_size of interactions: max_size = (number_of_pairs**2 + number_of_pairs)/2
    number_of_pairs = 50
    train_stream = get_interaction_stream(trainset, number_of_pairs=number_of_pairs, randomstate=randomstate)
    valid_stream = get_interaction_stream(validset, number_of_pairs=100)
    test_stream = get_interaction_stream(testset, number_of_pairs=100)
    ######################
    experiment = 'MC_FI_check_consolodated'
    ##################### Load and freeze/unfreeze params (training, no eval)
    ### path to pretrained docking model
    # path_pretrain = 'Log/RECODE_CHECK_BFDOCKING_30epochsend.th'
    path_pretrain = 'Log/FINAL_CHECK_DOCKING30.th'
    # training_case = 'A' # CaseA: train with docking model frozen
    # training_case = 'B' # CaseB: train with docking model unfrozen
    # training_case = 'C' # CaseC: train with docking model SE2 CNN frozen and scoring ("a") coeffs unfrozen
    training_case = 'scratch' # Case scratch: train everything from scratch
    experiment = training_case + '_' + experiment
    #####################
    train_epochs = 20
    lr_interaction = 10 ** -1
    lr_docking = 10 ** -4
    sample_steps = 10
    sample_buffer_length = max(len(train_stream), len(valid_stream), len(test_stream))

    debug = False
    plotting = False
    show = False

    interaction_model = Interaction().to(device=0)
    interaction_optimizer = optim.Adam(interaction_model.parameters(), lr=lr_interaction)

    num_angles = 1
    dockingFFT = TorchDockingFFT(num_angles=num_angles, angle=None)
    docking_model = SamplingModel(dockingFFT, num_angles=num_angles, sample_steps=sample_steps, FI_MC=True).to(device=0)
    docking_optimizer = optim.Adam(docking_model.parameters(), lr=lr_docking)
    Trainer = TrainerFI(docking_model, docking_optimizer, interaction_model, interaction_optimizer, experiment,
              training_case, path_pretrain, sample_buffer_length=sample_buffer_length,
              FI_MC=True)
    ######################
    ### Train model from beginning
    # Trainer.run_trainer(train_epochs, train_stream=train_stream, valid_stream=None, test_stream=None)

    ### resume training model
    Trainer.run_trainer(resume_training=True, resume_epoch=5, train_epochs=15,
                                               train_stream=train_stream, valid_stream=None, test_stream=None)

    ### Evaluate model at chosen epoch (Brute force or monte carlo evaluation)
    eval_model = SamplingModel(dockingFFT, num_angles=360, FI_MC=True).to(device=0)
    # # eval_model = SamplingModel(dockingFFT, num_angles=1, sample_steps=sample_steps, FI_MC=True, debug=debug).to(device=0) ## eval with monte carlo
    TrainerFI(eval_model, docking_optimizer, interaction_model, interaction_optimizer, experiment, debug=False
                                  ).run_trainer(resume_training=True, resume_epoch=train_epochs, train_epochs=1,
                                                train_stream=None, valid_stream=valid_stream, test_stream=test_stream)

    ### Plot loss and free energy distributions with learned F_0 decision threshold
    PlotterFI(experiment).plot_loss()
    PlotterFI(experiment).plot_deltaF_distribution(plot_epoch=train_epochs, show=True)