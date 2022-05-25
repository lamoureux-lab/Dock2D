import random
import torch
from torch import optim

import sys
## path for cluster
sys.path.append('/home/sb1638/')

import numpy as np
from tqdm import tqdm
from Dock2D.Utility.TorchDataLoader import get_interaction_stream
from Dock2D.Utility.ValidationMetrics import APR
from Dock2D.Utility.PlotterFI import PlotterFI
from Dock2D.Utility.UtilityFunctions import UtilityFunctions
from Dock2D.Models.model_interaction import Interaction
from Dock2D.Models.model_docking import Docking


class BruteForceInteractionTrainer:
    ## run replicates from sbatch script args, if provided
    if len(sys.argv) > 1:
        replicate = str(sys.argv[1])
    else:
        replicate = 'single_rep'

    def __init__(self, docking_model, docking_optimizer, interaction_model, interaction_optimizer, experiment, training_case, path_pretrain,
                 debug=False, plotting=False):
        self.debug = debug
        self.plotting = plotting

        self.check_epoch = 1
        self.eval_freq = 1
        self.save_freq = 1
        ## set paths
        self.model_savepath = 'Log/saved_models/'
        self.logfile_savepath = 'Log/losses/FI_loss/'
        self.logtraindF_prefix = 'log_deltaF_TRAINset_epoch'
        self.logloss_prefix = 'log_loss_TRAINset_'
        self.logAPR_prefix = 'log_validAPR_'

        self.loss_log_header = 'Epoch\tLoss\n'
        self.loss_log_format = '%d\t%f\n'

        self.deltaf_log_header = 'F\tF_0\tLabel\n'
        self.deltaf_log_format = '%f\t%f\t%d\n'

        self.docking_model = docking_model
        self.interaction_model = interaction_model
        self.docking_optimizer = docking_optimizer
        self.interaction_optimizer = interaction_optimizer
        self.experiment = experiment
        self.training_case = training_case
        self.path_pretrain = path_pretrain
        self.set_docking_model_state()
        self.freeze_weights()

        self.wReg = 1e-5
        self.zero_value = torch.zeros(1).squeeze().cuda()

        self.UtilityFuncs = UtilityFunctions()

    def run_model(self, data, training=True):
        receptor, ligand, gt_interact = data

        receptor = receptor.to(device='cuda', dtype=torch.float)
        ligand = ligand.to(device='cuda', dtype=torch.float)
        gt_interact = gt_interact.to(device='cuda', dtype=torch.float).squeeze()

        if training:
            self.docking_model.train()
            self.interaction_model.train()

        ### run model and loss calculation
        ##### call model(s)
        fft_scores = self.docking_model(receptor, ligand, plotting=self.plotting)
        pred_interact, deltaF, F, F_0 = self.interaction_model(brute_force=True, fft_scores=fft_scores)

        ### check parameters and gradients
        ### if weights are frozen or updating
        if self.debug:
            self.UtilityFuncs.check_model_gradients(self.docking_model)
            self.UtilityFuncs.check_model_gradients(self.interaction_model)

        #### Loss functions
        BCEloss = torch.nn.BCELoss()
        l1_loss = torch.nn.L1Loss()
        L_reg = self.wReg * l1_loss(deltaF, self.zero_value)
        loss = BCEloss(pred_interact, gt_interact) + L_reg

        if self.debug:
            print('\n predicted', pred_interact.item(), '; ground truth', gt_interact.item())

        if training:
            self.docking_model.zero_grad()
            self.interaction_model.zero_grad()
            loss.backward(retain_graph=True)
            self.docking_optimizer.step()
            self.interaction_optimizer.step()
            return loss.item(), F.item(), F_0.item(), gt_interact.item()
        else:
            self.docking_model.eval()
            self.interaction_model.eval()
            with torch.no_grad():
                TP, FP, TN, FN = self.classify(pred_interact, gt_interact)
                return TP, FP, TN, FN, F.item(), F_0.item(), gt_interact.item()


    @staticmethod
    def classify(pred_interact, gt_interact):
        threshold = 0.5
        TP, FP, TN, FN = 0, 0, 0, 0
        p = pred_interact.item()
        a = gt_interact.item()
        if p >= threshold and a >= threshold:
            TP += 1
        elif p >= threshold and a < threshold:
            FP += 1
        elif p < threshold and a >= threshold:
            FN += 1
        elif p < threshold and a < threshold:
            TN += 1
        return TP, FP, TN, FN

    def train_model(self, train_epochs, train_stream, valid_stream, test_stream, resume_training=False,
                    resume_epoch=0):
        if self.plotting:
            self.eval_freq = 1

        ### Continue training on existing model?
        start_epoch = self.resume_training_or_not(resume_training, resume_epoch)

        num_epochs = start_epoch + train_epochs

        for epoch in range(start_epoch, num_epochs):

            docking_checkpoint_dict = {
                'epoch': epoch,
                'state_dict': self.docking_model.state_dict(),
                'optimizer': self.docking_optimizer.state_dict(),
            }
            interaction_checkpoint_dict = {
                'epoch': epoch,
                'state_dict': self.interaction_model.state_dict(),
                'optimizer': self.interaction_optimizer.state_dict(),
            }

            if train_stream:
                self.run_epoch(train_stream, epoch, training=True)

                # plot loss and free energy distributions
                PlotterFI(self.experiment).plot_loss(show=False)
                PlotterFI(self.experiment).plot_deltaF_distribution(plot_epoch=epoch, show=False, xlim=None, binwidth=1)

                # F_0_scheduler.step()
                print('last F_0 learning rate', F_0_scheduler.get_last_lr())
                #### saving model while training
                if epoch % self.save_freq == 0:
                    docking_savepath = self.model_savepath + 'docking_' + self.experiment + str(epoch) + '.th'
                    self.save_checkpoint(docking_checkpoint_dict, docking_savepath, self.docking_model)
                    print('saving docking model ' + docking_savepath)

                    interaction_savepath = self.model_savepath + self.experiment + str(epoch) + '.th'
                    self.save_checkpoint(interaction_checkpoint_dict, interaction_savepath, self.interaction_model)
                    print('saving interaction model ' + interaction_savepath)

            ### evaluate on training and valid set
            ### training set to False downstream in calcAPR() run_model()

            if epoch % self.eval_freq == 0:
                if valid_stream:
                    stream_name = 'VALIDset'
                    deltaF_logfile = self.logfile_savepath + stream_name + self.logtraindF_prefix + str(
                        epoch) + self.experiment + '.txt'
                    with open(deltaF_logfile, 'w') as fout:
                        fout.write(self.deltaf_log_header)
                    self.checkAPR(epoch, valid_stream, stream_name=stream_name, deltaF_logfile=deltaF_logfile, experiment=self.experiment)
                if test_stream:
                    stream_name = 'TESTset'
                    deltaF_logfile = self.logfile_savepath + stream_name + self.logtraindF_prefix + str(
                        epoch) + self.experiment + '.txt'
                    with open(deltaF_logfile, 'w') as fout:
                        fout.write(self.deltaf_log_header)
                    self.checkAPR(epoch, test_stream, stream_name=stream_name, deltaF_logfile=deltaF_logfile, experiment=self.experiment)


    def run_epoch(self, data_stream, epoch, training=False):
        stream_loss = []
        deltaF_logfile = self.logfile_savepath + self.logtraindF_prefix + str(epoch) + self.experiment + '.txt'
        with open(deltaF_logfile, 'w') as fout:
            fout.write(self.deltaf_log_header)
        for data in tqdm(data_stream):
            train_output = [self.run_model(data, training=training)]
            stream_loss.append(train_output)
            with open(deltaF_logfile, 'a') as fout:
                fout.write(self.deltaf_log_format % (train_output[0][1], train_output[0][2], train_output[0][3]))

        loss_logfile = self.logfile_savepath + self.logloss_prefix + self.experiment + '.txt'
        avg_loss = np.average(stream_loss, axis=0)[0, :]
        print('\nEpoch', epoch, 'Train Loss: loss', avg_loss[0])
        with open(loss_logfile, 'a') as fout:
            fout.write(self.loss_log_format % (epoch, avg_loss[0]))

    def checkAPR(self, check_epoch, datastream, stream_name=None, deltaF_logfile=None, experiment=None):
        log_APRheader = 'Accuracy\tPrecision\tRecall\tF1score\tMCC\n'
        log_APRformat = '%f\t%f\t%f\t%f\t%f\n'
        print('Evaluating ', stream_name)
        Accuracy, Precision, Recall, F1score, MCC = APR().calc_APR(datastream, self.run_model, check_epoch, deltaF_logfile, experiment, stream_name)
        with open(self.logfile_savepath + self.logAPR_prefix + self.experiment + '.txt', 'a') as fout:
            fout.write('Epoch '+str(check_epoch)+'\n')
            fout.write(log_APRheader)
            fout.write(log_APRformat % (Accuracy, Precision, Recall, F1score, MCC))
        fout.close()

    def freeze_weights(self):
        if not self.param_to_freeze:
            print('\nAll docking model params unfrozen\n')
            return
        for name, param in self.docking_model.named_parameters():
            if self.param_to_freeze == 'all':
                print('Freeze ALL Weights', name)
                param.requires_grad = False
            elif self.param_to_freeze in name:
                print('Freeze Weights', name)
                param.requires_grad = False
            else:
                print('Unfreeze docking model weights', name)
                param.requires_grad = True

    def resume_training_or_not(self, resume_training, resume_epoch):
        if resume_training:
            print('Loading docking model at', str(resume_epoch))
            docking_ckp_path = self.model_savepath+'docking_' + self.experiment + str(resume_epoch) + '.th'
            self.docking_model, self.docking_optimizer, _ = self.load_checkpoint(
                docking_ckp_path, self.docking_model, self.docking_optimizer)
            print('Loading interaction model at', str(resume_epoch))
            interaction_ckp_path = self.model_savepath + self.experiment + str(resume_epoch) + '.th'
            self.interaction_model, self.interaction_optimizer, start_epoch = self.load_checkpoint(
                interaction_ckp_path, self.interaction_model, self.interaction_optimizer)

            start_epoch += 1

            print('\ndocking model:\n', self.docking_model)
            ## print model and params being loaded
            self.UtilityFuncs.check_model_gradients(self.docking_model)
            print('\ninteraction model:\n', self.interaction_model)
            ## print model and params being loaded
            self.UtilityFuncs.check_model_gradients(self.interaction_model)

            print('\nLOADING MODEL AT EPOCH', start_epoch, '\n')
        else:
            start_epoch = 1
            ### Loss log files
            with open(self.logfile_savepath + self.logloss_prefix + self.experiment + '.txt', 'w') as fout:
                fout.write(self.loss_log_header)
            with open(self.logfile_savepath + self.logtraindF_prefix + str(start_epoch) + self.experiment + '.txt', 'w') as fout:
                fout.write(self.deltaf_log_header)

        return start_epoch

    @staticmethod
    def save_checkpoint(state, filename, model):
        model.eval()
        torch.save(state, filename)

    @staticmethod
    def load_checkpoint(checkpoint_fpath, model, optimizer):
        model.eval()
        checkpoint = torch.load(checkpoint_fpath)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        return model, optimizer, checkpoint['epoch']

    def set_docking_model_state(self):
        # CaseA: train with docking model frozen
        if self.training_case == 'A':
            print('Training expA')
            self.param_to_freeze = 'all'
            self.docking_model.load_state_dict(torch.load(path_pretrain)['state_dict'])
        # CaseB: train with docking model unfrozen
        if self.training_case == 'B':
            print('Training expB')
            lr_docking = 10 ** -5
            print('Docking learning rate changed to', lr_docking)
            # self.experiment = 'case' + self.training_case + '_lr5change_' + self.experiment
            self.docking_model = Docking().to(device=0)
            self.docking_optimizer = optim.Adam(self.docking_model.parameters(), lr=lr_docking)
            self.param_to_freeze = None
            self.docking_model.load_state_dict(torch.load(path_pretrain)['state_dict'])
        # CaseC: train with docking model SE2 CNN frozen
        if self.training_case == 'C':
            print('Training expC')
            self.param_to_freeze = 'netSE2'  # leave "a" scoring coefficients unfrozen
            self.docking_model.load_state_dict(torch.load(path_pretrain)['state_dict'])
        # Case scratch: train everything from scratch
        if self.training_case == 'scratch':
            print('Training from scratch')
            self.param_to_freeze = None
            # self.experiment = self.training_case + '_' + self.experiment

    def run_trainer(self, train_epochs, train_stream=None, valid_stream=None, test_stream=None, resume_epoch=0, resume_training=False):
        self.train_model(train_epochs, train_stream, valid_stream, test_stream,
                                                   resume_training=resume_training, resume_epoch=resume_epoch)


if __name__ == '__main__':
    #################################################################################
    ##Datasets
    trainset = '../../Datasets/interaction_train_400pool.pkl'
    validset = '../../Datasets/interaction_valid_400pool.pkl'
    # ### testing set
    testset = '../../Datasets/interaction_test_400pool.pkl'
    #########################
    #### initialization torch settings
    random_seed = 42
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(0)
    # CUDA_LAUNCH_BLOCKING = 1
    # torch.autograd.set_detect_anomaly(True)
    #########################
    train_epochs = 20
    lr_interaction = 10**-1
    lr_docking = 10**-4
    gamma = 0.95
    interaction_model = Interaction().to(device=0)
    interaction_optimizer = optim.Adam(interaction_model.parameters(), lr=lr_interaction)
    F_0_scheduler = optim.lr_scheduler.ExponentialLR(interaction_optimizer, gamma=gamma)

    docking_model = Docking().to(device=0)
    docking_optimizer = optim.Adam(docking_model.parameters(), lr=lr_docking)

    ## number_of_pairs provides max_size of interactions: max_size = int(number_of_pairs + (number_of_pairs**2 - number_of_pairs)/2)
    number_of_pairs = 100

    train_stream = get_interaction_stream(trainset, number_of_pairs=number_of_pairs)
    valid_stream = get_interaction_stream(validset, number_of_pairs=number_of_pairs)
    test_stream = get_interaction_stream(testset, number_of_pairs=number_of_pairs)
    ######################
    # experiment = 'BF_FI_NEWDATA_CHECK_400pool_1000ex50ep'
    # experiment = 'BF_FI_NEWDATA_CHECK_400pool_2000ex50ep'
    # experiment = 'BF_FI_NEWDATA_CHECK_400pool_5000ex30ep'
    # experiment = 'BF_FI_NEWDATA_CHECK_400pool_10000ex30ep'
    # experiment = 'BF_FI_NEWDATA_CHECK_400pool_20000ex30ep'
    # experiment = 'BF_FI_NEWDATA_CHECK_400pool_20000ex30ep'
    # experiment = 'BF_FI_400pool_100pairs_100ep_filr1e-1_noFreg'
    experiment = 'BF_FI_400pool_100pairs_20ep_check_model_interaction'

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
    ### Train model from beginning
    # BruteForceInteractionTrainer(docking_model, docking_optimizer, interaction_model, interaction_optimizer, experiment, training_case, path_pretrain
    #                              ).run_trainer(train_epochs, train_stream=train_stream, valid_stream=None, test_stream=None)

    # ## Resume training model at chosen epoch
    # BruteForceInteractionTrainer(docking_model, docking_optimizer, interaction_model, interaction_optimizer, experiment, training_case, path_pretrain
    #                              ).run_trainer(resume_training=True, resume_epoch=60, train_epochs=40, train_stream=train_stream, valid_stream=None, test_stream=None)
    # #

    ### Validate model at chosen epoch
    BruteForceInteractionTrainer(docking_model, docking_optimizer, interaction_model, interaction_optimizer, experiment, training_case, path_pretrain
                                 ).run_trainer(train_epochs=1, train_stream=None, valid_stream=valid_stream, test_stream=test_stream,
                                               resume_training=True, resume_epoch=train_epochs)

    ### Plot loss and free energy distributions with learned F_0 decision threshold
    PlotterFI(experiment).plot_loss(show=True)
    PlotterFI(experiment).plot_deltaF_distribution(plot_epoch=train_epochs, show=True)
