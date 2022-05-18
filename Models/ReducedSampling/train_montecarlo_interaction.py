import torch
import random
from torch import optim
import sys
sys.path.append('/home/sb1638/') ## path for cluster

import numpy as np
from tqdm import tqdm
from Dock2D.Utility.TorchDataLoader import get_interaction_stream
from Dock2D.Utility.TorchDockingFFT import TorchDockingFFT
from Dock2D.Utility.validation_metrics import APR
from Dock2D.Utility.PlotterFI import PlotterFI
from Dock2D.Utility.SampleBuffer import SampleBuffer
from Dock2D.Utility.UtilityFunctions import UtilityFunctions

from Dock2D.Models.BruteForce.train_bruteforce_interaction import Interaction
from Dock2D.Models.ReducedSampling.model_sampling import SamplingModel


class EnergyBasedInteractionTrainer:

    def __init__(self, docking_model, docking_optimizer, interaction_model, interaction_optimizer, experiment, sigma_alpha=3.0,
                 debug=False, plotting=False):
        # print("RUNNING INIT")
        self.debug = debug
        self.plotting = plotting
        self.plot_freq = 1000
        self.check_epoch = 1
        self.eval_freq = 1
        self.save_freq = 1

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

        sample_buffer_length = max(len(train_stream), len(valid_stream), len(test_stream))
        self.alpha_buffer = SampleBuffer(num_examples=sample_buffer_length)
        self.free_energy_buffer = SampleBuffer(num_examples=sample_buffer_length)

        # self.sigma_alpha = 3.0
        self.sig_alpha = sigma_alpha
        self.wReg = 1e-5
        self.zero_value = torch.zeros(1).squeeze().cuda()
        # self.sigma_scheduler_initial = sigma_scheduler.get_last_lr()[0]

        self.UtilityFuncs = UtilityFunctions()
        self.BF_eval = False

    def run_model(self, data, pos_idx=torch.tensor([0]), training=True, stream_name='trainset', epoch=0):

        receptor, ligand, gt_interact = data

        receptor = receptor.to(device='cuda', dtype=torch.float)
        ligand = ligand.to(device='cuda', dtype=torch.float)
        gt_interact = gt_interact.to(device='cuda', dtype=torch.float).squeeze()

        if training:
            self.docking_model.train()
            self.interaction_model.train()
        else:
            self.docking_model.eval()
            self.interaction_model.eval()
            self.BF_eval = True

        ### run model and loss calculation
        ##### push/pull samples of alpha and free energies to sample buffer
        plot_count = int(pos_idx)
        alpha = self.alpha_buffer.get(pos_idx, samples_per_example=1)
        free_energies_visited = self.free_energy_buffer.get_free_energies(pos_idx)
        # print('BUFFER GET: free_energies_visited', free_energies_visited)
        # print('BUFFER GET: free_energies_visited.shape', free_energies_visited.shape)

        free_energies_visited, pred_rot, pred_txy, fft_score_stack, acceptance_rate = self.docking_model(alpha, receptor, ligand,
                                        free_energies_visited=free_energies_visited, sig_alpha=self.sig_alpha,
                                        plot_count=plot_count, stream_name=stream_name, plotting=self.plotting,
                                        training=training)

        # print('BUFFER PUSH: free_energies_visited', free_energies_visited)
        # print('BUFFER PUSH: free_energies_visited.shape', free_energies_visited.shape)

        self.alpha_buffer.push(pred_rot, pos_idx)
        self.free_energy_buffer.push_free_energies(free_energies_visited, pos_idx)
        pred_interact, deltaF, F, F_0 = self.interaction_model(brute_force=self.BF_eval, fft_scores=fft_score_stack, free_energies=free_energies_visited, debug=False)

        if plot_count % self.plot_freq == 0 and training:
            UtilityFunctions(self.experiment).plot_MCsampled_energysurface(free_energies_visited, acceptance_rate,
                                                stream_name, plot_count=plot_count, epoch=epoch)

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
        else:
            self.docking_model.eval()
            self.interaction_model.eval()
            with torch.no_grad():
                return self.classify(pred_interact, gt_interact)

        return loss.item(), F.item(), F_0.item(), gt_interact.item()

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
                print('sigma_alpha = ', self.sig_alpha)

                self.run_epoch(train_stream, epoch, training=True)
                PlotterFI(self.experiment).plot_loss(show=False)
                PlotterFI(self.experiment).plot_deltaF_distribution(plot_epoch=epoch, show=False, xlim=None, binwidth=1)

                # F_0_scheduler.step()
                # print('last learning rate', F_0_scheduler.get_last_lr())
                # self.sigma_alpha = self.sigma_alpha * F_0_scheduler.get_last_lr()[0]
                # print('sigma alpha stepped', self.sigma_alpha)

                # sigma_scheduler.step()
                # self.sigma_alpha = self.sigma_alpha * (sigma_scheduler.get_last_lr()[0]/self.sigma_scheduler_initial)

            ### evaluate on training and valid set
            ### training set to False downstream in calcAPR() run_model()

            if epoch % self.eval_freq == 0:
                if valid_stream:
                    self.checkAPR(epoch, valid_stream, 'VALIDset')
                if test_stream:
                    self.checkAPR(epoch, test_stream, 'TESTset')

            #### saving model while training
            if epoch % self.save_freq == 0:
                docking_savepath =  self.model_savepath + 'docking_' + self.experiment + str(epoch) + '.th'
                self.save_checkpoint(docking_checkpoint_dict, docking_savepath, self.docking_model)
                print('saving docking model ' + docking_savepath)

                interaction_savepath = self.model_savepath + self.experiment + str(epoch) + '.th'
                self.save_checkpoint(interaction_checkpoint_dict, interaction_savepath, self.interaction_model)
                print('saving interaction model ' + interaction_savepath)

    def run_epoch(self, data_stream, epoch, training=False):
        stream_loss = []
        pos_idx = 0
        deltaF_logfile = self.logfile_savepath + self.logtraindF_prefix + str(epoch) + self.experiment + '.txt'
        with open(deltaF_logfile, 'w') as fout:
            fout.write(self.deltaf_log_header)
        for data in tqdm(data_stream):
            train_output = [self.run_model(data, pos_idx=torch.tensor([pos_idx]), training=training, epoch=epoch)]
            stream_loss.append(train_output)
            pos_idx += 1
            with open(deltaF_logfile, 'a') as fout:
                fout.write(self.deltaf_log_format % (train_output[0][1], train_output[0][2], train_output[0][3]))

        loss_logfile = self.logfile_savepath + self.logloss_prefix + self.experiment + '.txt'
        avg_loss = np.average(stream_loss, axis=0)[0, :]
        print('\nEpoch', epoch, 'Train Loss: loss', avg_loss[0])
        with open(loss_logfile, 'a') as fout:
            fout.write(self.loss_log_format % (epoch, avg_loss[0]))

    def checkAPR(self, check_epoch, datastream, stream_name=None):
        log_APRheader = 'Accuracy\tPrecision\tRecall\tF1score\tMCC\n'
        log_APRformat = '%f\t%f\t%f\t%f\t%f\n'
        print('Evaluating ', stream_name)
        Accuracy, Precision, Recall, F1score, MCC = APR().calc_APR(datastream, self.run_model, check_epoch)
        with open(self.logfile_savepath + self.logAPR_prefix + self.experiment + '.txt', 'a') as fout:
            fout.write('Epoch '+str(check_epoch)+'\n')
            fout.write(log_APRheader)
            fout.write(log_APRformat % (Accuracy, Precision, Recall, F1score, MCC))
        fout.close()

    def resume_training_or_not(self, resume_training, resume_epoch):
        if resume_training:
            print('Loading docking model at', str(resume_epoch))
            ckp_path = self.model_savepath+'docking_' + self.experiment + str(resume_epoch) + '.th'
            self.docking_model, self.docking_optimizer, _ = self.load_checkpoint(ckp_path, self.docking_model, self.docking_optimizer)
            print('Loading interaction model at', str(resume_epoch))
            ckp_path = self.model_savepath + self.experiment + str(resume_epoch) + '.th'
            self.interaction_model, self.interaction_optimizer, start_epoch = self.load_checkpoint(ckp_path, self.interaction_model, self.interaction_optimizer)

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

    def run_trainer(self, train_epochs, train_stream=None, valid_stream=None, test_stream=None, resume_training=False, resume_epoch=0):
        self.train_model(train_epochs, train_stream, valid_stream, test_stream,
                         resume_training=resume_training, resume_epoch=resume_epoch)

if __name__ == '__main__':
    #################################################################################
    # Datasets
    trainset = '../../Datasets/interaction_train_400pool'
    validset = '../../Datasets/interaction_valid_400pool'
    ### testing set
    testset = '../../Datasets/interaction_test_400pool'
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
    #########################
    ## number_of_pairs provides max_size of interactions: max_size = int(number_of_pairs + (number_of_pairs**2 - number_of_pairs)/2)
    number_of_pairs = 100
    train_stream = get_interaction_stream(trainset + '.pkl', number_of_pairs=number_of_pairs)
    valid_stream = get_interaction_stream(validset + '.pkl', number_of_pairs=100)
    test_stream = get_interaction_stream(testset + '.pkl', number_of_pairs=100)
    ######################
    # experiment = 'MC_FI_NEWDATA_CHECK_400pool_5000ex30ep'
    # experiment = 'MC_FI_NEWDATA_CHECK_400pool_10000ex50ep'
    # experiment = 'MC_FI_NEWDATA_CHECK_400pool_1000ex50ep'
    # experiment = 'MC_FI_NEWDATA_CHECK_400pool_1000ex50ep10step'
    # experiment = 'MC_FI_NEWDATA_CHECK_400pool_1000ex50ep100step'
    # experiment = 'BF_FI_400pool_1000ex_100ep_10steps_filr1e-1_gamma95'
    # experiment = 'BF_FI_400pool_1000ex_50ep_50steps_filr1e-1_noFreg'

    # experiment = 'BF_FI_400pool_1000ex_50ep_50steps_uniqueEnergies'
    # experiment = 'BF_FI_400pool_1000ex_50ep_10steps_emptysliceFE'
    # experiment = 'BF_FI_400pool_1000ex_50ep_10steps_emptysliceFE_logsumexp'
    # experiment = 'BF_FI_400pool_1000ex_50ep_10steps_uniqueEnergies_zerosangleslist'
    # experiment = 'BF_FI_400pool_1000ex_50ep_50steps_labmeetingcheck'
    # experiment = 'BF_FI_400pool_25pairs_50ep_10steps_labmeetingcheck_fixedsigma'
    # experiment = 'BF_FI_400pool_25pairs_50ep_10steps_freeEvisitedBuffer'
    # experiment = 'BF_FI_400pool_100pairs_50ep_10steps_freeEvisitedBuffer'
    # experiment = 'BF_FI_400pool_100pairs_50ep_10steps_freeEvisitedBuffer_overwriteE'
    # experiment = 'BF_FI_400pool_100pairs_50ep_10steps_FEbufferoverwrite_sig0p5_plotsampsurf'
    # experiment = 'BF_FI_400pool_2pairs_100ep_10steps_FEbufferoverwrite_sig0p5_plotsampsurf'
    # experiment = 'BF_FI_400pool_2pairs_100ep_50steps_FEbufferoverwrite_sig0p5_plotsampsurf'
    # experiment = 'BF_FI_400pool_2pairs_20ep_10steps_FEbufferoverwrite_sig0p05_plotsampsurf'
    # experiment = 'BF_FI_400pool_100pairs_20ep_10steps_FEbufferunique_sig0p05_plotsampsurf'
    # experiment = 'BF_FI_400pool_100pairs_20ep_10steps_FEbufferunique_sig3p0_plotsampsurf_-BFvol'
    # experiment = 'BF_FI_400pool_100pairs_20ep_10steps_FEbufferoverwrite_sig3p0_plotsampsurf_-BFvol'
    # experiment = 'BF_FI_400pool_100pairs_10ep_10steps_FEbufferoverwrite_sig3p0_plotsampsurf_-BFvol'
    experiment = 'BF_FI_400pool_100pairs_10ep_50steps_FEbufferoverwrite_sig3p0_plotsampsurf_-BFvol'

    ######################
    train_epochs = 10
    lr_interaction = 10 ** -1
    lr_docking = 10 ** -4
    sample_steps = 50
    sigma_alpha = 3.0
    # gamma = 0.95

    debug = False
    plotting = False
    show = False

    interaction_model = Interaction().to(device=0)
    interaction_optimizer = optim.Adam(interaction_model.parameters(), lr=lr_interaction)
    # F_0_scheduler = optim.lr_scheduler.ExponentialLR(interaction_optimizer, gamma=gamma)
    # sigma_scheduler = optim.lr_scheduler.ExponentialLR(interaction_optimizer, gamma=gamma)
    dockingFFT = TorchDockingFFT(num_angles=1, angle=None, swap_plot_quadrants=False, debug=debug)
    docking_model = SamplingModel(dockingFFT, num_angles=1, sample_steps=sample_steps, FI=True, debug=debug).to(device=0)
    docking_optimizer = optim.Adam(docking_model.parameters(), lr=lr_docking)
    ######################
    ### Train model from beginning
    # EnergyBasedInteractionTrainer(docking_model, docking_optimizer, interaction_model, interaction_optimizer, experiment, sigma_alpha=sigma_alpha, debug=debug
    #                               ).run_trainer(train_epochs, train_stream=train_stream, valid_stream=None, test_stream=None)

    ### resume training model
    # EnergyBasedInteractionTrainer(docking_model, docking_optimizer, interaction_model, interaction_optimizer, experiment, debug=debug
    #                              ).run_trainer(resume_training=True, resume_epoch=13, train_epochs=27,
    #                                            train_stream=train_stream, valid_stream=None, test_stream=None)

    ### Evaluate model at chosen epoch (Brute force or monte carlo evaluation)
    eval_model = SamplingModel(dockingFFT, num_angles=360, FI=True, debug=debug).to(device=0)
    # # eval_model = SamplingModel(dockingFFT, num_angles=1, sample_steps=sample_steps, FI=True, debug=debug).to(device=0) ## eval with monte carlo
    EnergyBasedInteractionTrainer(eval_model, docking_optimizer, interaction_model, interaction_optimizer, experiment, debug=False
                                  ).run_trainer(resume_training=True, resume_epoch=5, train_epochs=1,
                                                train_stream=None, valid_stream=valid_stream, test_stream=test_stream)

    ### Plot loss and free energy distributions with learned F_0 decision threshold
    PlotterFI(experiment).plot_loss()
    PlotterFI(experiment).plot_deltaF_distribution(plot_epoch=5, show=True)
