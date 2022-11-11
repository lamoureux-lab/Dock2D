import pandas as pd
import torch
from torch import optim
import sys
sys.path.append('/home/sb1638/') ## path for cluster

import numpy as np
from tqdm import tqdm
from Dock2D.Utility.ValidationMetrics import APR
from Dock2D.Utility.SampleBuffer import SampleBuffer
from Dock2D.Utility.UtilityFunctions import UtilityFunctions
from Dock2D.Utility.PlotterFI import PlotterFI


class TrainerFI:
    def __init__(self, docking_model, docking_optimizer, interaction_model, interaction_optimizer, experiment,
                 training_case='scratch', path_pretrain=None, FI_MC=False,
                 debug=False, plotting=False, sample_buffer_length=1000):
        """

        :param docking_model: the current docking model initialized outside the trainer
        :param docking_optimizer: the docking optimizer initialized outside the trainer
        :param interaction_model: the current interaction model initialized outside the trainer
        :param interaction_optimizer: the interaction optimizer initialized outside the trainer
        :param experiment: current experiment name
        :param training_case: current training case
        :param path_pretrain: path to pretrained model
        :param FI_MC: set `True` to use MonteCarlo (MC) for FI task.
        :param debug: set to True show debug verbose model
        :param plotting: create plots or not
        :param sample_buffer_length: number of keys in the SampleBuffer, has to be `>=` to number of training, validation,
            or testing examples.
        """
        self.debug = debug
        self.plotting = plotting
        self.plot_freq = 10
        self.check_epoch = 1
        self.eval_freq = 1
        self.save_freq = 1

        self.model_savepath = 'Log/saved_models/FI_saved/'
        self.logfile_savepath = 'Log/losses/FI_loss/'
        self.logtraindF_prefix = 'log_deltaF_TRAINset_epoch'
        self.logloss_prefix = 'log_loss_TRAINset_'
        self.logAPR_prefix = 'log_validAPR_'
        self.log_saturation_prefix = 'log_MCFI_saturation_stats'

        self.saturation_dict = {}

        self.loss_log_header = 'Epoch\tLoss\n'
        self.loss_log_format = '%d\t%f\n'

        self.deltaf_log_header = 'F\tF_0\tLabel\n'
        self.deltaf_log_format = '%f\t%f\t%d\n'

        # self.saturation_log_header = 'saturation\tfree_energies_visited_indices\tn'
        # self.saturation_log_format = '%f\t%f\n'

        self.docking_model = docking_model
        self.interaction_model = interaction_model
        self.docking_optimizer = docking_optimizer
        self.interaction_optimizer = interaction_optimizer
        self.experiment = experiment

        self.training_case = training_case
        self.path_pretrain = path_pretrain
        self.set_docking_model_state()
        self.freeze_weights()

        self.alpha_buffer = SampleBuffer(num_examples=sample_buffer_length)
        self.free_energy_buffer = SampleBuffer(num_examples=sample_buffer_length)

        self.wReg = 1e-5
        self.zero_value = torch.zeros(1).squeeze().cuda()
        # self.sigma_scheduler_initial = sigma_scheduler.get_last_lr()[0]

        self.UtilityFuncs = UtilityFunctions()
        self.APR = APR()
        self.FI_MC = FI_MC
        self.BF_eval = False

        self.plot_saturation = False

    def train_model(self, train_epochs, train_stream, valid_stream, test_stream, resume_training=False,
                    resume_epoch=0):
        """
        Train model for specified number of epochs and data streams.

        :param train_epochs:  number of epoch to train
        :param train_stream: training set data stream
        :param valid_stream: valid set data stream
        :param test_stream: test set data stream
        :param resume_training: resume training from a loaded model state or train fresh model
        :param resume_epoch: epoch to load model and resume training
        """
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
                'alpha_buffer': self.alpha_buffer,
                'free_energy_buffer': self.free_energy_buffer,
            }

            if train_stream:
                self.run_epoch(train_stream, epoch, training=True)
                PlotterFI(self.experiment).plot_loss(show=False)
                PlotterFI(self.experiment).plot_deltaF_distribution(plot_epoch=epoch, show=False, xlim=None, binwidth=1)

                #### saving model while training
                if epoch % self.save_freq == 0:
                    docking_savepath = self.model_savepath + 'docking_' + self.experiment + str(epoch) + '.th'
                    self.save_checkpoint(docking_checkpoint_dict, docking_savepath, self.docking_model)
                    print('saving docking model ' + docking_savepath)

                    interaction_savepath = self.model_savepath + self.experiment + str(epoch) + '.th'
                    self.save_checkpoint(interaction_checkpoint_dict, interaction_savepath, self.interaction_model)
                    print('saving interaction model ' + interaction_savepath)

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
                    stream_name = 'VALIDset'
                    deltaF_logfile = self.logfile_savepath + stream_name + self.logtraindF_prefix + str(
                        epoch) + self.experiment + '.txt'
                    with open(deltaF_logfile, 'w') as fout:
                        fout.write(self.deltaf_log_header)
                    self.check_APR(epoch, valid_stream, stream_name=stream_name, deltaF_logfile=deltaF_logfile, experiment=self.experiment)
                if test_stream:
                    stream_name = 'TESTset'
                    deltaF_logfile = self.logfile_savepath + stream_name + self.logtraindF_prefix + str(
                        epoch) + self.experiment + '.txt'
                    with open(deltaF_logfile, 'w') as fout:
                        fout.write(self.deltaf_log_header)
                    self.check_APR(epoch, test_stream, stream_name=stream_name, deltaF_logfile=deltaF_logfile, experiment=self.experiment)

    def run_epoch(self, data_stream, epoch, training=False, stream_name='train_stream'):
        """
         Run the model for an epoch.

         :param data_stream: input data stream
         :param epoch: current epoch number
         :param training: set to `True` for training, `False` for evalutation.
         """
        stream_loss = []
        pos_idx = torch.tensor([0])
        deltaF_logfile = self.logfile_savepath + self.logtraindF_prefix + str(epoch) + self.experiment + '.txt'
        with open(deltaF_logfile, 'w') as fout:
            fout.write(self.deltaf_log_header)
        for data in tqdm(data_stream):
            train_output = [self.run_model(data, pos_idx=pos_idx, training=training, stream_name=stream_name, epoch=epoch)]
            stream_loss.append(train_output)
            pos_idx += 1
            with open(deltaF_logfile, 'a') as fout:
                fout.write(self.deltaf_log_format % (train_output[0][1], train_output[0][2], train_output[0][3]))

        loss_logfile = self.logfile_savepath + self.logloss_prefix + self.experiment + '.txt'
        avg_loss = np.average(stream_loss, axis=0)[0, :]
        print('\nEpoch', epoch, 'Train Loss: loss', avg_loss[0])
        with open(loss_logfile, 'a') as fout:
            fout.write(self.loss_log_format % (epoch, avg_loss[0]))

    def run_model(self, data, pos_idx, stream_name, training=True, epoch=0):
        """
        Run a model iteration on the current example.

        :param data: training example
        :param pos_idx: current example position index
        :param training: set to `True` for training, `False` for evalutation.
        :param stream_name: data stream name
        :param epoch: epoch count used in plotting
        :return: `loss`, `F`, `F_0`, `gt_interact` and under evalutation, `TP`, `FP`, `TN`, `FN`, plus previously listed values.
        """
        receptor, ligand, gt_interact = data

        receptor = receptor.to(device='cuda', dtype=torch.float)
        ligand = ligand.to(device='cuda', dtype=torch.float)
        gt_interact = gt_interact.to(device='cuda', dtype=torch.float).squeeze()

        # receptor, ligand, gt_rot, gt_txy = data
        #
        # receptor = receptor.to(device='cuda', dtype=torch.float)
        # ligand = ligand.to(device='cuda', dtype=torch.float)
        # gt_rot = gt_rot.to(device='cuda', dtype=torch.float).squeeze()
        # gt_txy = gt_txy.to(device='cuda', dtype=torch.float).squeeze()

        # print('stream_name', stream_name)
        # print('training', training)

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
        if self.FI_MC:
            if training:
                # print('MC Training =',training)
                alpha = self.alpha_buffer.get_alpha(pos_idx, samples_per_example=1)
                free_energies_visited_indices = self.free_energy_buffer.get_free_energies_indices(pos_idx)

                # print('BUFFER GET: free_energies_visited_indices', free_energies_visited_indices)
                # print('BUFFER GET: free_energies_visited_indices.shape', free_energies_visited_indices.shape)

                free_energies_visited_indices, accumulated_free_energies, pred_rot, pred_txy, fft_score_stack, acceptance_rate = self.docking_model(receptor, ligand, alpha,
                                                free_energies_visited=free_energies_visited_indices,
                                                plot_count=plot_count, stream_name=stream_name, plotting=self.plotting,
                                                training=training)

                # print('BUFFER PUSH: free_energies_visited_indices', free_energies_visited_indices)
                # print('BUFFER PUSH: free_energies_visited_indices.shape', free_energies_visited_indices.shape)

                self.plot_saturation = False
                if self.plot_saturation:
                    with torch.no_grad():
                        self.saturation_dict[pos_idx.item()] = [i.item() for i in free_energies_visited_indices.squeeze()]
                        if pos_idx == 5050-1:
                            df = pd.DataFrame.from_dict(self.saturation_dict, orient='index')
                            df = df.transpose()
                            df.to_csv(self.logfile_savepath + self.log_saturation_prefix + str(epoch) + self.experiment + '.csv',
                                                      # sep='\t'
                                      )
                            sys.exit()

                self.alpha_buffer.push_alpha(pred_rot, pos_idx)
                self.free_energy_buffer.push_free_energies_indices(free_energies_visited_indices, pos_idx)
                pred_interact, deltaF, F, F_0 = self.interaction_model(brute_force=self.BF_eval, fft_scores=fft_score_stack, free_energies_visited=accumulated_free_energies)

                if plot_count % self.plot_freq == 0 and training:
                    UtilityFunctions(self.experiment).plot_MCsampled_energysurface(free_energies_visited_indices, accumulated_free_energies, acceptance_rate,
                                                        stream_name, interaction=gt_interact, plot_count=plot_count, epoch=epoch)
            else:
                # print('MC BF Eval')
                free_energies_visited_indices, accumulated_free_energies, pred_rot, pred_txy, fft_score_stack, acceptance_rate = self.docking_model(
                                                                                receptor, ligand, alpha=None,
                                                                                free_energies_visited=None,
                                                                                plot_count=plot_count, stream_name=stream_name, plotting=self.plotting,
                                                                                training=training)
                pred_interact, deltaF, F, F_0 = self.interaction_model(brute_force=self.BF_eval, fft_scores=fft_score_stack, free_energies_visited=accumulated_free_energies)

        else:
            # print('BF Training =', training)
            fft_score_stack = self.docking_model(receptor, ligand, plot_count=plot_count, stream_name=stream_name, plotting=self.plotting, training=training)
            pred_interact, deltaF, F, F_0 = self.interaction_model(brute_force=True, fft_scores=fft_score_stack)


        ### check parameters and gradients
        ### if weights are frozen or updating
        if self.debug:
            self.UtilityFuncs.check_model_gradients(self.docking_model)
            self.UtilityFuncs.check_model_gradients(self.interaction_model)

        #### Loss functions
        BCEloss = torch.nn.BCELoss()
        l1_loss = torch.nn.L1Loss()
        # print(deltaF, self.zero_value)
        L_reg = self.wReg * l1_loss(deltaF, self.zero_value)
        loss = BCEloss(pred_interact.unsqueeze(0), gt_interact.unsqueeze(0)) + L_reg
        # print(L_reg)
        # print(loss)

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
        """
        Confusion matrix values.

        :param pred_interact: predicted interaction
        :param gt_interact: ground truth interaction
        :return: confusion matrix values `TP`, `FP`, `TN`, `FN`,
        """
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

    def check_APR(self, check_epoch, datastream, stream_name=None, deltaF_logfile=None, experiment=None):
        """
        Check accuracy, precision, recall, F1score and MCC

        :param check_epoch: epoch to evaluate
        :param datastream: data stream
        :param stream_name: data stream name
        :param deltaF_logfile: free energy log file name for evaluation set
        :param experiment: current experiment name
        """
        log_APRheader = 'Accuracy\tPrecision\tRecall\tF1score\tMCC\n'
        log_APRformat = '%f\t%f\t%f\t%f\t%f\n'
        print('Evaluating ', stream_name)
        Accuracy, Precision, Recall, F1score, MCC = self.APR.calc_APR(datastream, self.run_model, check_epoch, deltaF_logfile, experiment, stream_name)
        with open(self.logfile_savepath + self.logAPR_prefix + self.experiment + '.txt', 'a') as fout:
            fout.write('Epoch '+str(check_epoch)+'\n')
            fout.write(log_APRheader)
            fout.write(log_APRformat % (Accuracy, Precision, Recall, F1score, MCC))
        fout.close()

    def resume_training_or_not(self, resume_training, resume_epoch):
        """
        Resume training the model at specified epoch or not.

        :param resume_training: set to `True` to resume training, `False` to start fresh training.
        :param resume_epoch: epoch number to resume from
        :return: starting epoch number, 1 if `resume_training is True`, `resume_epoch+1` otherwise.
        """
        if resume_training:
            print('Loading docking model at', str(resume_epoch))
            ckp_path = self.model_savepath+'docking_' + self.experiment + str(resume_epoch) + '.th'
            self.docking_model, self.docking_optimizer, _ = self.load_checkpoint(ckp_path, self.docking_model, self.docking_optimizer)
            print('Loading interaction model at', str(resume_epoch))
            ckp_path = self.model_savepath + self.experiment + str(resume_epoch) + '.th'
            if self.FI_MC:
                self.interaction_model, self.interaction_optimizer, start_epoch, self.alpha_buffer, self.free_energy_buffer=self.load_checkpoint(
                                                        ckp_path, self.interaction_model, self.interaction_optimizer, FI_MC=self.FI_MC)
            else:
                self.interaction_model, self.interaction_optimizer, start_epoch =self.load_checkpoint(
                                                        ckp_path, self.interaction_model, self.interaction_optimizer, FI_MC=self.FI_MC)

            start_epoch += 1

            ## print model and params being loaded
            print('\ndocking model:\n', self.docking_model)
            self.UtilityFuncs.check_model_gradients(self.docking_model)
            print('\ninteraction model:\n', self.interaction_model)
            self.UtilityFuncs.check_model_gradients(self.interaction_model)

            print('\nRUNNING MODEL AT EPOCH', start_epoch, '\n')
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
        """
        Save current state of the model to a checkpoint dictionary.

        :param state: checkpoint state dictionary
        :param filename: name of saved file
        :param model: model to save, either docking or interaction models
        """
        model.eval()
        torch.save(state, filename)

    @staticmethod
    def load_checkpoint(checkpoint_fpath, model, optimizer, FI_MC=False):
        """
        Load saved checkpoint state dictionary.


        :param checkpoint_fpath: path to saved model
        :param model:  model to load, either docking or interaction models
        :param optimizer: model optimizer
        :param FI_MC: return addtionally saved values from state dict
        :return: `model`, `optimizer`, `checkpoint['epoch']`, and if `FI_MC==True`, additionally return `checkpoint['alpha_buffer']`, `checkpoint['free_energy_buffer']`
        """
        model.eval()
        checkpoint = torch.load(checkpoint_fpath)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        if FI_MC:
            return model, optimizer, checkpoint['epoch'],  checkpoint['alpha_buffer'], checkpoint['free_energy_buffer']
        else:
            return model, optimizer, checkpoint['epoch']

    def freeze_weights(self):
        """
        Freeze model weights depending on the experiment training case.
        These range from A) frozen pretrained docking model,
        B) unfrozen pretrained docking model,
        C) unfrozen pretrained docking model scoring coefficients, but frozen conv net,
        D) train the docking model from scratch.
        """
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

    def set_docking_model_state(self):
        """
        Initialize the docking model training case.
        A) frozen pretrained docking model,
        B) unfrozen pretrained docking model,
        C) frozen conv net, unfrozen pretrained docking model scoring coefficients
        D) train the docking model from scratch.
        """
        # CaseA: train with docking model frozen
        if self.training_case == 'A':
            print('Training expA')
            self.param_to_freeze = 'all'
            self.docking_model.load_state_dict(torch.load(self.path_pretrain)['state_dict'])
        # CaseB: train with docking model unfrozen
        if self.training_case == 'B':
            print('Training expB')
            lr_docking = 10 ** -5
            print('Docking learning rate changed to', lr_docking)
            # self.experiment = 'case' + self.training_case + '_lr5change_' + self.experiment
            # self.docking_model = Docking().to(device=0)
            self.docking_optimizer = optim.Adam(self.docking_model.parameters(), lr=lr_docking)
            self.param_to_freeze = None
            self.docking_model.load_state_dict(torch.load(self.path_pretrain)['state_dict'])
        # CaseC: train with docking model SE2 CNN frozen
        if self.training_case == 'C':
            print('Training expC')
            self.param_to_freeze = 'netSE2'  # leave "a" scoring coefficients unfrozen
            self.docking_model.load_state_dict(torch.load(self.path_pretrain)['state_dict'])
        # Case scratch: train everything from scratch
        if self.training_case == 'scratch':
            print('Training from scratch')
            self.param_to_freeze = None
            # self.experiment = self.training_case + '_' + self.experiment

    def run_trainer(self, train_epochs, train_stream=None, valid_stream=None, test_stream=None, resume_training=False, resume_epoch=0):
        """
        Helper function to run trainer.

        :param train_epochs:  number of epoch to train
        :param train_stream: training set data stream
        :param valid_stream: valid set data stream
        :param test_stream: test set data stream
        :param resume_training: resume training from a loaded model state or train fresh model
        :param resume_epoch: epoch to load model and resume training
        """
        self.train_model(train_epochs, train_stream, valid_stream, test_stream,
                         resume_training=resume_training, resume_epoch=resume_epoch)
