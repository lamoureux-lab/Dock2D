import torch
import sys
sys.path.append('/home/sb1638/') ## path for cluster

import numpy as np
from tqdm import tqdm
from Dock2D.Utility.TorchDockingFFT import TorchDockingFFT
from Dock2D.Models.model_docking import Docking
from Dock2D.Utility.UtilityFunctions import UtilityFunctions
from Dock2D.Utility.ValidationMetrics import RMSD
from Dock2D.Utility.SampleBuffer import SampleBuffer


class TrainerIP:
    def __init__(self, dockingFFT, cur_model, cur_optimizer, cur_experiment, BF_eval=False, MC_eval=False,
                 MC_eval_num_epochs=10, debug=False, plotting=False,
                 sigma_alpha=3.0, sample_buffer_length=1000):
        """
        Initialize trainer for IP task models, paths, and class instances.

        :param dockingFFT:
        :param cur_model: the current docking model initialized outside the trainer
        :param cur_optimizer: the optimizer initialized outside the trainer
        :param cur_experiment: current experiment name
        :param BF_eval:
        :param MC_eval:
        :param MC_eval_num_epochs:
        :param debug: set to `True` to check model parameter gradients
        :param plotting: create plots or not
        :param sigma_alpha:
        :param sample_buffer_length:
        """

        self.debug = debug
        self.plotting = plotting
        self.eval_freq = 1
        self.save_freq = 1
        self.model_savepath = 'Log/saved_models/IP_saved/'
        self.logfile_savepath = 'Log/losses/IP_loss/'
        self.plot_freq = Docking().plot_freq

        self.log_header = 'Epoch\tLoss\trmsd\n'
        self.log_format = '%d\t%f\t%f\n'

        self.dockingFFT = dockingFFT
        self.dim = TorchDockingFFT().dim
        self.num_angles = TorchDockingFFT().num_angles

        self.model = cur_model
        self.optimizer = cur_optimizer
        self.experiment = cur_experiment

        ## sample buffer for BF or MC eval on ideal learned energy surface
        self.alpha_buffer = SampleBuffer(num_examples=sample_buffer_length)
        self.free_energy_buffer = SampleBuffer(num_examples=sample_buffer_length)

        self.BF_eval = BF_eval
        self.MC_eval = MC_eval
        self.MC_eval_num_epochs = MC_eval_num_epochs
        if self.MC_eval:
            self.eval_epochs = self.MC_eval_num_epochs
            self.sigma_alpha = sigma_alpha

        self.UtilityFunctions = UtilityFunctions()

    def train_model(self, train_epochs, train_stream=None, valid_stream=None, test_stream=None,
                    resume_training=False,
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

            checkpoint_dict = {
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }

            if train_stream:
                ### Training epoch
                stream_name = 'TRAINset'
                self.run_epoch(train_stream, epoch, training=True, stream_name=stream_name)

                #### saving model while training
                if epoch % self.save_freq == 0:
                    model_savefile = self.model_savepath + self.experiment + str(epoch) + '.th'
                    self.save_checkpoint(checkpoint_dict, model_savefile)
                    print('saving model ' + model_savefile)

            ### Evaluation epoch(s)
            if epoch % self.eval_freq == 0 or epoch == 1:
                if self.MC_eval:
                    self.resume_training_or_not(resume_training, resume_epoch)
                    for i in range(self.eval_epochs):
                        print('Monte Carlo eval epoch', i)
                        print('current sigma_alpha', self.sigma_alpha)
                        rmsd_validlog = self.logfile_savepath + 'log_RMSDsVALIDset_epoch' + str(
                            epoch - 1) + self.experiment + '.txt'
                        rmsd_testlog = self.logfile_savepath + 'log_RMSDsTESTset_epoch' + str(
                            epoch - 1) + self.experiment + '.txt'
                        with open(rmsd_validlog, 'w') as fout:
                            fout.write('Validation RMSD\n')
                        with open(rmsd_testlog, 'w') as fout:
                            fout.write('Testing RMSD\n')

                        if valid_stream:
                            stream_name = 'VALIDset'
                            self.run_epoch(valid_stream, epoch, training=False, stream_name=stream_name)
                        if test_stream:
                            stream_name = 'TESTset'
                            self.run_epoch(test_stream, epoch, training=False, stream_name=stream_name)

                else:
                    if valid_stream:
                        stream_name = 'VALIDset'
                        self.run_epoch(valid_stream, epoch, training=False, stream_name=stream_name)
                    if test_stream:
                        stream_name = 'TESTset'
                        self.run_epoch(test_stream, epoch, training=False, stream_name=stream_name)

    def run_epoch(self, data_stream, epoch, training=False, stream_name='train_stream'):
        """
        Run the model for an epoch.

        :param data_stream: input data stream
        :param epoch: current epoch number
        :param training: set to `True` for training, `False` for evalutation.
        :param stream_name: name of the data stream
        """
        stream_loss = []
        pos_idx = torch.tensor([0])
        rmsd_logfile = self.logfile_savepath + 'log_RMSDs' + stream_name + '_epoch' + str(epoch) + self.experiment + '.txt'
        for data in tqdm(data_stream):
            train_output = [
                self.run_model(data, pos_idx=pos_idx, training=training, stream_name=stream_name, epoch=epoch)]
            stream_loss.append(train_output)
            with open(rmsd_logfile, 'a') as fout:
                fout.write('%f\n' % (train_output[0][-1]))
                fout.close()
            pos_idx += 1

        loss_logfile = self.logfile_savepath + 'log_loss_' + stream_name + '_' + self.experiment + '.txt'
        avg_loss = np.average(stream_loss, axis=0)[0, :]
        print('\nEpoch', epoch, stream_name, ':', avg_loss)
        with open(loss_logfile, 'a') as fout:
            fout.write(self.log_format % (epoch, avg_loss[0], avg_loss[1]))
            fout.close()

    def run_model(self, data, pos_idx, training=True, stream_name='trainset', epoch=0):
        """
        Run a model iteration on the current example.

        :param data: training example
        :param pos_idx: current example position index
        :param training: set to `True` for training, `False` for evalutation.
        :param stream_name: data stream name
        :param epoch: epoch count used in plotting
        :return:
        """

        receptor, ligand, gt_rot, gt_txy = data

        receptor = receptor.to(device='cuda', dtype=torch.float)
        ligand = ligand.to(device='cuda', dtype=torch.float)
        gt_rot = gt_rot.to(device='cuda', dtype=torch.float).squeeze()
        gt_txy = gt_txy.to(device='cuda', dtype=torch.float).squeeze()

        if training:
            self.model.train()
        else:
            self.model.eval()

        ### run model and loss calculation
        ##### call model
        plot_count = int(pos_idx)
        if training:
            lowest_energy, pred_rot, pred_txy, fft_score = self.model(receptor, ligand, gt_rot, plot_count=plot_count,
                                                                      stream_name=stream_name, plotting=self.plotting)
        else:
            if self.BF_eval:
                lowest_energy, pred_rot, pred_txy, fft_score = self.model(receptor, ligand, plot_count=plot_count,
                                                                          stream_name=stream_name, plotting=self.plotting,
                                                                          training=False)
            else:
                ## for evaluation, sample buffer is necessary for Monte Carlo multi epoch eval
                alpha = self.alpha_buffer.get_alpha(pos_idx, samples_per_example=1)
                free_energies_visited_indices = self.free_energy_buffer.get_free_energies_indices(pos_idx)

                free_energies_visited_indices, accumulated_free_energies, pred_rot, pred_txy, fft_score, acceptance_rate = \
                                self.model(alpha, receptor, ligand,
                                free_energies_visited=free_energies_visited_indices, sig_alpha=self.sigma_alpha,
                                plot_count=plot_count, stream_name=stream_name, plotting=self.plotting, training=False)
                self.free_energy_buffer.push_free_energies_indices(free_energies_visited_indices, pos_idx)
                self.alpha_buffer.push_alpha(pred_rot, pos_idx)

                if plot_count % self.plot_freq == 0:
                    UtilityFunctions(self.experiment).plot_MCsampled_energysurface(free_energies_visited_indices,
                                                                                   accumulated_free_energies,
                                                                                   acceptance_rate,
                                                                                   stream_name, plot_count=plot_count,
                                                                                   epoch=epoch)

        ### Encode ground truth transformation index into empty energy grid
        with torch.no_grad():
            rmsd_out = RMSD(ligand, gt_rot, gt_txy, pred_rot.squeeze(), pred_txy.squeeze()).calc_rmsd()
            target_flatindex = self.dockingFFT.encode_transform(gt_rot, gt_txy)

        if self.debug:
            print('\npredicted')
            print(pred_rot, pred_txy)
            print('\nground truth')
            print(gt_rot, gt_txy)

        ### check parameters and gradients
        ### if weights are frozen or updating
        if self.debug:
            self.UtilityFunctions.check_model_gradients(self.model)

        if training:
            #### Loss functions
            CE_loss = torch.nn.CrossEntropyLoss()
            loss = CE_loss(fft_score.flatten().unsqueeze(0), target_flatindex.unsqueeze(0))
            self.model.zero_grad()
            loss.backward()
            self.optimizer.step()
        else:
            loss = torch.zeros(1)
            self.model.eval()
            if self.plotting and pos_idx % self.plot_freq == 0:
                with torch.no_grad():
                    self.UtilityFunctions.plot_predicted_pose(receptor, ligand, gt_rot, gt_txy, pred_rot.squeeze(),
                                                              pred_txy.squeeze(), pos_idx, stream_name)
        return loss.item(), rmsd_out.item()

    def save_checkpoint(self, state, filename):
        """
        Save current state of the model to a checkpoint dictionary.

        :param state: checkpoint state dictionary
        :param filename: name of saved file
        """
        self.model.eval()
        torch.save(state, filename)

    def load_checkpoint(self, checkpoint_fpath):
        """
        Load saved checkpoint state dictionary.

        :param checkpoint_fpath: path to saved model
        :return: `self.model`, `self.optimizer`, `checkpoint['epoch']`
        """
        self.model.eval()
        checkpoint = torch.load(checkpoint_fpath)
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        return self.model, self.optimizer, checkpoint['epoch']

    def resume_training_or_not(self, resume_training, resume_epoch):
        """
        Resume training the model at specified epoch or not.

        :param resume_training: set to `True` to resume training, `False` to start fresh training.
        :param resume_epoch: epoch number to resume from
        :return: starting epoch number, 1 if `resume_training is True`, `resume_epoch+1` otherwise.
        """
        if resume_training:
            ckp_path = self.model_savepath + self.experiment + str(resume_epoch) + '.th'
            self.model, self.optimizer, start_epoch = self.load_checkpoint(ckp_path)
            start_epoch += 1

            # print(self.model)
            # print(list(self.model.named_parameters()))
            print('\nRESUMING TRAINING AT EPOCH', start_epoch, '\n')
            with open(self.logfile_savepath + 'log_RMSDsTRAINset_epoch' + str(start_epoch) + self.experiment + '.txt',
                      'w') as fout:
                fout.write('IP Training RMSD\n')
            with open(self.logfile_savepath + 'log_RMSDsVALIDset_epoch' + str(start_epoch) + self.experiment + '.txt',
                      'w') as fout:
                fout.write('IP Validation RMSD\n')
            with open(self.logfile_savepath + 'log_RMSDsTESTset_epoch' + str(start_epoch) + self.experiment + '.txt',
                      'w') as fout:
                fout.write('IP Testing RMSD\n')
        else:
            start_epoch = 1
            ### Loss log files
            with open(self.logfile_savepath + 'log_loss_TRAINset_' + self.experiment + '.txt', 'w') as fout:
                fout.write('IP Training Loss:\n')
                fout.write(self.log_header)
            with open(self.logfile_savepath + 'log_loss_VALIDset_' + self.experiment + '.txt', 'w') as fout:
                fout.write('IP Validation Loss:\n')
                fout.write(self.log_header)
            with open(self.logfile_savepath + 'log_loss_TESTset_' + self.experiment + '.txt', 'w') as fout:
                fout.write('IP Testing Loss:\n')
                fout.write(self.log_header)
        return start_epoch

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
