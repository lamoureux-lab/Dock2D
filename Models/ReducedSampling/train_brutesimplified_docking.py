import torch
import random
from torch import optim
import sys
sys.path.append('/home/sb1638/') ## path for cluster

import numpy as np
from tqdm import tqdm
from Dock2D.Utility.TorchDataLoader import get_docking_stream
from Dock2D.Utility.TorchDockingFFT import TorchDockingFFT
from Dock2D.Utility.UtilityFunctions import UtilityFunctions
from Dock2D.Utility.validation_metrics import RMSD
from Dock2D.Utility.PlotterIP import PlotterIP
from Dock2D.Utility.SampleBuffer import SampleBuffer
from Dock2D.Models.BruteForce.train_bruteforce_docking import Docking
from Dock2D.Models.ReducedSampling.model_sampling import SamplingModel


class BruteSimplifiedDockingTrainer:
    def __init__(self, dockingFFT, cur_model, cur_optimizer, cur_experiment, MC_eval=False, MC_eval_num_epochs=10, debug=False, plotting=False,
                 sigma_scheduler=None, sigma_alpha=3.0, sample_buffer_length=1000):

        self.debug = debug
        self.plotting = plotting
        self.eval_freq = 1
        self.save_freq = 1
        self.model_savepath = 'Log/saved_models/'
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
        self.evalbuffer = SampleBuffer(num_examples=sample_buffer_length)
        self.free_energy_buffer = SampleBuffer(num_examples=sample_buffer_length)

        self.MC_eval = MC_eval
        self.MC_eval_num_epochs = MC_eval_num_epochs
        if self.MC_eval:
            self.eval_epochs = self.MC_eval_num_epochs
            self.sigma_alpha = sigma_alpha
            # self.sigma_alpha = sigma_scheduler.get_last_lr()[0]
            # print('sigma alpha', self.sigma_alpha)
        else:
            self.eval_epochs = 1
            self.sigma_alpha = 1

        self.UtilityFunctions = UtilityFunctions()

    def run_model(self, data, training=True, pos_idx=torch.tensor([0]), stream_name='trainset', epoch=0):
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
            neg_energy, pred_rot, pred_txy, fft_score = self.model(gt_rot, receptor, ligand, plot_count=plot_count, stream_name=stream_name, plotting=self.plotting)
        else:
            ## for evaluation, sample buffer is necessary for Monte Carlo multi epoch eval
            alpha = self.evalbuffer.get(pos_idx, samples_per_example=1)
            free_energies_visited = self.free_energy_buffer.get_free_energies(pos_idx)

            free_energies_visited, pred_rot, pred_txy, fft_score, acceptance_rate = self.model(alpha, receptor, ligand,
                                                                                               free_energies_visited=free_energies_visited, sig_alpha=self.sigma_alpha,
                                                                                               plot_count=plot_count, stream_name=stream_name, plotting=self.plotting, training=False)
            self.free_energy_buffer.push_free_energies(free_energies_visited, pos_idx)
            self.evalbuffer.push(pred_rot, pos_idx)

            if plot_count % self.plot_freq == 0:
                # print(free_energies_visited.shape)
                # print(free_energies_visited)
                UtilityFunctions(self.experiment).plot_MCsampled_energysurface(free_energies_visited, acceptance_rate,
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
                    self.UtilityFunctions.plot_predicted_pose(receptor, ligand, gt_rot, gt_txy, pred_rot.squeeze(), pred_txy.squeeze(), pos_idx, stream_name)
        return loss.item(), rmsd_out.item()

    def train_model(self, train_epochs, train_stream=None, valid_stream=None, test_stream=None,
                    resume_training=False,
                    resume_epoch=0, sigma_optimizer=None, sigma_scheduler=None):

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
                for i in range(self.eval_epochs):
                    self.resume_training_or_not(resume_training, resume_epoch)
                    if valid_stream:
                        stream_name = 'VALIDset'
                        self.run_epoch(valid_stream, epoch-1, training=False, stream_name=stream_name)
                    if test_stream:
                        stream_name = 'TESTset'
                        self.run_epoch(test_stream, epoch-1, training=False, stream_name=stream_name)

                    if self.MC_eval:
                        sigma_optimizer.step()
                        sigma_scheduler.step()
                        self.sigma_alpha = sigma_scheduler.get_last_lr()[0]
                        print('Monte Carlo eval epoch', i)
                        print('sigma_alpha stepped', self.sigma_alpha)

    def run_epoch(self, data_stream, epoch, training=False, stream_name='train_stream'):
        stream_loss = []
        pos_idx = 0
        rmsd_logfile = self.logfile_savepath + 'log_RMSDs'+stream_name+'_epoch' + str(epoch) + self.experiment + '.txt'
        for data in tqdm(data_stream):
            train_output = [self.run_model(data, pos_idx=torch.tensor([pos_idx]), training=training, stream_name=stream_name, epoch=epoch)]
            stream_loss.append(train_output)
            with open(rmsd_logfile, 'a') as fout:
                fout.write('%f\n' % (train_output[0][-1]))
            pos_idx += 1

        loss_logfile = self.logfile_savepath + 'log_loss_' + stream_name + '_' + self.experiment + '.txt'
        avg_loss = np.average(stream_loss, axis=0)[0, :]
        print('\nEpoch', epoch, stream_name, ':', avg_loss)
        with open(loss_logfile, 'a') as fout:
            fout.write(self.log_format % (epoch, avg_loss[0], avg_loss[1]))

    def save_checkpoint(self, state, filename):
        self.model.eval()
        torch.save(state, filename)

    def load_checkpoint(self, checkpoint_fpath):
        self.model.eval()
        checkpoint = torch.load(checkpoint_fpath)
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        return self.model, self.optimizer, checkpoint['epoch']

    def resume_training_or_not(self, resume_training, resume_epoch):
        if resume_training:
            ckp_path = self.model_savepath + self.experiment + str(resume_epoch) + '.th'
            self.model, self.optimizer, start_epoch = self.load_checkpoint(ckp_path)
            start_epoch += 1

            # print(self.model)
            # print(list(self.model.named_parameters()))
            print('\nRESUMING TRAINING AT EPOCH', start_epoch, '\n')
            ### RMSD log files
            rmsd_trainlog = self.logfile_savepath+'log_RMSDsTRAINset_epoch' + str(start_epoch) + self.experiment + '.txt'
            rmsd_validlog = self.logfile_savepath + 'log_RMSDsVALIDset_epoch' + str(start_epoch) + self.experiment + '.txt'
            rmsd_testlog = self.logfile_savepath+'log_RMSDsTESTset_epoch' + str(start_epoch) + self.experiment + '.txt'
            with open(rmsd_trainlog, 'w') as fout:
                fout.write('Training RMSD\n')
            with open(rmsd_validlog, 'w') as fout:
                fout.write('Validation RMSD\n')
            with open(rmsd_testlog, 'w') as fout:
                fout.write('Testing RMSD\n')
        else:
            start_epoch = 1
            ### Loss log files
            loss_trainlog = self.logfile_savepath+'log_loss_TRAINset_' + self.experiment + '.txt'
            loss_validlog = self.logfile_savepath+'log_loss_VALIDset_' + self.experiment + '.txt'
            loss_testlog = self.logfile_savepath+'log_loss_TESTset_' + self.experiment + '.txt'
            with open(loss_trainlog, 'w') as fout:
                fout.write('Docking Training Loss:\n')
                fout.write(self.log_header)
            with open(loss_validlog, 'w') as fout:
                fout.write('Docking Validation Loss:\n')
                fout.write(self.log_header)
            with open(loss_testlog, 'w') as fout:
                fout.write('Docking Testing Loss:\n')
                fout.write(self.log_header)
        return start_epoch

    def run_trainer(self, train_epochs, train_stream=None, valid_stream=None, test_stream=None, resume_training=False, resume_epoch=0,
                    sigma_scheduler=None, sigma_optimizer=None):
        self.train_model(train_epochs, train_stream, valid_stream, test_stream,
                         resume_training=resume_training, resume_epoch=resume_epoch,
                         sigma_scheduler=sigma_scheduler, sigma_optimizer=sigma_optimizer)


if __name__ == '__main__':
    #################################################################################
    # Datasets
    trainset = '../../Datasets/docking_train_400pool'
    validset = '../../Datasets/docking_valid_400pool'
    ### testing set
    testset = '../../Datasets/docking_test_400pool'
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
    train_stream = get_docking_stream(trainset + '.pkl', max_size=max_size)
    valid_stream = get_docking_stream(validset + '.pkl', max_size=max_size)
    test_stream = get_docking_stream(testset + '.pkl', max_size=max_size)
    sample_buffer_length = max(len(train_stream), len(valid_stream), len(test_stream))
    ######################
    # experiment = 'BS_IP_FINAL_DATASET_400pool_1000ex_30ep'
    # experiment = 'BS_IP_FINAL_DATASET_400pool_1000ex_5ep'
    # experiment = 'BS_IP_FINAL_DATASET_400pool_ALLex_30ep'
    experiment = 'BS_IP_FINAL_DATASET_400pool_1000ex_10ep'
    ######################
    train_epochs = 10
    lr = 10 ** -2
    debug = False
    plotting = False
    show = True
    norm = 'ortho'
    #####################
    dockingFFT = TorchDockingFFT(num_angles=1, angle=None, swap_plot_quadrants=False, debug=debug, normalization=norm)
    model = SamplingModel(dockingFFT, num_angles=1, IP=True, debug=debug).to(device=0)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    ######################
    ### Train model from beginning
    # BruteSimplifiedDockingTrainer(dockingFFT, model, optimizer, experiment, debug=debug).run_trainer(train_epochs, train_stream=train_stream)

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
    eval_model = SamplingModel(dockingFFT, num_angles=eval_angles, IP=True).to(device=0)
    for epoch in range(start, stop):
        ### Evaluate model using all 360 angles (or less).
        if stop-1 == epoch:
            plotting = False
            BruteSimplifiedDockingTrainer(dockingFFT, eval_model, optimizer, experiment, plotting=plotting, sample_buffer_length=sample_buffer_length).run_trainer(
            train_epochs=1, train_stream=None, valid_stream=valid_stream, test_stream=test_stream,
            resume_training=True, resume_epoch=epoch)

    ## Plot loss and RMSDs from current experiment
    PlotterIP(experiment).plot_loss(ylim=None)
    PlotterIP(experiment).plot_rmsd_distribution(plot_epoch=train_epochs, show=show)
