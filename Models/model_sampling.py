import sys

sys.path.append('/home/sb1638/')

import torch
from torch import optim
import torch.nn as nn
import numpy as np

from Dock2D.Models.model_docking import Docking
from Dock2D.Utility.UtilityFunctions import UtilityFunctions


class SamplingDocker(nn.Module):
    def __init__(self, dockingFFT,  debug=False):
        """
        Initialize docking FFT and feature generation using the SE(2)-CNN.

        :param dockingFFT: dockingFFT initialized to match dimensions of current sampling scheme
        :param num_angles: If a single rotation slice correlation is desired, specify `num_angles=1`,
            else `num_angles` is the number of angles to linearly space `-pi` to `+pi`
        :param debug:  set to True show debug verbose model and plots
        """
        super(SamplingDocker, self).__init__()
        self.dockingFFT = dockingFFT
        self.num_angles = self.dockingFFT.num_angles
        self.dockingConv = Docking(dockingFFT=self.dockingFFT, debug=debug)

    def forward(self, receptor, ligand, rotation=None, plot_count=1, stream_name='trainset', plotting=False, training=False):
        """
        Uses TorchDockingFFT() to compute feature correlations for a rotationally sampled stack of examples.

        :param receptor: receptor shape grid image
        :param ligand: ligand shape grid image
        :param rotation: pass rotation for single angle correlation
        :param plot_count: current plotting index
        :param stream_name: data stream name
        :param plotting: create plots or not
        :return: `lowest_energy`, `pred_rot`, `pred_txy`, `fft_score`
        """
        # if 'trainset' not in stream_name:
        #     training = False
        # else:
        #     training = True

        if self.num_angles == 360:
            stream_name = 'BFeval_' + stream_name
        else:
            plotting = False

        # print('calling dockingConv')
        # print('training', 'plotting', training, plotting)

        fft_score = self.dockingConv(receptor, ligand, angle=rotation, plotting=plotting, training=training,
                                             plot_count=plot_count, stream_name=stream_name)

        with torch.no_grad():
            pred_rot, pred_txy = self.dockingFFT.extract_transform(fft_score)

            if len(fft_score.shape) > 2:
                deg_index_rot = (((pred_rot * 180.0 / np.pi) + 180.0) % self.num_angles).type(torch.long)
                best_score = fft_score[deg_index_rot, pred_txy[0], pred_txy[1]]
                if plotting and self.num_angles == 360 and plot_count % 10 == 0:
                    UtilityFunctions().plot_rotation_energysurface(fft_score, pred_txy, self.num_angles, stream_name,
                                                                   plot_count)
                    # import matplotlib.pyplot as plt
                    # plt.show()
            else:
                best_score = fft_score[pred_txy[0], pred_txy[1]]

        lowest_energy = best_score

        return lowest_energy, pred_rot, pred_txy, fft_score


class SamplingModel(nn.Module):
    def __init__(self, dockingFFT, sample_steps=10, step_size=10, sig_alpha=2,
                 IP=False, IP_MC=False, IP_LD=False,
                 FI_BF=False, FI_MC=False,
                 experiment=None):
        """
        Initialize sampling for the two molecular recognition tasks, IP and FI.
        For IP, BruteForce (BF) and BruteSimplified (BS).
        For FI, BruteForce and MonteCarlo(MC)

        .. note::

            Langevin dynamics (LD) code for IP is included but not reported in our work.

        :param dockingFFT: dockingFFT initialized to match dimensions of current sampling scheme
        :param num_angles: If a single rotation slice correlation is desired, specify `num_angles=1`,
            else `num_angles` is the number of angles to linearly space `-pi` to `+pi`
        :param sample_steps: number of samples per example
        :param step_size: step size to use in the LD energy gradient based method
        :param sig_alpha: sigma for rotation used in LD
        :param IP: interaction pose prediction used for either BF or BS, only difference is the `num_angles` specified.
            If `num_angles==1` runs the BS model, and BF otherwise.
        :param IP_MC: Interaction pose trained using BS for docking features, and evaluation using MC.
        :param IP_LD: Interaction pose trained using either BS or BF and evaluated using LD
        :param FI_BF: Fact of interaction trained and evaluated using BF
        :param FI_MC: Fact of interaction trained using MC and evaluated using BF
        :param experiment: current experiment name
        """
        super(SamplingModel, self).__init__()

        self.docker = SamplingDocker(dockingFFT)

        self.sample_steps = sample_steps
        self.step_size = step_size
        self.plot_idx = 0

        self.experiment = experiment
        self.sig_alpha = sig_alpha
        self.step_size = self.sig_alpha

        self.IP = IP
        self.IP_MC = IP_MC
        self.IP_LD = IP_LD

        self.FI_BF = FI_BF
        self.FI_MC = FI_MC

        self.random_walk_steps = 99
        self.rot_step = 0.01745329251

    def forward(self, receptor, ligand, alpha=None, free_energies_visited=None, sig_alpha=None, plot_count=1, stream_name='trainset', plotting=False,
                training=True):
        """
        Run models and sampling for the two molecular recognition tasks, IP and FI.
        For IP, BruteForce (BF) and BruteSimplified (BS).
        For FI, BruteForce and MonteCarlo(MC)

        :param receptor: receptor shape grid image
        :param ligand: ligand shape grid image
        :param alpha: rotation, default is `None`
        :param free_energies_visited: free energies indices for `FI_MC`
        :param sig_alpha: sigma alpha used in LD
        :param plot_count: current plotting index
        :param stream_name: data stream name
        :param plotting: create plots or not
        :param training: train if `True`, else evaluate
        :return: depends on which model and task is being trained/evaluated
        """
        if sig_alpha: ## for Langevin Dynamics
            self.sig_alpha = sig_alpha
            self.step_size = sig_alpha

        if self.IP:
            if training:
                ## both BS/BF model train
                lowest_energy, _, dr, fft_score = self.docker(receptor, ligand, alpha,
                                                              plot_count=plot_count, stream_name=stream_name,
                                                              plotting=plotting)

                return lowest_energy, alpha.unsqueeze(0).clone(), dr.clone(), fft_score
            else:
                ## brute force eval
                self.docker.eval()
                lowest_energy, alpha, dr, fft_score = self.docker(receptor, ligand, plot_count=plot_count,
                                                                  stream_name=stream_name, plotting=plotting)

                return lowest_energy, alpha.unsqueeze(0).clone(), dr.unsqueeze(0).clone(), fft_score

        if self.IP_MC:
            if training:
                ## BS model train giving the ground truth rotation
                lowest_energy, _, dr, fft_score = self.docker(receptor, ligand, alpha,
                                                              plot_count=plot_count, stream_name=stream_name,
                                                              plotting=plotting)

                return lowest_energy, alpha.unsqueeze(0).clone(), dr.clone(), fft_score
            else:
                ## MC sampling eval
                self.docker.eval()
                return self.montecarlo_sampling(alpha, receptor, ligand, plot_count, stream_name, free_energies_visited)

        if self.IP_LD:
            if training:
                ## train either using BF or BS
                lowest_energy, _, dr, fft_score = self.docker(receptor, ligand, alpha,
                                                              plot_count=plot_count, stream_name=stream_name,
                                                              plotting=plotting)

                return lowest_energy, alpha.unsqueeze(0).clone(), dr.clone(), fft_score
            else:
                ## Langegvin sampling eval
                self.docker.eval()
                return self.langevin_dynamics(alpha, receptor, ligand, plot_count, stream_name)

        if self.FI_BF:
            if training:
                _, _, _, fft_score = self.docker(receptor, ligand, alpha,
                                                              plot_count=plot_count, stream_name=stream_name,
                                                              plotting=plotting)
                return fft_score
            else:
                ## brute force eval
                self.docker.eval()
                _, _, _, fft_score = self.docker(receptor, ligand, plot_count, plot_count=plot_count,
                                                                  stream_name=stream_name, plotting=plotting, training=training)

                return fft_score

        if self.FI_MC:
            if training:
                ## MC sampling for Fact of Interaction training
                return self.montecarlo_sampling(alpha, receptor, ligand, plot_count, stream_name, free_energies_visited)
            else:
                print('evaluating MC model with bruteforce')
                print('training', training)
                print('stream_name', stream_name)
                ### evaluate with brute force
                self.docker.eval()
                lowest_energy, _, dr, fft_score = self.docker(receptor, ligand, alpha, plot_count,
                                                              stream_name, plotting=plotting, training=training)
                current_free_energies = None
                acceptance_rate = None
                return lowest_energy, current_free_energies, _, dr.unsqueeze(0).clone(), fft_score, acceptance_rate
                ### evaluate with Monte Carlo?
                # self.docker.eval()
                # return self.MCsampling(alpha, receptor, ligand, plot_count, stream_name, debug=False)

    def montecarlo_sampling(self, alpha, receptor, ligand, plot_count, stream_name, free_energies_visited_indices=None):
        """
        Monte Carlo sampling for free energy surface approximation.
        The `alpha` and `free_energies_visited_indices` are initialized, sampled, and pushed per example per epoch, using instances of SampleBuffer.

        :param alpha: previously encountered rotation initialized from SampleBuffer
        :param receptor: receptor shape grid image
        :param ligand: ligand shape grid image
        :param plot_count: current plotting index
        :param stream_name: name of data stream
        :param free_energies_visited_indices: array of previously encountered free energies indices.
        :return: `free_energies_visited_indices`, `accumulated_free_energies`, `alpha`, `dr`, `fft_score_stack`, `acceptance_rate`
        """
        self.docker.eval()

        accumulated_free_energies = torch.tensor([[]]).cuda()
        for index_alpha in free_energies_visited_indices[0]:
            alpha_update = (index_alpha * np.pi / 180)
            _, _, _, fft_score_update = self.docker(receptor, ligand, alpha_update)
            E_update = fft_score_update
            free_energy = -(torch.logsumexp(-E_update, dim=(0, 1)))
            accumulated_free_energies = torch.cat((accumulated_free_energies, free_energy.reshape(1,1)), dim=1)

        _, _, dr, fft_score = self.docker(receptor, ligand, alpha,
                                          plot_count=plot_count, stream_name=stream_name,
                                          plotting=False)
        E = fft_score
        free_energy = -(torch.logsumexp(-E, dim=(0, 1)))

        prob_list = []
        acceptance = []
        fft_score_list = []
        for i in range(self.sample_steps):
            if i == self.sample_steps - 1:
                plotting = True
            else:
                plotting = False

            rand_rot = 0
            for i in range(self.random_walk_steps):
                if torch.rand(1).float() >= 0.5:
                    rand_rot += self.rot_step
                else:
                    rand_rot -= self.rot_step
            alpha_new = alpha + rand_rot
            _, _, dr_new, fft_score_new = self.docker(receptor, ligand, alpha_new,
                                                      plot_count=plot_count, stream_name=stream_name,
                                                      plotting=plotting)
            E_new = fft_score_new
            free_energy_new = -(torch.logsumexp(-E_new, dim=(0, 1)))

            ## Accept
            if free_energy_new <= free_energy:
                acceptance.append(1)
                prob_list.append(1)
                free_energy = free_energy_new
                alpha = alpha_new
                dr = dr_new
                fft_score = fft_score_new
                fft_score_list.append(fft_score)
                deg_index_alpha = (((alpha * 180.0 / np.pi) + 180.0) % 360).type(torch.long)
                free_energies_visited_indices = torch.cat((free_energies_visited_indices, deg_index_alpha.reshape(1,1)), dim=1)
                accumulated_free_energies = torch.cat((accumulated_free_energies, free_energy.reshape(1,1)), dim=1)
            else:
                prob = min(torch.exp(-(free_energy_new - free_energy)).item(), 1)
                rand0to1 = torch.rand(1).cuda()
                prob_list.append(prob)
                ## Accept
                if prob > rand0to1:
                    acceptance.append(1)
                    free_energy = free_energy_new
                    alpha = alpha_new
                    dr = dr_new
                    fft_score = fft_score_new
                    fft_score_list.append(fft_score)
                    deg_index_alpha = (((alpha * 180.0 / np.pi) + 180.0) % 360).type(torch.long)
                    free_energies_visited_indices = torch.cat((free_energies_visited_indices, deg_index_alpha.reshape(1,1)), dim=1)
                    accumulated_free_energies = torch.cat((accumulated_free_energies, free_energy.reshape(1,1)), dim=1)
                else:
                    ## Reject
                    fft_score_list.append(fft_score)
                    pass

        acceptance_rate = sum(acceptance) / self.sample_steps

        if self.FI_MC or self.IP_MC:
            fft_score_stack = torch.stack(fft_score_list)
        else:
            fft_score_stack = fft_score
            free_energies_visited_indices = free_energy

        self.docker.train()

        return free_energies_visited_indices, accumulated_free_energies, alpha.clone(), dr.clone(), fft_score_stack.squeeze(), acceptance_rate

    def langevin_dynamics(self, alpha, receptor, ligand, plot_count, stream_name):
        """
        Langevin dynamics sampling for free energy surface approximation.

        :param alpha: previously encountered rotation initialized from SampleBuffer
        :param receptor: receptor shape grid image
        :param ligand: ligand shape grid image
        :param plot_count: current plotting index
        :param stream_name: name of data stream
        :return: `energy`, `alpha`, `dr`, `fft_score`
        """

        noise_alpha = torch.zeros_like(alpha)
        langevin_opt = optim.SGD([alpha], lr=self.step_size, momentum=0.0)

        energy = None
        dr = None
        fft_score = None
        for i in range(self.sample_steps):
            if i == self.sample_steps - 1:
                plotting = True
            else:
                plotting = False

            langevin_opt.zero_grad()

            energy, _, dr, fft_score = self.docker(receptor, ligand, alpha, plot_count, stream_name, plotting=plotting)
            # energy = -(torch.logsumexp(fft_scores, dim=(0, 1)) - torch.log(torch.tensor(100 ** 2)))

            energy.backward()
            langevin_opt.step()
            # a, b, n = 40, 4, 4  # 100steps RMSD 38.2
            # self.sigma_alpha = float(b * torch.exp(-(energy / a) ** n))
            # self.step_size = self.sigma_alpha
            rand_rot = noise_alpha.normal_(0, self.sig_alpha)
            alpha = alpha + rand_rot

        return energy, alpha.clone(), dr.clone(), fft_score


if __name__ == "__main__":
    pass
