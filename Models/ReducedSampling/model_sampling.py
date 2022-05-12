import sys

sys.path.append('/home/sb1638/')

import torch
from torch import optim
import torch.nn as nn
import numpy as np
from matplotlib import pylab as plt

from DeepProteinDocking2D.Models.model_docking import Docking
from DeepProteinDocking2D.Utility.utility_functions import UtilityFuncs


class Docker(nn.Module):
    def __init__(self, dockingFFT, num_angles=1, debug=False):
        super(Docker, self).__init__()
        self.num_angles = num_angles
        self.dim = 100
        self.dockingConv = Docking(dim=self.dim, num_angles=self.num_angles, debug=debug)
        self.dockingFFT = dockingFFT

    def forward(self, receptor, ligand, rotation, plot_count=1, stream_name='trainset', plotting=False):
        if 'trainset' not in stream_name:
            training = False
        else:
            training = True

        if self.num_angles == 360:
            stream_name = 'BFeval_' + stream_name
        else:
            plotting = False

        fft_score = self.dockingConv.forward(receptor, ligand, angle=rotation, plotting=plotting, training=training,
                                             plot_count=plot_count, stream_name=stream_name)

        with torch.no_grad():
            pred_rot, pred_txy = self.dockingFFT.extract_transform(fft_score)

            if len(fft_score.shape) > 2:
                deg_index_rot = (((pred_rot * 180.0 / np.pi) + 180.0) % self.num_angles).type(torch.long)
                best_score = fft_score[deg_index_rot, pred_txy[0], pred_txy[1]]
                if plotting and self.num_angles == 360 and plot_count % 10 == 0:
                    UtilityFuncs().plot_rotation_energysurface(fft_score, pred_txy, self.num_angles, stream_name,
                                                               plot_count)
            else:
                best_score = fft_score[pred_txy[0], pred_txy[1]]

        lowest_energy = -best_score

        return lowest_energy, pred_rot, pred_txy, fft_score


class SamplingModel(nn.Module):
    def __init__(self, dockingFFT, num_angles=1, step_size=10, sample_steps=10, sig_alpha=2, IP=False, IP_MC=False, IP_LD=False,
                 FI=False, experiment=None, debug=False):
        super(SamplingModel, self).__init__()
        self.debug = debug
        self.num_angles = num_angles

        self.docker = Docker(dockingFFT, num_angles=self.num_angles, debug=self.debug)

        self.sample_steps = sample_steps
        self.step_size = step_size
        self.plot_idx = 0

        self.experiment = experiment
        self.sig_alpha = sig_alpha
        self.step_size = self.sig_alpha
        self.BETA = torch.tensor(1.0)

        self.IP = IP
        self.IP_MC = IP_MC
        self.IP_LD = IP_LD

        self.FI = FI
        self.log_volume = torch.log(torch.tensor(100 ** 2))
        self.slice_free_energy = torch.logsumexp(torch.zeros(100,100), dim=(0,1))

    def forward(self, alpha, receptor, ligand, sig_alpha=None, plot_count=1, stream_name='trainset', plotting=False,
                training=True):
        if sig_alpha: ## for Langevin Dynamics
            self.sig_alpha = sig_alpha
            self.step_size = sig_alpha

        if self.IP:
            if training:
                ## BS model train giving the ground truth rotation
                lowest_energy, _, dr, FFT_score = self.docker(receptor, ligand, alpha,
                                                              plot_count=plot_count, stream_name=stream_name,
                                                              plotting=plotting)

                return lowest_energy, alpha.unsqueeze(0).clone(), dr.clone(), FFT_score
            else:
                ## BS model brute force eval
                alpha = 0
                self.docker.eval()
                lowest_energy, alpha, dr, FFT_score = self.docker(receptor, ligand, alpha, plot_count,
                                                                  stream_name, plotting=plotting)

                return lowest_energy, alpha.unsqueeze(0).clone(), dr.unsqueeze(0).clone(), FFT_score

        if self.IP_MC:
            if training:
                ## BS model train giving the ground truth rotation
                lowest_energy, _, dr, FFT_score = self.docker(receptor, ligand, alpha,
                                                              plot_count=plot_count, stream_name=stream_name,
                                                              plotting=plotting)

                return lowest_energy, alpha.unsqueeze(0).clone(), dr.clone(), FFT_score
            else:
                ## MC sampling eval
                self.docker.eval()
                return self.MCsampling(alpha, receptor, ligand, plot_count, stream_name, debug=False)

        if self.IP_LD:
            if training:
                ## train using Langevin dynamics
                lowest_energy, _, dr, FFT_score = self.docker(receptor, ligand, alpha,
                                                              plot_count=plot_count, stream_name=stream_name,
                                                              plotting=plotting)

                return lowest_energy, alpha.unsqueeze(0).clone(), dr.clone(), FFT_score
            else:
                ## Langegvin sampling eval
                self.docker.eval()
                return self.langevin_dynamics(alpha, receptor, ligand, plot_count, stream_name)

        if self.FI:
            if training:
                ## MC sampling for Fact of Interaction training
                return self.MCsampling(alpha, receptor, ligand, plot_count, stream_name, debug=False)
            else:
                ### evaluate with brute force
                self.docker.eval()
                lowest_energy, _, dr, FFT_score = self.docker(receptor, ligand, alpha, plot_count,
                                                              stream_name, plotting=plotting)
                return lowest_energy, alpha.unsqueeze(0).clone(), dr.unsqueeze(0).clone(), FFT_score
                ### evaluate with Monte Carlo?
                # self.docker.eval()
                # return self.MCsampling(alpha, receptor, ligand, plot_count, stream_name, debug=False)

    def MCsampling(self, alpha, receptor, ligand, plot_count, stream_name, debug=False):

        _, _, dr, FFT_score = self.docker(receptor, ligand, alpha,
                                          plot_count=plot_count, stream_name=stream_name,
                                          plotting=False)

        self.docker.eval()

        betaE = -self.BETA * FFT_score
        free_energy = -1 / self.BETA * (torch.logsumexp(-betaE, dim=(0, 1)) - self.log_volume)

        noise_alpha = torch.zeros_like(alpha)
        prob_list = []
        acceptance = []
        fft_score_list = []
        free_energies_encountered = torch.zeros(360)
        for i in range(self.sample_steps):
            if i == self.sample_steps - 1:
                plotting = True
            else:
                plotting = False
            rand_rot = noise_alpha.normal_(0, self.sig_alpha)
            alpha_new = alpha + rand_rot

            _, _, dr_new, FFT_score_new = self.docker(receptor, ligand, alpha_new,
                                                      plot_count=plot_count, stream_name=stream_name,
                                                      plotting=plotting)
            betaE_new = -self.BETA * FFT_score_new
            free_energy_new = -1 / self.BETA * (torch.logsumexp(-betaE_new, dim=(0, 1)) - self.log_volume)

            if free_energy_new <= free_energy:
                acceptance.append(1)
                prob_list.append(1)
                if debug:
                    print('accept <')
                    print('current', free_energy_new.item(), 'previous', free_energy.item(), 'alpha', alpha.item(),
                          'prev alpha', alpha.item())
                free_energy = free_energy_new
                alpha = alpha_new
                dr = dr_new
                FFT_score = FFT_score_new
                fft_score_list.append(FFT_score)
                deg_index_alpha = (((alpha * 180.0 / np.pi) + 180.0) % 360).type(torch.long)
                free_energies_encountered[deg_index_alpha] = free_energy
            else:
                prob = min(torch.exp(-self.BETA * (free_energy_new - free_energy)).item(), 1)
                rand0to1 = torch.rand(1).cuda()
                prob_list.append(prob)
                if prob > rand0to1:
                    acceptance.append(1)
                    if debug:
                        print('accept > and prob', prob, ' >', rand0to1.item())
                        print('current', free_energy_new.item(), 'previous', free_energy.item(), 'alpha', alpha.item(),
                              'prev alpha', alpha.item())
                    free_energy = free_energy_new
                    alpha = alpha_new
                    dr = dr_new
                    FFT_score = FFT_score_new
                    fft_score_list.append(FFT_score)
                    deg_index_alpha = (((alpha * 180.0 / np.pi) + 180.0) % 360).type(torch.long)
                    free_energies_encountered[deg_index_alpha] = free_energy
                else:
                    fft_score_list.append(FFT_score)
                    # if debug:
                    #     print('reject')
                    pass

        # print(alphas_encountered)

        self.docker.train()

        if debug:
            print('acceptance rate', acceptance)
            print(sum(acceptance) / self.sample_steps)

        if self.FI:
            fft_score_stack = torch.stack(fft_score_list)
        else:
            fft_score_stack = FFT_score
            free_energies_encountered = free_energy

        return free_energies_encountered, alpha.clone(), dr.clone(), fft_score_stack.squeeze()

    def langevin_dynamics(self, alpha, receptor, ligand, plot_count, stream_name):

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
            # self.sig_alpha = float(b * torch.exp(-(energy / a) ** n))
            # self.step_size = self.sig_alpha
            rand_rot = noise_alpha.normal_(0, self.sig_alpha)
            alpha = alpha + rand_rot

        return energy, alpha.clone(), dr.clone(), fft_score

    @staticmethod
    def check_gradients(model, param=None):
        for n, p in model.named_parameters():
            if param and param in str(n):
                print('Name', n, '\nParam', p, '\nGradient', p.grad)
                return
            if not param:
                print('Name', n, '\nParam', p, '\nGradient', p.grad)


if __name__ == "__main__":
    pass
