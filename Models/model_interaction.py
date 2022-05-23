import torch
from torch import nn


class Interaction(nn.Module):
    def __init__(self):
        super(Interaction, self).__init__()
        self.F_0 = nn.Parameter(torch.zeros(1, requires_grad=True))
        self.num_angles = 360
        self.BF_log_volume = torch.log(self.num_angles * torch.tensor(100 ** 2))

    def forward(self, brute_force=True, fft_scores=None, free_energies=None):
        ##TODO: pass BETA, has to be returned from MC docker

        if brute_force:
            E = -fft_scores.squeeze()
            if len(E.shape) < 3:
                E = E.unsqueeze(0)
            F = -(torch.logsumexp(-E, dim=(0, 1, 2)) - self.BF_log_volume)
        else:
            num_angles_visited = len(free_energies[-1])
            unvisited_count = self.num_angles-num_angles_visited
            free_energies = torch.cat((free_energies, torch.ones(1, unvisited_count).cuda()), dim=1)
            F = -(torch.logsumexp(-free_energies, dim=(0, 1)) - self.BF_log_volume)

        deltaF = F - self.F_0
        pred_interact = torch.sigmoid(-deltaF)

        return pred_interact.squeeze(), deltaF.squeeze(), F, self.F_0


if __name__ == '__main__':
    print('works')
    print(Interaction())
    print(list(Interaction().parameters()))
