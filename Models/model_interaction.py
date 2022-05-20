import torch
from torch import nn


class Interaction(nn.Module):
    def __init__(self):
        super(Interaction, self).__init__()
        self.F_0 = nn.Parameter(torch.zeros(1, requires_grad=True))
        self.BF_log_volume = torch.log(360 * torch.tensor(100 ** 2))

    def forward(self, brute_force=True, fft_scores=None, free_energies=None):
        ##TODO: pass BETA, has to be returned from MC docker

        if brute_force:
            E = -fft_scores.squeeze()
            if len(E.shape) < 3:
                E = E.unsqueeze(0)
            F = -(torch.logsumexp(-E, dim=(0, 1, 2)) - self.BF_log_volume)
        else:
            num_slices = len(free_energies[-1])
            if num_slices > 0:
                log_volume = torch.log(num_slices * torch.tensor(100 ** 2))
                F = -(torch.logsumexp(-free_energies, dim=(0, 1)) - log_volume)
            else:
                F = torch.ones(1).cuda()

        deltaF = F - self.F_0
        pred_interact = torch.sigmoid(-deltaF)

        return pred_interact.squeeze(), deltaF.squeeze(), F, self.F_0


if __name__ == '__main__':
    print('works')
    print(Interaction())
    print(list(Interaction().parameters()))
