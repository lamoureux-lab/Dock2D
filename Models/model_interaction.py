import torch
from torch import nn


class Interaction(nn.Module):
    def __init__(self):
        super(Interaction, self).__init__()
        self.F_0 = nn.Parameter(torch.zeros(1, requires_grad=True))
        self.BF_log_volume = torch.log(360 * torch.tensor(100 ** 2))

    def forward(self, brute_force=True, fft_scores=None, free_energies=None, debug=False):
        ##TODO: pass BETA, has to be returned from MC docker

        #  include only unique angles, remove redundant visits
        #  include grid of alphas with sample E and unsampled as 0

        if brute_force:
            E = -fft_scores.squeeze()
            if len(E.shape) < 3:
                E = E.unsqueeze(0)
            F = -(torch.logsumexp(-E, dim=(0, 1, 2)) - self.BF_log_volume)
        else:
            # F = torch.sum(free_energies)
            F = -(torch.logsumexp(-free_energies, dim=0) - self.BF_log_volume)




        # if E.shape[0] > 1:
        #     self.log_slice_volume = torch.log(E.shape[0]*torch.tensor(100 ** 2))

        # if E.shape[0] == 360:
        #     F = -(torch.logsumexp(-E, dim=(0, 1, 2)) - self.log_slice_volume)
        # else:
        #     translationsF = torch.logsumexp(-E, dim=(1, 2))
        #     F = -(translationsF - self.log_slice_volume)
        #     F = torch.mean(F, dim=0)

        # E_adjusted = 1

        # print('unique E values', E.shape[0])
        # F = -(torch.logsumexp(-E, dim=(0, 1, 2)))

        deltaF = F - self.F_0
        pred_interact = torch.sigmoid(-deltaF)

        if debug:
            with torch.no_grad():
                print('\n(F - F_0): ', deltaF.item())
                print('F_0: ', self.F_0.item())

        return pred_interact.squeeze(), deltaF.squeeze(), F, self.F_0


if __name__ == '__main__':
    print('works')
    print(Interaction())
    print(list(Interaction().parameters()))
