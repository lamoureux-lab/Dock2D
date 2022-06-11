import torch
from torch import nn


class Interaction(nn.Module):
    def __init__(self):
        """
        Initialize parameters for Interaction module and free energy integral calculation.
        The learned parameter for free energy decision threshold, :math:`F_0`, is initialized here.
        For the free energy integral, the volume used in the denominator is also initialized here.
        """
        super(Interaction, self).__init__()
        self.BF_num_angles = 360
        self.translation_volume = torch.tensor(100 ** 2)
        self.BF_log_volume = torch.log(self.BF_num_angles * self.translation_volume)

        self.F_0 = nn.Parameter(self.BF_log_volume, requires_grad=True)

        # self.F_0 = nn.Parameter(torch.zeros(1, requires_grad=True))
        self.F_0_prime = nn.Parameter(-torch.log(torch.tensor(100 ** 2)))

    def forward(self, brute_force=True, fft_scores=None, free_energies_visited=None):
        """
        Calculate the difference in free energy, :math:`\\Delta F`, using either a stack of `fft_scores` or sampled free energies.

            .. math::
                \Delta F = -\ln Z = -\ln \sum_{\mathbf{t}, \phi} e^{-E(\mathbf{t}, \phi)} - F_0

        then convert to a probability using a sigmoid function.

        :param brute_force: set to True to calculate the BruteForce free energy integral using `fft_scores` converted to energies.
        :param fft_scores: used in BruteForce free energy calculation
        :param free_energies: sampling method free energy array
        :return: `pred_interact`, `deltaF`, `F`, `F_0`
        """

        if brute_force:
            E = -fft_scores.squeeze()
            if len(E.shape) < 3:
                E = E.unsqueeze(0)
            F = -(torch.logsumexp(-E, dim=(0, 1, 2)) - self.BF_log_volume)
        else:
            visited_count = len(free_energies_visited[-1])
            unvisited_count = self.BF_num_angles - visited_count
            unvisited = self.F_0_prime * torch.ones(1, unvisited_count).cuda()
            free_energies_all = torch.cat((free_energies_visited, unvisited), dim=1)
            F = -(torch.logsumexp(-free_energies_all, dim=(0, 1)) - self.BF_log_volume)

            # visited_count = len(free_energies_visited[-1])
            # unvisited_count = self.BF_num_angles - visited_count
            # free_energies = torch.cat((free_energies_visited, torch.ones(1, unvisited_count).cuda()), dim=1)
            # F = -(torch.logsumexp(-free_energies, dim=(0, 1)) - self.BF_log_volume)

        deltaF = F - self.F_0
        pred_interact = torch.sigmoid(-deltaF)

        return pred_interact.squeeze(), deltaF.squeeze(), F, self.F_0


if __name__ == '__main__':
    print('works')
    print(Interaction())
    print(list(Interaction().parameters()))
