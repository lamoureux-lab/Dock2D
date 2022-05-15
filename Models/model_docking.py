import torch
from torch import nn

from Dock2D.Utility.TorchDockingFFT import TorchDockingFFT
from Dock2D.Utility.UtilityFunctions import UtilityFunctions
from e2cnn import nn as enn
from e2cnn import gspaces


class Docking(nn.Module):

    def __init__(self, dim=100, num_angles=360, plot_freq=10, debug=False):
        super(Docking, self).__init__()
        self.debug = debug
        self.plot_freq = plot_freq
        self.dim = dim
        self.num_angles = num_angles
        self.boundW = nn.Parameter(torch.ones(1, requires_grad=True))
        self.crosstermW1 = nn.Parameter(torch.ones(1, requires_grad=True))
        self.crosstermW2 = nn.Parameter(torch.ones(1, requires_grad=True))
        self.bulkW = nn.Parameter(torch.ones(1, requires_grad=True))

        self.scal = 1
        self.vec = 4

        self.SO2 = gspaces.Rot2dOnR2(N=-1, maximum_frequency=4)
        self.feat_type_in1 = enn.FieldType(self.SO2, 1 * [self.SO2.trivial_repr])
        self.feat_type_out1 = enn.FieldType(self.SO2, self.scal * [self.SO2.irreps['irrep_0']] + self.vec * [self.SO2.irreps['irrep_1']])
        self.feat_type_out_final = enn.FieldType(self.SO2, 1 * [self.SO2.irreps['irrep_0']] + 1 * [self.SO2.irreps['irrep_1']])

        self.kernel = 5
        self.pad = self.kernel//2
        self.stride = 1
        self.dilation = 1

        self.netSE2 = enn.SequentialModule(
            enn.R2Conv(self.feat_type_in1, self.feat_type_out1, kernel_size=self.kernel, stride=self.stride, dilation=self.dilation, padding=self.pad, bias=False),
            enn.NormNonLinearity(self.feat_type_out1, function='n_relu', bias=False),
            enn.R2Conv(self.feat_type_out1, self.feat_type_out_final, kernel_size=self.kernel, stride=self.stride, dilation=self.dilation, padding=self.pad, bias=False),
            enn.NormNonLinearity(self.feat_type_out_final, function='n_relu', bias=False),
            enn.NormPool(self.feat_type_out_final),
        )

    def forward(self, receptor, ligand, training=True, plotting=False, plot_count=1, stream_name='trainset', angle=None):

        receptor_geomT = enn.GeometricTensor(receptor.unsqueeze(0), self.feat_type_in1)
        ligand_geomT = enn.GeometricTensor(ligand.unsqueeze(0), self.feat_type_in1)

        rec_feat = self.netSE2(receptor_geomT).tensor.squeeze()
        lig_feat = self.netSE2(ligand_geomT).tensor.squeeze()

        fft_score = TorchDockingFFT(dim=self.dim, num_angles=self.num_angles, angle=angle, debug=self.debug).dock_global(
            rec_feat,
            lig_feat,
            weight_bound=self.boundW,
            weight_crossterm1=self.crosstermW1,
            weight_crossterm2=self.crosstermW2,
            weight_bulk=self.bulkW
        )

        #### Plot shape features
        if plotting and not training:
            if plot_count % self.plot_freq == 0:
                with torch.no_grad():
                    scoring_weights = (self.boundW, self.crosstermW1, self.crosstermW2, self.bulkW)
                    UtilityFunctions().plot_features(rec_feat, lig_feat, receptor, ligand, scoring_weights, plot_count, stream_name)

        return fft_score

if __name__ == '__main__':
    print('works')
    print(Docking())
    print(list(Docking().parameters()))
