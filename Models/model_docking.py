import torch
from torch import nn

from Dock2D.Utility.TorchDockingFFT import TorchDockingFFT
from Dock2D.Utility.UtilityFunctions import UtilityFunctions
from e2cnn import nn as enn
from e2cnn import gspaces


class Docking(nn.Module):

    def __init__(self, dockingFFT, plot_freq=10, debug=False):
        """
        Initialize parameters for Docking module and TorchDockingFFT used in the forward pass
        to generate shape features and scoring coefficients for the scoring function.

        :param dim: dimension of final padded shape
        :param num_angles: number of angles to use in FFT correlation
        :param plot_freq: frequency at which to plot features
        :param debug: set to True show debug verbose model
        """
        super(Docking, self).__init__()
        self.debug = debug
        self.plot_freq = plot_freq
        self.boundW = nn.Parameter(torch.ones(1, requires_grad=True))
        self.crosstermW1 = nn.Parameter(torch.ones(1, requires_grad=True))
        self.crosstermW2 = nn.Parameter(torch.ones(1, requires_grad=True))
        self.bulkW = nn.Parameter(torch.ones(1, requires_grad=True))

        self.dockingFFT = dockingFFT

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
        """
        Generates features for both receptor and ligand shapes using the SE(2)-ConvNet.
        These features are then scored based on rotation and translationally sampled FFT correlation,

            .. math::
                \mathrm{corr}(\mathbf{t}, \phi, R, L) = \int R(\mathbf{r}) \mathbf{M}_\phi L(\mathbf{r}-\mathbf{t}) d\mathbf{r}

        on pairwise features, i.e. boundary:boundary, bulk:boundary, boundary:bulk, bulk:bulk.

        :param receptor: receptor shape grid image
        :param ligand: ligand shape grid image
        :param training: set `True` for training, `False` for evalution.
        :param plotting: create plots or not
        :param plot_count: current plotting index
        :param stream_name: data stream name
        :param angle: single angle to rotate shape for single rotation slice FFT in sampling models
        :return: `fft_score`
        """
        receptor_geomT = enn.GeometricTensor(receptor.unsqueeze(0), self.feat_type_in1)
        ligand_geomT = enn.GeometricTensor(ligand.unsqueeze(0), self.feat_type_in1)

        rec_feat = self.netSE2(receptor_geomT).tensor.squeeze()
        lig_feat = self.netSE2(ligand_geomT).tensor.squeeze()

        fft_score = self.dockingFFT.dock_rotations(
            rec_feat,
            lig_feat,
            angle,
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
