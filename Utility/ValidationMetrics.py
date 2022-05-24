import torch
import numpy as np
from tqdm import tqdm
from Dock2D.Utility.PlotterFI import PlotterFI


class RMSD:
    def __init__(self, ligand, gt_rot, gt_txy, pred_rot, pred_txy):
        self.bulk = np.array(ligand.detach().cpu())
        self.size = self.bulk.shape[-1]
        self.gt_rot = gt_rot
        self.gt_txy = gt_txy
        self.pr_rot = pred_rot
        self.pr_txy = pred_txy
        self.epsilon = 1e-5

    def get_XC(self):
        r"""
        Analog of inertia tensor and center of mass for rmsd calc.
        Where `X` is the inertia tensor, `W` is the sum of the bulk shape, and `C` is the center of mass.

        :return:
            .. math::
                \frac{2*X}{W}, \frac{C}{W}
        """
        X = torch.zeros(2, 2)
        C = torch.zeros(2)
        x_i = (torch.arange(self.size).unsqueeze(dim=0) - self.size / 2.0).repeat(self.size, 1)
        y_i = (torch.arange(self.size).unsqueeze(dim=1) - self.size / 2.0).repeat(1, self.size)
        mask = torch.from_numpy(self.bulk > 0.5)
        W = torch.sum(mask.to(dtype=torch.float32))
        x_i = x_i.masked_select(mask)
        y_i = y_i.masked_select(mask)
        # Inertia tensor
        X[0, 0] = torch.sum(x_i * x_i)
        X[1, 1] = torch.sum(y_i * y_i)
        X[0, 1] = torch.sum(x_i * y_i)
        X[1, 0] = torch.sum(y_i * x_i)
        # Center of mass
        C[0] = torch.sum(x_i)
        C[1] = torch.sum(x_i)
        return 2.0 * X / (W + self.epsilon), C / (W + self.epsilon)

    def calc_rmsd(self):
        """
        :return: RMSD of predicted versus ground truth shapes.
        """
        rotation1, translation1, rotation2, translation2 = self.gt_rot, self.gt_txy, self.pr_rot, self.pr_txy
        X, C = self.get_XC()
        X = X.type(torch.float).cuda()
        C = C.type(torch.float).cuda()

        T1 = translation1.clone().detach().cuda()
        T2 = translation2.clone().detach().cuda()
        T = T1 - T2

        rotation1 = torch.tensor([rotation1], dtype=torch.float64).cuda()
        rotation2 = torch.tensor([rotation2], dtype=torch.float64).cuda()
        R1 = torch.zeros(2, 2, dtype=torch.float64).cuda()
        R1[0, 0] = torch.cos(rotation1)
        R1[1, 1] = torch.cos(rotation1)
        R1[1, 0] = torch.sin(rotation1)
        R1[0, 1] = -torch.sin(rotation1)
        R2 = torch.zeros(2, 2, dtype=torch.float64).cuda()
        R2[0, 0] = torch.cos(rotation2)
        R2[1, 1] = torch.cos(rotation2)
        R2[1, 0] = torch.sin(rotation2)
        R2[0, 1] = -torch.sin(rotation2)
        R = R2.transpose(0, 1) @ R1

        I = torch.diag(torch.ones(2, dtype=torch.float64)).cuda()
        # RMSD
        rmsd = torch.sum(T * T)
        rmsd = rmsd + torch.sum((I - R) * X, dim=(0, 1))
        rmsd = rmsd + 2.0 * torch.sum(torch.sum(T.unsqueeze(dim=1) * (R1 - R2), dim=0) * C, dim=0) + self.epsilon
        # print(rmsd)

        # sqrt_rmsd = torch.sqrt(rmsd)
        sqrt_rmsd = torch.sqrt(rmsd) - torch.sqrt(torch.tensor(self.epsilon))

        return sqrt_rmsd


class APR:
    def __init__(self):
        """
        Initialize epsilon to avoid division by ~zero.
        """
        self.epsilon = 1e-5

    def calc_APR(self, data_stream, run_model, epoch=0, deltaF_logfile=None, experiment=None):
        """
        Calculate accuracy, precision, recall, F1-score, and MCC (Matthews Correlation Coefficient)
        based on confusion matrix values during fact-of-interaction model evaluation.

        :param data_stream: data stream
        :param run_model: run_model() passed from FI model
        :param epoch: current evaluation epoch
        :param deltaF_logfile: free energies per example and F_0 logfile
        :param experiment: current experiment name
        :return: accuracy, precision, recall, F1score, MCC
        """
        print('Calculating Accuracy, Precision, Recall')
        TP, FP, TN, FN = 0, 0, 0, 0

        for data in tqdm(data_stream):
            tp, fp, tn, fn, F, F_0, label = run_model(data, training=False)
            # print(tp, fp, tn,fn)
            TP += tp
            FP += fp
            TN += tn
            FN += fn
            with open(deltaF_logfile, 'a') as fout:
                fout.write('%f\t%f\t%d\n' % (F, F_0, label))

        PlotterFI(experiment).plot_deltaF_distribution(plot_epoch=epoch, show=True, xlim=None, binwidth=1)


        Accuracy = float(TP + TN) / float(TP + TN + FP + FN)
        if (TP + FP) > 0:
            Precision = float(TP) / float(TP + FP)
        else:
            Precision = 0.0
        if (TP + FN) > 0:
            Recall = float(TP) / float(TP + FN)
        else:
            Recall = 0.0
        F1score = TP / (TP + 0.5*(FP + FN)+1E-5)

        MCC = ((TP * TN) - (FP * FN)) / (np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)) + self.epsilon)

        print(f'Epoch {epoch} Acc: {Accuracy} Prec: {Precision} Rec: {Recall} F1: {F1score} MCC: {MCC}')

        return Accuracy, Precision, Recall, F1score, MCC


if __name__ == '__main__':
    pass
