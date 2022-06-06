import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from os.path import exists


class PlotterIP:
    def __init__(self, experiment=None, logfile_savepath='Log/losses/IP_loss/'):
        """
        Initialize paths and filename prefixes for saving plots.

        :param experiment: current experiment name
        :param logfile_savepath: path to load and save data and figs
        """
        self.experiment = experiment
        self.logfile_savepath = logfile_savepath

        if not experiment:
            print('no experiment name given')
            sys.exit()

    def plot_loss(self, ylim=None, show=False, save=True):
        """
        Plot the current interaction pose (IP) experiments loss curve.
        The plot will plot all epochs present in the log file.

        :param ylim: set the upper limit of the y-axis, initial IP loss can vary widely
        :param show: show the plot in a window
        :param save: save the plot at specified path
        """
        plt.close()
        #LOSS WITH ROTATION
        train = pd.read_csv(self.logfile_savepath+'log_loss_TRAINset_'+ self.experiment +'.txt', sep='\t', header=1, names=['Epoch', 'Loss', 'RMSD'])
        valid = pd.read_csv(self.logfile_savepath+'log_loss_VALIDset_'+ self.experiment +'.txt', sep='\t', header=1, names=['Epoch', 'Loss', 'RMSD'])
        test = pd.read_csv(self.logfile_savepath+'log_loss_TESTset_'+ self.experiment +'.txt', sep='\t', header=1, names=['Epoch', 'Loss', 'RMSD'])

        fig, ax = plt.subplots(2, figsize=(20,10))
        ax[0].plot(train['Epoch'].to_numpy(), train['RMSD'].to_numpy())
        ax[0].plot(valid['Epoch'].to_numpy(), valid['RMSD'].to_numpy())
        ax[0].plot(test['Epoch'].to_numpy(), test['RMSD'].to_numpy())
        ax[0].legend(('train RMSD', 'valid RMSD', 'test RMSD'))

        ax[0].set_title('Loss: ' + self.experiment)
        ax[0].set_ylabel('RMSD')
        ax[0].grid(visible=True)

        ax[1].plot(train['Epoch'].to_numpy(), train['Loss'].to_numpy())
        ax[1].plot(valid['Epoch'].to_numpy(), valid['Loss'].to_numpy())
        ax[1].plot(test['Epoch'].to_numpy(), test['Loss'].to_numpy())
        ax[1].legend(('train loss', 'valid loss', 'test loss'))

        ax[1].set_xlabel('epochs')
        ax[1].set_ylabel('loss')
        ax[1].grid(visible=True)

        # num_epochs = len(train['Epoch'].to_numpy())
        # ax[0].set_xticks(np.arange(0, num_epochs+1, num_epochs/10))
        # ax[1].set_xticks(np.arange(0, num_epochs+1, num_epochs/10))

        plt.xlabel('Epochs')
        if ylim:
            ax[0].set_ylim([0,ylim])
            ax[1].set_ylim([0,ylim])

        if save:
            plt.savefig('Figs/IP_loss_plots/lossplot_'+self.experiment+'.png')
        if show:
            plt.show()

    def plot_rmsd_distribution(self, plot_epoch=1, show=False, save=True):
        """
        Plot the predicted RMSDs distributions as histogram(s), depending on how many log files exist.

        :param plot_epoch: epoch of training/evalution to plot
        :param show: show the plot in a window
        :param save: save the plot at specified path
        """
        plt.close()
        # Plot RMSD distribution of all samples across epoch
        train_log = self.logfile_savepath+'log_RMSDsTRAINset_epoch' + str(plot_epoch) + self.experiment + ".txt"
        valid_log = self.logfile_savepath+'log_RMSDsVALIDset_epoch' + str(plot_epoch) + self.experiment + ".txt"
        test_log = self.logfile_savepath+'log_RMSDsTESTset_epoch' + str(plot_epoch) + self.experiment + ".txt"
        train, valid, test, avg_trainRMSD, avg_validRMSD, avg_testRMSD = None, None, None, None, None, None
        subplot_count = 0

        print('average RMSDs:')
        if exists(train_log):
            subplot_count += 1
            train = pd.read_csv(train_log, sep='\t', header=0, names=['RMSD'])
            avg_trainRMSD = str(train['RMSD'].mean())[:4]
            print('train:', avg_trainRMSD)

        if exists(valid_log):
            subplot_count += 1
            valid = pd.read_csv(valid_log, sep='\t', header=0, names=['RMSD'])
            avg_validRMSD = str(valid['RMSD'].mean())[:4]
            print('valid:', avg_validRMSD)

        if exists(test_log):
            subplot_count += 1
            test = pd.read_csv(test_log, sep='\t', header=0, names=['RMSD'])
            avg_testRMSD = str(test['RMSD'].mean())[:4]
            print('test:', avg_testRMSD)

        fig, ax = plt.subplots(3, figsize=(20, 10))
        plt.suptitle('RMSD distribution: epoch' + str(plot_epoch) + ' ' + self.experiment +'\n'
                      + 'train:'+ avg_trainRMSD + ' valid:' + avg_validRMSD + ' test:' + avg_testRMSD)
        plt.legend(['train rmsd', 'valid rmsd', 'test rmsd'])
        plt.xlabel('RMSD')
        binwidth=1
        bins = np.arange(0, 100 + binwidth, binwidth)

        if train is not None:
            ax[0].hist(train['RMSD'].to_numpy(), bins=bins, color='b')
            ax[0].set_ylabel('Training set counts')
            ax[0].grid(visible=True)
            ax[0].set_xticks(np.arange(0, 100, 10))
        if valid is not None:
            ax[1].hist(valid['RMSD'].to_numpy(), bins=bins, color='r')
            ax[1].set_ylabel('Valid set counts')
            ax[1].grid(visible=True)
            ax[1].set_xticks(np.arange(0, 100, 10))
        if test is not None:
            ax[2].hist(test['RMSD'].to_numpy(), bins=bins, color='g')
            ax[2].set_ylabel('Test set counts')
            ax[2].grid(visible=True)
            ax[2].set_xticks(np.arange(0, 100, 10))

        if save:
            plt.savefig('Figs/IP_RMSD_distribution_plots/RMSDplot_epoch' + str(
                plot_epoch) + self.experiment + '.png')
        if show:
            plt.show()


if __name__ == "__main__":
    loadpath = 'Log/losses/IP_loss/'
    experiment = 'BF_IP_NEWDATA_CHECK_400pool_30ep'
    Plotter = PlotterIP(experiment, logfile_savepath=loadpath)
    Plotter.plot_loss(show=True)
    Plotter.plot_rmsd_distribution(plot_epoch=30, show=True)
