import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class PlotterFI:
    def __init__(self, experiment=None, logfile_savepath='Log/losses/FI_loss/'):
        """
        Initialize paths and filename prefixes for saving plots.

        :param experiment: current experiment name
        :param logfile_savepath: path to load and save data and figs
        """
        self.experiment = experiment
        self.logfile_savepath = logfile_savepath
        self.logtraindF_prefix = 'log_deltaF_TRAINset_epoch'
        self.logloss_prefix = 'log_loss_TRAINset_'
        if not experiment:
            print('no experiment name given')
            self.experiment = "NOTSET"

    def plot_loss(self, show=False, save=True):
        """
        Plot the current fact-of-interaction (FI) experiments loss curve.
        The plot will plot all epochs present in the log file.

        :param show: show the plot in a window
        :param save: save the plot at specified path
        """
        plt.close()
        #LOSS WITH ROTATION
        train = pd.read_csv(self.logfile_savepath+self.logloss_prefix+ self.experiment +'.txt', sep='\t', header=1, names=['Epoch', 'Loss'])

        plt.subplots(figsize=(20,10))

        train_loss = plt.plot(train['Epoch'].to_numpy(), train['Loss'].to_numpy())
        plt.title('log_loss_TRAINset_'+ self.experiment)
        plt.xlabel('epochs')
        plt.ylabel('loss')
        if len(train['Loss'].to_numpy()) > 2:
            plt.ylim([0, train['Loss'].to_numpy().max()])
        plt.grid(visible=True)
        # num_epochs = len(train['Epoch'].to_numpy())
        # plt.xticks(np.arange(0, num_epochs+1, num_epochs//10))
        plt.xlabel('Epochs')

        if save:
            plt.savefig('Figs/FI_loss_plots/lossplot_'+self.experiment+'.png')
        if show:
            plt.show()

    def plot_deltaF_distribution(self, filename=None, plot_epoch=None, xlim=None, binwidth=1, show=False, save=True):
        """
        Plot the labeled free energies of interacting and non-interacting shape pairs as a histogram,
        with a vertical line demarcating the learned `F_0` interaction decision threshold, if applicable.

        :param filename: specify the file to load, default of `None` sets filename to match the project convention
        :param plot_epoch: epoch of training/evalution to plot
        :param xlim: absolute value lower limit of the x-axis.
        :param binwidth: histogram bin width
        :param show: show the plot in a window
        :param save: save the plot at specified path
        :return:
        """
        plt.close()
        # Plot free energy distribution of all samples across epoch
        if not filename:
            filename = self.logfile_savepath+self.logtraindF_prefix+str(plot_epoch)+ self.experiment +'.txt'
        dataframe = pd.read_csv(filename, sep='\t', header=0, names=['F', 'F_0', 'Label'])

        fig, ax = plt.subplots(figsize=(10,10))
        plt.suptitle('deltaF distribution: epoch'+ str(plot_epoch) + ' ' + self.experiment)

        labels = sorted(dataframe.Label.unique())
        F = dataframe['F']
        bins = np.arange(F.min(), F.max() + binwidth, binwidth)
        hist_data = [dataframe.loc[dataframe.Label == x, 'F'] for x in labels]
        y1, x1, _ = plt.hist(hist_data[0], label=labels, bins=bins, rwidth=binwidth, color=['r'], alpha=0.25)
        y2, x2, _ = plt.hist(hist_data[1], label=labels, bins=bins, rwidth=binwidth, color=['g'], alpha=0.25)

        if dataframe['F_0'][0] != 'NA':
            plt.vlines(dataframe['F_0'].to_numpy()[-1], ymin=0, ymax=max(max(y1), max(y2))+1, linestyles='dashed', label='F_0', colors='k')
            plt.legend(('non-interaction (-)', ' interaction (+)', 'final F_0'), prop={'size': 10})
        else:
            plt.legend(('non-interaction (-)', ' interaction (+)'), prop={'size': 10})

        if xlim:
            ax.set_xlim([-xlim, 0])
        ax.set_ylabel('Training set counts')
        ax.set_xlabel('Free Energy (F)')
        ax.grid(visible=True)

        if save:
            plt.savefig('Figs/FI_deltaF_distribution_plots/deltaFplot_epoch'+ str(plot_epoch) + '_' + self.experiment + '.png', format='png')
        if show:
            plt.show()


if __name__ == "__main__":
    ### Unit test
    load_path = 'Log/losses/FI_loss/'
    testcase = 'scratch_'
    experiment = testcase+'BF_FI_NEWDATA_CHECK_400pool_20000ex30ep'
    Plotter = PlotterFI(experiment, logfile_savepath=load_path)
    Plotter.plot_loss(show=True)
    Plotter.plot_deltaF_distribution(plot_epoch=30, show=True)
