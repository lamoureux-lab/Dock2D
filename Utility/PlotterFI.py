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
        self.log_saturation_prefix = 'log_MCFI_saturation_stats'

        if not experiment:
            print('no experiment name given')
            self.experiment = "NOTSET"

    def plot_loss(self, show=False, save=True, plot_combined=False):
        """
        Plot the current fact-of-interaction (FI) experiments loss curve.
        The plot will plot all epochs present in the log file.

        :param show: show the plot in a window
        :param save: save the plot at specified path
        """
        plt.close()
        #LOSS WITH ROTATION
        train = pd.read_csv(self.logfile_savepath+self.logloss_prefix+ self.experiment +'.txt', sep='\t', header=0, names=['Epoch', 'Loss'])

        plt.subplots(figsize=(20,10))

        train_loss, = plt.plot(train['Epoch'].to_numpy(), train['Loss'].to_numpy())
        plt.title('log_loss_TRAINset_'+ self.experiment)
        plt.xlabel('epochs')
        plt.ylabel('loss')
        # if len(train['Loss'].to_numpy()) > 2:
        #     plt.ylim([0, train['Loss'].to_numpy().max()])
        plt.grid(visible=True)
        plt.xlabel('Epochs')

        if save:
            plt.savefig('Figs/FI_loss_plots/lossplot_'+self.experiment+'.png')
        if show and not plot_combined:
            plt.show()
        return train_loss

    def plot_deltaF_distribution(self, filename=None, plot_epoch=None, xlim=250, binwidth=1, show=False, save=True, plot_pub=False):
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

        plt.figure(figsize=(8,6))
        if not plot_pub:
            plt.title('deltaF distribution: epoch'+ str(plot_epoch) + ' ' + self.experiment)
            format = 'png'
        else:
            format = 'pdf'

        labels = sorted(dataframe.Label.unique())
        F = dataframe['F']
        bins = np.arange(F.min(), F.max() + binwidth, binwidth)
        hist_data = [dataframe.loc[dataframe.Label == x, 'F'] for x in labels]
        # print('labels', labels)
        # print(F.shape)

        if len(labels) > 1:
            y1, x1, _ = plt.hist(hist_data[0], label=labels, bins=bins, rwidth=binwidth, color=['r'], alpha=0.25)
            y2, x2, _ = plt.hist(hist_data[1], label=labels, bins=bins, rwidth=binwidth, color=['g'], alpha=0.25)
            ymax = max(max(y1), max(y2)) + 1
        else:
            # print(hist_data)
            if labels == [1]:
                color = 'g'
                plt.legend(('interaction (+)'), prop={'size': 10})
            else:
                color = 'r'
                plt.legend(('non-interaction (-)'), prop={'size': 10})

            y1, x1, _ = plt.hist(hist_data, label=labels, bins=bins, color=color, rwidth=binwidth, alpha=0.25)
            ymax = y1.max() + 1

        plt.vlines(dataframe['F_0'].to_numpy()[-1], ymin=0, ymax=ymax, linestyles='dashed', label='F_0', colors='k')
        plt.legend(('non-interaction (-)', ' interaction (+)', 'final F_0'), prop={'size': 10}, loc='upper left')

        if xlim:
            plt.xlim([-xlim, 0])
        plt.ylabel('counts')
        plt.xlabel('free energy (F)')
        plt.grid(visible=False)
        plt.margins(x=None)

        if save:
            plt.savefig('Figs/FI_deltaF_distribution_plots/deltaFplot_epoch'+ str(plot_epoch) + '_' + self.experiment + '.'+format, format=format)
        if show:
            plt.show()

    def plot_MCFI_saturation(self, filename=None, plot_epoch=None, show=False, save=True, plot_pub=False):

        plt.close()

        # Plot free energy distribution of all samples across epoch
        if not filename:
            filename = self.logfile_savepath+self.log_saturation_prefix+str(plot_epoch)+ self.experiment +'.csv'
        dataframe = pd.read_csv(filename)

        saturations = []
        for col in dataframe:
            cur_dist = dataframe[col]
            cur_saturation = cur_dist.count()#/360
            print('saturation', cur_saturation)
            saturations.append(cur_saturation)

        xlimit = 180
        binwidth = 1
        bins = np.arange(0, xlimit + binwidth, binwidth)

        x, y, _ = plt.hist(saturations, bins=bins)
        # plt.title('Saturation of rotation MC FI sampling')
        # plt.title(self.experiment)

        mean_saturation = np.array(saturations).mean()
        std_dev_saturation = np.array(saturations).std()

        plt.vlines(mean_saturation, ymin=0, ymax=y.max()/2*binwidth, colors='k', linestyles='dashed')
        plt.xlim([0, xlimit])
        plt.xlabel('unique rotations visited')
        plt.margins(x=None, y=None)
        plt.legend([r'$\mathcal{\mu}$ = '+str(int(mean_saturation)),
                    r'$\mathcal{\sigma}$ = ' + str(int(std_dev_saturation)),
                    ])
        plt.grid(False)


        # plt.close()
        # for array in dataframe.hist(bins=bins):
        #     for subplot in array:
        #         # subplot.axis('off')
        #         subplot.set_xlim([0, 360])
        #         subplot.set_xticks([0, 360])
        #         # subplot.set_ylim([0, 10])
        #
        # plt.show()

        if not plot_pub:
            plt.title('MCFI sampling saturation distribution: epoch'+ str(plot_epoch) + ' ' + self.experiment)
            format = 'png'
        else:
            format = 'pdf'

        if save:
            plt.savefig('Figs/EnergySurfaces/saturationMCFI_epoch'+ str(plot_epoch) + '_' + self.experiment + '.'+format, format=format)
        if show:
            plt.show()

# if __name__ == "__main__":
#     ### Unit test
#     load_path = 'Log/losses/FI_loss/'
#     testcase = 'scratch_'
#     experiment = testcase+'BF_FI_NEWDATA_CHECK_400pool_20000ex30ep'
#     Plotter = PlotterFI(experiment, logfile_savepath=load_path)
#     Plotter.plot_loss(show=True)
#     Plotter.plot_deltaF_distribution(plot_epoch=30, show=True)

if __name__ == "__main__":
    loadpath = 'Log/losses/FI_loss/'
    # experiment = 'BF_IP_NEWDATA_CHECK_400pool_30ep'
    experiments_list = [
                       'scratch_BF_FI_finaldataset_100pairs_500ep',
                       'C_BF_FI_finaldataset_100pairs_expC_BFIP_1000ex100ep',
                       'C_BF_FI_finaldataset_100pairs_expC_BSIP_1000ex100ep',
                       'scratch_MC_FI_finaldataset_100pairs_1000ep_2sample_100steps',
                       ]
    fig_data = []
    for experiment in experiments_list:
        Plotter = PlotterFI(experiment, logfile_savepath=loadpath)
        line_loss = Plotter.plot_loss(show=True, plot_combined=True)
        # Plotter.plot_rmsd_distribution(plot_epoch=100, show=True)
        fig_data.append(line_loss)
        plt.close()

    # plt.close()
    plt.figure(figsize=(10,5))
    color_style = ['b-', 'b--', 'r-', 'r--']

    for i in range(len(fig_data)):
        plt.plot(fig_data[i].get_data()[0], fig_data[i].get_data()[1], color_style[i], )

    plt.margins(x=None)
    plt.legend(['BruteForce IF 1000pairs', 'BruteForce IP pretrain', 'BruteSimplified IP pretrained', 'MCFI 100pairs'])
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.xlim([0, 1000])
    plt.ylim([0, 1])
    plt.savefig('Figs/FI_loss_plots/sup_combined_FI_loss_plot_1000epoch.pdf', format='pdf')
    plt.show()
