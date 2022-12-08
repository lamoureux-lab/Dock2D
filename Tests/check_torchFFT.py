import torch
import matplotlib.pyplot as plt
import numpy as np


def gaussian_1d(M, mean, sigma,
                a=1, gaussian_norm=False):
    """
    Create 1D gaussian vector

    :param M: 1D array length
    :param mean: set the mean
    :param sigma: set the standard deviation
    :param a: amplitude of distribution
    :param gaussian_norm: if `True` 1D distrubtution sums to 1
    :return: 1D gaussian vector
    """
    if gaussian_norm:
        a = 1 / (sigma * np.sqrt(2 * np.pi))

    x = torch.arange(0, M) - (M / 2)
    var = 2 * sigma ** 2
    w = a * torch.exp(-((x - mean) ** 2 / var))
    return w


def gaussian_2d(kernel_len=50, mean=0, sigma=5, a=1, gaussian_norm=False):
    """
    Use the outer product of two gaussian vectors to create 2D gaussian

    :param kernel_len: dimension for 2D square kernel
    :param mean: center of gaussian
    :param sigma: spread around the center of gaussian
    :param a: height of gaussian
    :param gaussian_norm: individual 1d gaussians sum to 1
    :return: 2D gaussian kernel
    """

    gaussian_kernel1d = gaussian_1d(kernel_len, mean=mean, sigma=sigma, a=a, gaussian_norm=gaussian_norm)
    return torch.outer(gaussian_kernel1d, gaussian_kernel1d)


def swap_fft_plot_quadrants(input_volume):
    """
    Swap quadrants of 2d FFT output. The FFT operation moves the output distribution mean with the following:

      Correlation: ~N(mean1 - mean2, sigma1^2 + sigma2^2)

      Convolution: ~N(mean1 + mean2, sigma1^2 + sigma2^2)

    Therefore, distributions with means at the center will get p

    :param input_volume: FFT output 2D array with swapped quadrants
    """
    num_features = input_volume.size(0)
    L = input_volume.size(-1)
    L2 = int(L / 2)
    output_volume = torch.zeros(num_features, L, L, device=input_volume.device, dtype=input_volume.dtype)

    ### swap quadrants by index slices
    output_volume[:, :L2, :L2] = input_volume[:, L2:L, L2:L]
    output_volume[:, L2:L, L2:L] = input_volume[:, :L2, :L2]

    output_volume[:, L2:L, :L2] = input_volume[:, :L2, L2:L]
    output_volume[:, :L2, L2:L] = input_volume[:, L2:L, :L2]

    output_volume[:, L2:L, L2:L] = input_volume[:, :L2, :L2]
    output_volume[:, :L2, :L2] = input_volume[:, L2:L, L2:L]

    return output_volume


def run_test(
        mean_std_amplitudes=((-5, 3, 1),
                             (5, 4, 1)),
        box_size=50,
        gaussian_norm=True,
        fft_norm='ortho',
        correlation=True,
        cmap='gray',
        vmin=0, vmax=1,
        swap_quadrants=True
):
    """
    Check the latest Pytorch FFT implementation is behaving as expected. Generate and plot the FFT correlation of two 2D Gaussian distributions.

    :param mean_std_amplitudes: Tuple of tuples, for the two 2D Gaussian mean, standard deviation (sigma), and amplitude. A mean of 5 sets the 2D mean position to (5,5).
    :param box_size: An even number to set the square dimension of an NxN 2D Gaussian kernel.
    :param gaussian_norm: Gaussian 1D kernel sums to 1.
    :param fft_norm: Normalize the fft output using Pytorch built-in methods
    :param correlation: Run an FFT correlation or FFT convolution
    :param cmap: color map for imshow()
    :param vmin: minimum plotted value
    :param vmax: maximum plotted value
    :return: None
    """

    assert box_size % 2 == 0, "Use and even number for box_size, otherwise test requires box padding"

    mean1, sigma1, amp1 = mean_std_amplitudes[0]
    mean2, sigma2, amp2 = mean_std_amplitudes[1]
    amp_check = amp1 * amp2

    gaussian_input1 = gaussian_2d(box_size, mean=mean1, sigma=sigma1, a=amp1, gaussian_norm=gaussian_norm)
    gaussian_input2 = gaussian_2d(box_size, mean=mean2, sigma=sigma2, a=amp2, gaussian_norm=gaussian_norm)

    if gaussian_norm:
        print('Gaussian norm = ', gaussian_norm)
        print('Check if kernel1 sums to 1: shape', gaussian_input1.shape, ' sigma=', sigma1, 'sum=',
              torch.sum(gaussian_input1))
        print('Check if kernel2 sums to 1: shape', gaussian_input1.shape, ' sigma=', sigma2, 'sum=',
              torch.sum(gaussian_input2))

    cplx_G1 = torch.fft.rfft2(gaussian_input1, dim=(-2, -1), norm=fft_norm)
    cplx_G2 = torch.fft.rfft2(gaussian_input2, dim=(-2, -1), norm=fft_norm)

    if correlation:
        ## Correlation
        gaussian_fft = torch.fft.irfft2(cplx_G1 * torch.conj(cplx_G2), dim=(-2, -1),
                                        norm=fft_norm)  ## this performs a proper correlation (what we want)
    else:
        ## Convolution
        gaussian_fft = torch.fft.irfft2(cplx_G1 * cplx_G2, dim=(-2, -1),
                                        norm=fft_norm)  ## this performs a convolution of the two shapes
        ### gaussian_fft = torch.fft.irfft2(torch.conj(cplx_G1) * torch.conj(cplx_G2), dim=(-2, -1), norm=fft_norm)   ## this line also performs a fft of the two shapes

    ### initialize plot figure
    fig, ax = plt.subplots(1, 5, figsize=(25, 5))

    ##### Plots 1 and 2
    ### Plotting 2D Gaussian inputs
    g1 = ax[0].imshow(gaussian_input1, cmap=cmap, vmin=vmin, vmax=vmax)
    ax[0].set_title('Gaussian1 ' + '$\mu_1$=' + str(mean1) + ' $\sigma_1=$' + str(sigma1))
    g2 = ax[1].imshow(gaussian_input2, cmap=cmap, vmin=vmin, vmax=vmax)
    ax[1].set_title('Gaussian2 ' + '$\mu_2$=' + str(mean2) + ' $\sigma_1=$' + str(sigma2))

    ##### Plot 3
    ### Plotting Convolution Output
    if swap_quadrants:
        fft = swap_fft_plot_quadrants(gaussian_fft.unsqueeze(0)).squeeze()
    else:
        fft = gaussian_fft
    conv = ax[2].imshow(fft, cmap=cmap, vmin=vmin, vmax=vmax)
    ax[2].set_title(r'Gaussian1 $\bigstar$ Gaussian2')

    ##### Plot 4
    ### Checking with output distribution mean and sigma: ~N(mean1+mean2, sigma1^2 + sigma2^2)
    result_sigma = np.sqrt(sigma1 ** 2 + sigma2 ** 2)
    if correlation:
        ## use difference for correlation (correlation == -fft; sign shifted mean due to reverse order of kernel operations compared to convolution)
        result_mean = mean1 - mean2
        fft_output = 'Correlation'
        ax[3].set_title('Expected Correlation ' + '\n $\mu_1-\mu_2$=' + str(
            result_mean) + '\n $\sqrt{\sigma_1^2 + \sigma_2^2}=$' + str(result_sigma)[:3])
    else:
        result_mean = mean1 + mean2
        fft_output = 'Convolution'
        ax[3].set_title('Expected Convolution ' + '\n $\mu_1+\mu_2$=' + str(
            result_mean) + '\n $\sqrt{\sigma_1^2 + \sigma_2^2}=$' + str(result_sigma)[:3])

    print('Generated FFT', fft_output)

    gaussian_check = gaussian_2d(boxsize, mean=result_mean, sigma=result_sigma, a=amp_check)
    scaled_gaussiancheck = gaussian_check
    gaussian_summedvariance = ax[3].imshow(scaled_gaussiancheck, cmap=cmap, vmin=vmin, vmax=vmax)

    ### Difference between fft output and expected gaussian output distribution (should give very small numbers)
    diff = ax[4].imshow(fft - gaussian_check, cmap=cmap, vmin=vmin, vmax=vmax)
    ax[4].set_title(fft_output + r' - Expected Gaussian')

    ax[0].grid(color='w')
    ax[1].grid(color='w')
    ax[2].grid(color='w')
    ax[3].grid(color='w')

    plt.show()


if __name__ == '__main__':
    ### Testing latest Pytorch FFT implementation behavior
    ## Specifically checking the FFT Correlation of two 2D Gaussians.

    ### initialize two different 2D gaussians mean and standard deviation
    mean1, sigma1, amplitude1 = (5, 3, 1)
    mean2, sigma2, amplitude2 = (5, 4, 1)

    mean_std_amplitudes = ((mean1, sigma1, amplitude1),
                           (mean2, sigma2, amplitude2))

    boxsize = 50
    cmap = 'gist_heat'
    # cmap = 'gist_heat_r'
    vmin = 0

    ### Gaussian norm applied to input
    # gaussian_norm = True
    gaussian_norm = False
    if gaussian_norm:
        vmax = 0.1
    else:
        vmax = 1

    ### Normalization schemes available for Torch >=v1.8 FFT
    # fft_norm = None
    fft_norm = 'ortho'
    # fft_norm = 'forward'
    # fft_norm = 'backward'

    correlation = True  # FFT Correlation
    # correlation = False # FFT Convolution

    run_test(
        mean_std_amplitudes,
        boxsize,
        gaussian_norm=gaussian_norm,
        fft_norm=fft_norm,
        correlation=correlation,
        cmap=cmap,
        vmin=vmin, vmax=vmax
    )
