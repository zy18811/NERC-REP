"""
Functions to do cross wavelet analysis and plot results

Most of code sourced from https://github.com/regeirk/pycwt/blob/master/pycwt/sample/sample_xwt.py
"""

import numpy as np
from july21_get_all_data import get_july21_all_data
import matplotlib.pyplot as plt
import pycwt as wavelet
from pycwt.helpers import find
from matplotlib.image import NonUniformImage


def cross_wavelet(df, col1, col2, x_skip=20, y_skip=5, y_lim=400, box_pdf_s1=False, box_pdf_s2=False, cache=False):
    """
    Perfroms cross-wavelet analysis on col1 and col2 of df and plots results

    :param df: dataframe, index should be in time order
    :param col1: Column name of df
    :param col2: Column name of df
    :param x_skip: Spread of phase arrows in x direction
    :param y_skip: Spread of phase arrows in y direction
    :param y_lim: y limit of correlation plot
    :param box_pdf_s1: Bool to change PDF of col1, good for highly bi-modal data
    :param box_pdf_s2: Bool to change PDF of col2, good for highly bi-modal data
    :param cache: Bool to cache significance, makes subsequent runs of code for same time series faster
    :return: None
    """

    # takes the values of col1 and col2 from df
    s1 = df[col1].values
    s2 = df[col2].values
    t2 = t1 = df.index.values


    """
    plots s1 and and s2 against time
    """
    fig, (ax1, ax2) = plt.subplots(2,1)
    ax1.plot(t1, s1, label=col1)
    ax1.grid()
    ax1.legend()
    ax2.plot(t2, s2, label=col2)
    ax2.grid()
    ax2.legend()
    plt.show()

    # calculates variables used later
    dt = np.diff(t1)[0]
    n1 = t1.size
    n2 = t2.size
    n = min(n1, n2)

    # Change the probability density function (PDF) of the data if Bool is True
    if box_pdf_s1: s1, _, _ = wavelet.helpers.boxpdf(s1)
    if box_pdf_s2: s2, _, _ = wavelet.helpers.boxpdf(s2)

    # Calculates standard deviation of s1 and s2 for later use
    std1 = s1.std()
    std2 = s2.std()

    """
    Calculate the CWT of both normalized time series. The function wavelet.cwt
    returns a a list with containing [wave, scales, freqs, coi, fft, fftfreqs]
    variables.
    """
    mother = wavelet.Morlet(6)  # Morlet mother wavelet with m = 6
    slevel = 0.95   # Significance level
    dj = 1 / 12     # Twelve sub-octaves per octaves
    s0 = -1     # Default value
    J = -1      # Default value

    # Lag-1 autocorrelation for red noise
    alpha1, _, _ = wavelet.ar1(s1)
    alpha2, _, _ = wavelet.ar1(s2)

    """
    The following routines perform the wavelet transform and significance analysis for the two time series
    """
    W1, scales1, freqs1, coi1, _, _ = wavelet.cwt(s1 / std1, dt, dj, s0, J, mother)
    signif1, fft_theor1 = wavelet.significance(1.0, dt, scales1, 0, alpha1,
                                               significance_level=slevel,
                                               wavelet=mother)
    W2, scales2, freqs2, coi2, _, _ = wavelet.cwt(s2 / std2, dt, dj, s0, J, mother)
    signif2, fft_theor2 = wavelet.significance(1.0, dt, scales2, 0, alpha2,
                                               significance_level=slevel,
                                               wavelet=mother)

    power1 = (np.abs(W1)) ** 2  # Normalized wavelet power spectrum
    power2 = (np.abs(W2)) ** 2  # Normalized wavelet power spectrum
    period1 = 1 / freqs1
    period2 = 1 / freqs2
    sig95_1 = np.ones([1, n1]) * signif1[:, None]
    sig95_1 = power1 / sig95_1  # Where ratio > 1, power is significant
    sig95_2 = np.ones([1, n2]) * signif2[:, None]
    sig95_2 = power2 / sig95_2  # Where ratio > 1, power is significant

    """
    Plots CWT for both time series
    """
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)

    extent1 = [t1.min(), t1.max(), 0, max(period1)]
    extent2 = [t2.min(), t2.max(), 0, max(period2)]
    im1 = NonUniformImage(ax1, interpolation='bilinear', extent=extent1)
    im1.set_data(t1, period1, power1)
    ax1.images.append(im1)
    ax1.contour(t1, period1, sig95_1, [-99, 1], colors='k', linewidths=2,
                extent=extent1)
    ax1.fill(np.concatenate([t1, t1[-1:] + dt, t1[-1:] + dt, t1[:1] - dt, t1[:1] - dt]),
             np.concatenate([coi1, [1e-9], period1[-1:], period1[-1:], [1e-9]]),
             'k', alpha=0.3, hatch='x')
    ax1.set_title('{} Wavelet Power Spectrum ({})'.format(col1,
                                                          mother.name))

    im2 = NonUniformImage(ax2, interpolation='bilinear', extent=extent2)
    im2.set_data(t2, period2, power2)
    ax2.images.append(im2)
    ax2.contour(t2, period2, sig95_2, [-99, 1], colors='k', linewidths=2,
                extent=extent2)
    ax2.fill(np.concatenate([t2, t2[-1:] + dt, t2[-1:] + dt, t2[:1] - dt, t2[:1] - dt]),
             np.concatenate([coi2, [1e-9], period2[-1:], period2[-1:], [1e-9]]),
             'k', alpha=0.3, hatch='x')
    ax2.set_xlim(max(t1.min(), t2.min()), min(t1.max(), t2.max()))
    ax2.set_title('{} Wavelet Power Spectrum ({})'.format(col2,
                                                          mother.name))

    """
    Calculate the cross wavelet transform (XWT). The XWT finds regions in time
    frequency space where the time series show high common power. Torrence and
    Compo (1998) state that the percent point function -- PPF (inverse of the
    cumulative distribution function) -- of a chi-square distribution at 95%
    confidence and two degrees of freedom is Z2(95%)=3.999. However, calculating
    the PPF using chi2.ppf gives Z2(95%)=5.991. To ensure similar significance
    intervals as in Grinsted et al. (2004), one has to use confidence of 86.46%.
    """
    W12, cross_coi, freq, signif = wavelet.xwt(s1, s2, dt, dj=1 / 12, s0=-1, J=-1,
                                               significance_level=0.8646,
                                               wavelet='morlet', normalize=True)

    cross_power = np.abs(W12) ** 2
    cross_sig = np.ones([1, n]) * signif[:, None]
    cross_sig = cross_power / cross_sig  # Power is significant where ratio > 1
    cross_period = 1 / freq

    """
    Calculate the wavelet coherence (WTC). The WTC finds regions in time
    frequency space where the two time seris co-vary, but do not necessarily have
    high power.
    """
    WCT, aWCT, corr_coi, freq, sig = wavelet.wct(s1, s2, dt, dj=1 / 12, s0=-1, J=-1,
                                                 significance_level=0.8646,
                                                 wavelet='morlet', normalize=True,
                                                 cache=cache)

    cor_sig = np.ones([1, n]) * sig[:, None]
    cor_sig = np.abs(WCT) / cor_sig  # Power is significant where ratio > 1
    cor_period = 1 / freq

    """
    Calculates the phase between both time series. The phase arrows in the
    cross wavelet power spectrum rotate clockwise with 'north' origin.
    The relative phase relationship convention is the same as adopted
    by Torrence and Webster (1999), where in phase signals point
    upwards (N), anti-phase signals point downwards (S). If X leads Y,
    arrows point to the right (E) and if X lags Y, arrow points to the
    left (W).
    """
    angle = 0.5 * np.pi - aWCT
    u, v = np.cos(angle), np.sin(angle)

    """
    Plots figures
    """
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.55, 0.05, 0.35])
    cbar_ax_1 = fig.add_axes([0.85, 0.05, 0.05, 0.35])

    extent_cross = [t1.min(), t1.max(), 0, max(cross_period)]
    extent_corr = [t1.min(), t1.max(), 0, max(cor_period)]
    im1 = NonUniformImage(ax1, interpolation='bilinear', extent=extent_cross)
    im1.set_data(t1, cross_period, cross_power)
    ax1.images.append(im1)
    ax1.contour(t1, cross_period, cross_sig, [-99, 1], colors='k', linewidths=2,
                extent=extent_cross)
    ax1.fill(np.concatenate([t1, t1[-1:] + dt, t1[-1:] + dt, t1[:1] - dt, t1[:1] - dt]),
             np.concatenate([cross_coi, [1e-9], cross_period[-1:],
                             cross_period[-1:], [1e-9]]),
             'k', alpha=0.3, hatch='x')
    ax1.set_title('Cross-Wavelet')

    ax1.quiver(t1[::x_skip], cross_period[::y_skip], u[::y_skip, ::x_skip], v[::y_skip, ::x_skip],
               units='width', angles='uv', pivot='mid', linewidth=1,
               edgecolor='k', headwidth=10, headlength=10, headaxislength=5,
               minshaft=2, minlength=5)
    fig.colorbar(im1, cax=cbar_ax)

    im2 = NonUniformImage(ax2, interpolation='bilinear', extent=extent_corr)
    im2.set_data(t1, cor_period, WCT)
    ax2.images.append(im2)
    ax2.contour(t1, cor_period, cor_sig, [-99, 1], colors='k', linewidths=2,
                extent=extent_corr)
    ax2.fill(np.concatenate([t1, t1[-1:] + dt, t1[-1:] + dt, t1[:1] - dt, t1[:1] - dt]),
             np.concatenate([corr_coi, [1e-9], cor_period[-1:], cor_period[-1:],
                             [1e-9]]),
             'k', alpha=0.3, hatch='x')
    ax2.set_title('Cross-Correlation')

    ax2.quiver(t1[::x_skip], cor_period[::y_skip], u[::y_skip, ::x_skip], v[::y_skip, ::x_skip],
               units='height', angles='uv', pivot='mid', linewidth=1,
               edgecolor='k', headwidth=10, headlength=10, headaxislength=5,
               minshaft=2, minlength=5)

    ax2.set_ylim(2, y_lim)
    ax2.set_xlim(max(t1.min(), t2.min()), min(t1.max(), t2.max()))
    fig.colorbar(im2, cax=cbar_ax_1)

    plt.draw()
    plt.show()


if __name__ == '__main__':
    """
    Does cross-wavelet on vwc and co2 flux for Spen Farm,Hanger Field, July 2021 data
    """
    df = get_july21_all_data('interp')
    df = df.reset_index(level=0)

    col1 = 'COSMOS_VWC'
    col2 = 'co2_flux'

    cross_wavelet(df, col1, col2,cache=True)

