"""
Utility functions for signal processing and visualization used
in the LIGO Gravitational Wave Detection Tutorial.

Functions:
- whiten:      Whiten a strain time-series using its PSD.
- write_wavfile: Save an audio waveform to a .wav file.
- reqshift:    Frequency-shift a time-series by a given factor.
- plot_psd:    Plot a power spectral density curve for a signal.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.io.wavfile import write
from scipy.interpolate import interp1d
from scipy.io import wavfile
from scipy.signal import filtfilt






def whiten(strain, interp_psd, dt):
    Nt = len(strain)
    freqs = np.fft.rfftfreq(Nt, dt)
    freqs1 = np.linspace(0, 2048, Nt // 2 + 1)

    # whitening: transform to freq domain, divide by asd, then transform back,
    # taking care to get normalization right.
    hf = np.fft.rfft(strain)
    norm = 1./np.sqrt(1./(dt*2))
    white_hf = hf / np.sqrt(interp_psd(freqs)) * norm
    white_ht = np.fft.irfft(white_hf, n=Nt)
    return white_ht


def write_wavfile(filename, fs, data):
    """
    Save a time-series array as a normalized 16-bit WAV audio file.

    Parameters
    ----------
    filename : str
        Output .wav file path.
    fs : int
        Sampling frequency (Hz).
    data : array_like
        Signal array to write (automatically scaled to integer limits).
    """
    d = np.int16(data / np.max(np.abs(data)) * 32767 * 0.9)
    wavfile.write(filename, int(fs), d)



def reqshift(data, fshift=100.0, sample_rate=4096.0):
    """
    Frequency shift a real-valued signal by a constant offset.

    Parameters
    ----------
    data : array_like
        Input time-domain signal.
    fshift : float, optional
        Frequency shift in Hz (default 100 Hz).
    sample_rate : float, optional
        Sampling rate in Hz (default 4096 Hz).

    Returns
    -------
    ndarray
        Frequency-shifted signal in the time domain.
    """
    x = np.fft.rfft(data)
    T = len(data) / float(sample_rate)
    df = 1.0 / T
    nbins = int(fshift / df)
    y = np.roll(x.real, nbins) + 1j * np.roll(x.imag, nbins)
    y[0:nbins] = 0.0
    z = np.fft.irfft(y)
    return z






def plot_match_results(
    det,
    eventname,
    time,
    timemax,
    tevent,
    SNR,
    strain_whitenbp,
    template_match,
    template_fft,
    datafreq,     # freq axis for template_fft
    data_psd,     # PSD evaluated on `freqs`
    fs,
    d_eff,
    freqs,        # freq axis for PSD
    plottype="png",
):
    """
    Plot SNR, whitened data vs template (and residual), and PSD/ASD overlay.

    Parameters
    ----------
    det : {"H1","L1"}
    eventname : str
    time : array_like
    timemax : float
    tevent : float
    SNR : array_like
    strain_whitenbp : array_like
    template_match : array_like
    template_fft : array_like
    datafreq : array_like
        Frequencies for template_fft (same length/spacing).
    data_psd : array_like
        One-sided PSD values on `freqs`.
    fs : float
    d_eff : float
        Effective distance used for scaling template in freq domain.
    freqs : array_like
        Frequencies for `data_psd`.
    plottype : str
    """
    pcolor = "g" if det == "L1" else "r"

    # --- SNR plots ---
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(time - timemax, SNR, pcolor, label=f"{det} SNR(t)")
    plt.grid(True)
    plt.ylabel("SNR")
    plt.xlabel(f"Time since {timemax:.4f}")
    plt.legend(loc="upper left")
    plt.title(f"{det} matched filter SNR around event")

    plt.subplot(2, 1, 2)
    plt.plot(time - timemax, SNR, pcolor, label=f"{det} SNR(t)")
    plt.grid(True)
    plt.ylabel("SNR")
    plt.xlim([-0.15, 0.05])
    plt.xlabel(f"Time since {timemax:.4f}")
    plt.legend(loc="upper left")
    plt.savefig(f"figures/{eventname}_{det}_SNR.{plottype}")

    # --- Time-domain fit and residual ---
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(time - tevent, strain_whitenbp, pcolor, label=f"{det} whitened h(t)")
    plt.plot(time - tevent, template_match, "k", label="Template(t)")
    plt.ylim([-10, 10])
    plt.xlim([-0.15, 0.05])
    plt.grid(True)
    plt.xlabel(f"Time since {timemax:.4f}")
    plt.ylabel("whitened strain (units of noise stdev)")
    plt.legend(loc="upper left")
    plt.title(f"{det} whitened data around event")

    plt.subplot(2, 1, 2)
    plt.plot(time - tevent, strain_whitenbp - template_match, pcolor, label=f"{det} resid")
    plt.ylim([-10, 10])
    plt.xlim([-0.15, 0.05])
    plt.grid(True)
    plt.xlabel(f"Time since {timemax:.4f}")
    plt.ylabel("whitened strain (units of noise stdev)")
    plt.legend(loc="upper left")
    plt.title(f"{det} Residual whitened data after subtracting template around event")
    plt.savefig(f"figures/{eventname}_{det}_matchtime.{plottype}")

    # --- PSD/ASD vs template in frequency ---
    plt.figure(figsize=(10, 6))
    template_f = np.abs(template_fft) * np.sqrt(np.abs(datafreq)) / d_eff
    plt.loglog(datafreq, template_f, "k", label="template(f)*sqrt(f)")
    plt.loglog(freqs, np.sqrt(data_psd), pcolor, label=f"{det} ASD")
    plt.xlim(20, fs / 2)
    plt.ylim(1e-24, 1e-20)
    plt.grid(True)
    plt.xlabel("frequency (Hz)")
    plt.ylabel("strain noise ASD (strain/rtHz), template h(f)*rt(f)")
    plt.legend(loc="upper left")
    plt.title(f"{det} ASD and template around event")
    plt.savefig(f"figures/{eventname}_{det}_matchfreq.{plottype}")


