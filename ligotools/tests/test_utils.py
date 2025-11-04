
import numpy as np
from ligotools.utils import whiten, reqshift

def test_whiten_preserves_length():
    # Set sampling rate and time step
    fs = 4096.0
    dt = 1.0 / fs
    # Create a 1-second 100 Hz sine wave
    x = np.sin(2 * np.pi * 100 * np.arange(0, 1.0, dt))
    # Flat PSD: whitening should mostly pass through shape/length
    flat_psd = lambda f: np.ones_like(f)
    # Run whitening
    y = whiten(x, flat_psd, dt)
    # Sanity: whitening must not change array length
    assert len(y) == len(x)

def test_reqshift_preserves_length():
    # 1-second signal at 4096 Hz sample rate
    fs = 4096
    t = np.arange(fs) / fs
    # Pure 100 Hz tone
    x = np.sin(2 * np.pi * 100 * t)
    # Shift frequency content by +400 Hz
    y = reqshift(x, fshift=400.0, sample_rate=fs)
    # Sanity: frequency shift must not change array length
    assert len(y) == len(x)

