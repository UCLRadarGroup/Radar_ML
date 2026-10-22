"""
Transforms for the raw .npy pulse dataset.

Unlike the augmentations, transforms are deterministic: they change how a pulse is
represented (e.g. IQ -> spectrogram image), not the pulse itself. Pass one to
AugmentedWaveformDataset(transform=...) and it is applied as the last step, after
every augmentation.

Every transform takes a (2, length) [I, Q] tensor.
"""

import numpy as np
import torch


class Spectrogram:
    """
    STFT power spectrogram in dB as a 1-channel image, at its natural resolution.
    Hann window, NFFT=256, 50% overlap, zero frequency centred.
    Normalised to [0, 1] using each spectrogram's own min and max dB.

    Input:  (2, length) - [I, Q]
    Output: (1, NFFT, n_segments) float32 in [0, 1] - (channel, frequency, time)
            Frequency row 0 = -fs/2. For a 16800-sample pulse: (1, 256, 130).
    """
    NFFT = 256
    NOVERLAP = 128

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        I = tensor[0].numpy().astype(np.float64)
        Q = tensor[1].numpy().astype(np.float64)
        cmplx_data = I + 1j * Q

        L = cmplx_data.shape[0]
        step = self.NFFT - self.NOVERLAP       # hop size (128 for 256 FFT w/ 50% overlap)
        if L < self.NFFT:
            num_segments = 0
        else:
            num_segments = 1 + (L - self.NFFT) // step

        freq_data = np.zeros((num_segments, self.NFFT), dtype=complex)
        window = np.hanning(self.NFFT)

        for i in range(num_segments):
            start = i * step
            segment = cmplx_data[start : start + self.NFFT] * window
            freq_data[i, :] = np.fft.fft(segment, self.NFFT)

        freq_data = np.fft.fftshift(freq_data, axes=1)
        freq_data = freq_data.T                # (frequency, time)

        power = np.abs(freq_data) ** 2
        power_db = 10 * np.log10(power + 1e-12)   # Add small amount to avoid log 0

        # Normalise to [0, 1] using this spectrogram's own min and max
        min_val = power_db.min()
        max_val = power_db.max()
        img = (power_db - min_val) / (max_val - min_val)

        # (1, frequency, time) float32, ready for a model
        return torch.from_numpy(img.astype(np.float32)).unsqueeze(0)

