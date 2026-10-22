import numpy as np
from scipy import signal
import torch

class DegradeToSNR:
    """
    Adds white Gaussian noise to reach target_snr_db. Split into two steps:
        measure(): noise power from the guranteed noise window (0-19 us), signal power
                   from the guranteed signal window (66-74 us) minus the noise power
        degrade(): adds the noise needed to bring signal/noise down to the target

    Input:  (2, length) - [I, Q]
    Output: (2, length) - [I, Q] float32, and the achieved SNR
    """
    
    # Fixed regions for noise and signal power estimation
    NOISE_WINDOW = slice(0, 2280)        # 0-19 us
    SIGNAL_WINDOW = slice(7920, 8880)    # 66-74 us

    def measure(self, tensor: torch.Tensor):
        """Returns (signal_power, noise_power), linear."""
        x = tensor.double()
        power = x[0] ** 2 + x[1] ** 2          # |I + jQ|^2
        noise_power = power[self.NOISE_WINDOW].mean().item()
        signal_power = power[self.SIGNAL_WINDOW].mean().item() - noise_power
        return signal_power, noise_power

    def degrade(self, tensor: torch.Tensor, signal_power: float, noise_power: float, target_snr_db):
        """Adds noise to go from the measured powers to target_snr_db. Returns (tensor, achieved_snr_db)."""
        current_snr_db = 10 * np.log10(signal_power / noise_power)
        if current_snr_db <= target_snr_db:
            return tensor

        target_snr = 10 ** (target_snr_db / 10)
        additive_noise_power = (signal_power / target_snr) - noise_power

        x = tensor.float().clone()
        length = x.shape[1]
        x[0] += torch.from_numpy(np.random.normal(0, np.sqrt(additive_noise_power / 2), length)).float()
        x[1] += torch.from_numpy(np.random.normal(0, np.sqrt(additive_noise_power / 2), length)).float()

        return x

    def __call__(self, tensor: torch.Tensor, target_snr_db):
        # No SNR degradation requested
        if target_snr_db == "NA":
            return tensor

        signal_power, noise_power = self.measure(tensor)
        return self.degrade(tensor, signal_power, noise_power, target_snr_db)

class NormaliseByPeakMagnitude:
    """
    Input:  (2, length) - [I, Q]
    Output: (2, length) - [I, Q] / peak, where peak = max(sqrt(I^2 + Q^2))
    """

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.float()

        # Manually calculate magnitude per call
        mag = torch.sqrt(tensor[0] ** 2 + tensor[1] ** 2)
        peak = mag.max()

        if not torch.isfinite(peak):
            raise ValueError(f"Non-finite peak magnitude: {peak}")

        if peak <= 1e-12:
            raise ValueError("Zero/near-zero magnitude pulse encountered")

        return tensor / peak
        
class FrequencyShift:
    """
    Applies a SPECIFIC frequency shift to a complex baseband I/Q signal --
    deterministic mechanism, the shift amount is provided by the caller.
    Used both for random augmentation (via RandomFrequencyShift) and for
    deliberate baseband CFO correction using a real PDW-measured value.

    Input:  (length, 2) - [I, Q]
    Output: (length, 2) - [I, Q], shifted by shift_hz
    """
    def __call__(self, tensor: torch.Tensor, shift_hz: float, fs: float) -> torch.Tensor:
        I = tensor[0].numpy().astype(np.float64)
        Q = tensor[1].numpy().astype(np.float64)
        signal = I + 1j * Q

        N = signal.shape[0]
        t = np.arange(N, dtype=np.float64) / fs
        osc = np.exp(1j * 2 * np.pi * shift_hz * t)

        shifted = signal * osc

        result = np.stack([shifted.real, shifted.imag], axis=0).astype(np.float32)
        return torch.from_numpy(result)


class RandomFrequencyShift:
    """
    Wraps FrequencyShift: samples
    shift_hz ~ Uniform(offset_hz - max_shift_hz, offset_hz + max_shift_hz)
    per call. Draws from NumPy's global RNG (same stream as DegradeToSNR).

    offset_hz defaults to 0.0, reproducing the original symmetric
    Uniform(-max_shift_hz, +max_shift_hz) range exactly. Setting offset_hz
    re-centres that same width elsewhere -- e.g. max_shift_hz=100e3 with
    offset_hz=100e3 gives Uniform(0, 200e3) instead of Uniform(-100e3, 100e3).
    """
    def __init__(self, max_shift_hz: float, offset_hz: float = 0.0):
        self.max_shift_hz = max_shift_hz
        self.offset_hz = offset_hz
        self.frequency_shift = FrequencyShift()

    def __call__(self, tensor: torch.Tensor, fs: float) -> torch.Tensor:
        shift_hz = np.random.uniform(
            self.offset_hz - self.max_shift_hz, self.offset_hz + self.max_shift_hz
        )
        return self.frequency_shift(tensor, shift_hz=shift_hz, fs=fs)
        
class PhaseShift:
    """
    Applies a SPECIFIC phase rotation to a complex baseband I/Q signal --
    deterministic mechanism, the angle is provided by the caller. Used
    both for random augmentation (via RandomPhaseShift) and for deliberate
    phase correction, mirroring how FrequencyShift/RandomFrequencyShift
    split mechanism from random-augmentation policy.

    Input:  (length, 2) - [I, Q]
    Output: (length, 2) - [I, Q], rotated by theta:
        I' + jQ' = (I + jQ) * e^{j*theta}
    which expands to:
        I' = I*cos(theta) - Q*sin(theta)
        Q' = I*sin(theta) + Q*cos(theta)
    """
    def __call__(self, tensor: torch.Tensor, theta: float) -> torch.Tensor:
        tensor = tensor.float()
        I = tensor[0]
        Q = tensor[1]

        cos_t = torch.cos(torch.tensor(theta, dtype=tensor.dtype))
        sin_t = torch.sin(torch.tensor(theta, dtype=tensor.dtype))

        I_rot = I * cos_t - Q * sin_t
        Q_rot = I * sin_t + Q * cos_t

        return torch.stack([I_rot, Q_rot], dim=0)

class RandomPhaseShift:
    """
    Wraps PhaseShift: samples theta ~ Uniform(0, 2*pi) per call. Draws
    from NumPy's global RNG (same stream as DegradeToSNR), so a single
    np.random.seed(...) call controls both, rather than phase shift
    living on its own separate, independently-seeded RNG stream.

    Useful as an augmentation since a receiver's carrier phase offset is
    arbitrary - the model shouldn't be sensitive to it.

    Input:  (length, 2) - [I, Q]
    Output: (length, 2) - [I, Q], rotated by a random phase
    """
    def __init__(self):
        self.phase_shift = PhaseShift()

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        theta = np.random.uniform(0, 2 * np.pi)
        return self.phase_shift(tensor, theta=theta)
 
 
class TimeShift:
    """
    Rolls the pulse right by a SPECIFIC number of samples. Samples falling off the end
    wrap to the start (like a shift register), so the trailing noise moves to the front.
    """
    def __call__(self, tensor: torch.Tensor, shift_samples: int) -> torch.Tensor:
        return torch.roll(tensor.float(), shifts=int(shift_samples), dims=-1)
 
class RandomTimeShift:
    """
    shift ~ randint(-max_shift_samples, max_shift_samples) per call, moving the pulse left
    or right.
    max_shift_samples must be given by the caller, since the safe range depends on the 
    pulse width.
    """
    def __init__(self, max_shift_samples: int):
        self.max_shift_samples = max_shift_samples
        self.time_shift = TimeShift()
 
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        shift = np.random.randint(-self.max_shift_samples, self.max_shift_samples + 1)
        return self.time_shift(tensor, shift_samples=shift)
