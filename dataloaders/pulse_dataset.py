"""
DataLoader for raw multi-ADC radar pulse captures stored as .npy files.

Each .npy file has shape [3, 1000, 33600]:
    - 3     : ADCs, in the order ADC_CHANNELS = ["ADC0", "ADC2", "ADC4"]
    - 1000  : pulses
    - 33600 : int16 samples, interleaved IQIQIQ... -> 16800 complex samples

Files are named <pulse_width>_<waveform>_<bandwidth>_<freq>.npy, e.g.
    10us_barker13_0Mhz_230Mhz.npy
    50us_fmcw_10Mhz_265Mhz-275Mhz.npy
    10us_FSK#1_5Mhz_240Mhz.npy        (waveform "fsk" - the #index is dropped)

The loader supports:
- Waveform filtering
- ADC selection via adc_names
- Lazy loading (default) or load_to_memory
- Limiting the number of pulses per waveform

Pulses are returned as int16 with I and Q as separate channels (channel 0 = I, 1 = Q):
adc_mode="contrastive": each item is (n_adc, 2, 16800), one row per adc_names entry.
adc_mode="single":      each ADC view is its own item, (2, 16800), so the dataset is
                        len(adc_names) times longer.

All ADCs are in the same file, so pulse i is the same physical pulse on every ADC -
no time-of-arrival matching is needed.
"""

from pathlib import Path
from typing import List, Optional, Literal, Union, Callable
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset

from .augmentations import DegradeToSNR, NormaliseByPeakMagnitude, RandomFrequencyShift, RandomTimeShift, RandomPhaseShift

class WaveformDataset(Dataset):

    ADC_CHANNELS = ["ADC0", "ADC2", "ADC4"]   # order of axis 0 in every .npy file
    SAMPLE_RATE = 3.84e9 / 32

    def __init__(
        self,
        data_dir: str,
        waveform_filter: Optional[List[str]] = None,
        adc_names: Optional[List[str]] = None,
        adc_mode: Literal["single", "contrastive"] = "single",
        load_to_memory: bool = False,
        max_pulses_per_waveform: Optional[int] = None,
    ):
        """
        Args:
            data_dir: folder of raw .npy files.
            waveform_filter: waveforms to keep, e.g. ["barker13", "fsk"]. None = all.
            adc_names: REQUIRED, e.g. ["ADC0"] or ["ADC0", "ADC4"]. Output follows this order.
            adc_mode: "single" (default) or "contrastive", see module docstring.
            load_to_memory: load the selected pulses into RAM for faster training.
            max_pulses_per_waveform: limit pulses per waveform. Pulses are taken evenly
                across that waveform's files (pulse 0 of every file, then pulse 1, ...).
        """
        if not adc_names:
            raise ValueError(f"adc_names is required, e.g. adc_names=['ADC0']")

        self.adc_names = adc_names
        self.adc_idx = [self.ADC_CHANNELS.index(a) for a in adc_names]
        self.adc_mode = adc_mode
        self.load_to_memory = load_to_memory
        self.waveform_filter = waveform_filter

        # ---- Group files by waveform ----
        self.files = sorted(Path(data_dir).glob("*.npy"))
        if not self.files:
            raise FileNotFoundError(f"No .npy files found in {Path(data_dir).resolve()}")
        self.file_labels = []
        type_to_files = defaultdict(list)

        for file_idx, file in enumerate(self.files):
            parts = file.stem.split("_")
            waveform = parts[1].lower().split("#")[0]

            # Correct frequencies (MHz): true freq = 240 - filename freq
            if "-" in parts[3]:
                # Frequency modulated, e.g. 265Mhz-275Mhz
                left, right = parts[3].split("-")
                start_freq = 240 - int(left.replace("Mhz", ""))
                stop_freq = 240 - int(right.replace("Mhz", ""))
            else:
                # Single frequency, e.g. 230Mhz
                start_freq = stop_freq = 240 - int(parts[3].replace("Mhz", ""))
            centre_freq = (start_freq + stop_freq) / 2

            self.file_labels.append({
                "waveform": waveform,
                "pulse_width": parts[0],
                "bandwidth": parts[2],
                "start_freq": start_freq,
                "stop_freq": stop_freq,
                "centre_freq": centre_freq,
                "file": file.stem,
            })
            if waveform_filter and waveform not in waveform_filter:
                continue
            type_to_files[waveform].append(file_idx)

        self.waveforms = sorted(type_to_files)
        if not self.waveforms:
            raise ValueError(f"No files match waveform_filter={waveform_filter}")
        self.waveform_to_idx = {w: i for i, w in enumerate(self.waveforms)}

        # ---- Build index of (file_idx, pulse_idx) ----
        n_pulses = np.load(self.files[0], mmap_mode="r").shape[1]

        # Above the smallest waveform's total, waveforms are no longer balanced
        smallest = min(len(type_to_files[w]) for w in self.waveforms) * n_pulses
        if max_pulses_per_waveform and max_pulses_per_waveform > smallest:
            print(f"WARNING: max_pulses_per_waveform={max_pulses_per_waveform} is more than the "
                  f"smallest waveform has ({smallest}), so waveforms will be unbalanced")

        self.pulse_index = []
        for waveform in self.waveforms:
            pulses = [(f, p) for p in range(n_pulses) for f in type_to_files[waveform]]
            self.pulse_index += pulses[:max_pulses_per_waveform]

        print(f"Dataset: {len(self.pulse_index)} pulses x {len(adc_names)} ADCs, "
              f"waveforms: {self.waveforms}")

        if load_to_memory:
            self._load_to_memory()

    def _load_to_memory(self):
        print("Loading all data to memory...")
        self._mem = np.empty((len(self.pulse_index), len(self.adc_idx), 33600), dtype=np.int16)

        by_file = defaultdict(list)
        for i, (file_idx, pulse_idx) in enumerate(self.pulse_index):
            by_file[file_idx].append((i, pulse_idx))

        for file_idx, items in by_file.items():
            data = np.load(self.files[file_idx], mmap_mode="r")
            for i, pulse_idx in items:
                self._mem[i] = data[self.adc_idx, pulse_idx]

        print(f"Loaded {len(self.pulse_index)} pulses to memory")

    def _read(self, i):
        """int16 (n_adc, 33600) for pulse i, selected ADCs only."""
        if self.load_to_memory:
            return self._mem[i]
        file_idx, pulse_idx = self.pulse_index[i]
        return np.load(self.files[file_idx], mmap_mode="r")[self.adc_idx, pulse_idx]

    def __len__(self):
        if self.adc_mode == "single":
            return len(self.pulse_index) * len(self.adc_names)
        return len(self.pulse_index)

    def __getitem__(self, idx):
        """
        contrastive: pulse (n_adc, 2, 16800) int16, labels dict
        single:      pulse (2, 16800) int16,        labels dict + 'adc'
        Channel 0 is I, channel 1 is Q. Convert to float on the GPU, e.g.
            x = x.to(device, non_blocking=True).float()
        """
        if self.adc_mode == "single":
            i, adc = divmod(idx, len(self.adc_names))
        else:
            i, adc = idx, None

        file_idx, pulse_idx = self.pulse_index[i]
        labels = dict(self.file_labels[file_idx])
        labels["waveform_idx"] = self.waveform_to_idx[labels["waveform"]]
        labels["pulse_idx"] = pulse_idx

        raw = self._read(i)
        if adc is not None:
            raw = raw[adc]
            labels["adc"] = self.adc_names[adc]

        # Interleaved IQIQ -> separate I and Q channels (..., 2, 16800), kept as int16
        iq = np.stack([raw[..., 0::2], raw[..., 1::2]], axis=-2)
        return torch.from_numpy(iq), labels
        
class AugmentedWaveformDataset(Dataset):
    """
    SNR-degradation and normalisation.
    Wraps a WaveformDataset (or Subset of one, e.g. from random_split).
 
    Output is always float32: added noise and normalised values aren't whole numbers,
    so they can't go back into int16. The base dataset stays int16.
 
    `mode` controls how the target SNRs combine:
        mode="single" (default): every base item appears once per target SNR, pulse-major:
            aug[k * len(target_snrs_db) + j] is base item k at target_snrs_db[j].
        mode="contrastive": one item per base item, with every target SNR stacked.
 
    ADC-aware: if the base was built with adc_mode="contrastive", each base item is
    (n_adc, 2, L). Each ADC view is degraded independently, using THAT view's own
    measured noise and signal power, then re-stacked:
        mode="single":      (n_adc, 2, L)
        mode="contrastive": (n_adc * n_snr, 2, L), ADC-major then SNR order
    If the base was built with adc_mode="single", each base item is (2, L):
        mode="single":      (2, L)
        mode="contrastive": (n_snr, 2, L)
 
    labels gain:
        "snr_idx": (mode="single" only) index into target_snrs_db
    """
 
    # Constants for frequency randomisation to prevent nyquist issues when mixing
    BASEBAND_MHZ = WaveformDataset.SAMPLE_RATE / 2 / 1e6    # complex baseband covers -fs/2 to +fs/2
    GUARD_MHZ = 2.0                                         # margin kept clear at each band edge after a random frequency shift

    SAMPLES_PER_US = int(WaveformDataset.SAMPLE_RATE / 1e6) # calculated number of samples per us
    TIME_GUARD_US = 2.0                                     # margin kept clear at the end of the frame after a random time shift
 
    def __init__(
        self,
        base_dataset: WaveformDataset,
        target_snrs_db: List[Union[float, str]],
        mode: Literal["single", "contrastive"] = "single",
        random_freq: bool = False,
        random_phase: bool = False,
        random_time: bool = False,
        normalise: bool = False,
        transform: Optional[Callable] = None
    ):
        if mode not in ("single", "contrastive"):
            raise ValueError(f"mode must be 'single' or 'contrastive', got {mode!r}")
 
        self.base_dataset = base_dataset
        self.target_snrs_db = target_snrs_db
        self.normalise = normalise
        self.mode = mode
        self.snr_copies = len(target_snrs_db)
 
        self.random_freq = random_freq
        self.random_phase = random_phase
        self.random_time = random_time
 
        self.degrade_snr = DegradeToSNR()
 
        self.transform = transform
       
        # Only treat the base as ADC-stacked if it's actually handing us stacks
        self.multi_adc = self._base_attr(base_dataset, "adc_mode", "single") == "contrastive"
 
    @staticmethod
    def _base_attr(dataset, name, default):
        """Walk through Subset wrapping (e.g. from random_split) to find an attribute on the underlying base dataset."""
        seen = dataset
        while not hasattr(seen, name) and hasattr(seen, "dataset"):
            seen = seen.dataset
        return getattr(seen, name, default)
 
    def __len__(self):
        if self.mode == "contrastive":
            return len(self.base_dataset)
        return len(self.base_dataset) * self.snr_copies
 
    # Order is important here: SNR degradation first (its windows are fixed positions), normalise last
    def _augment_pulse(self, pulse, snr_idx, labels):
        """One (2, L) view -> float32 (2, L)."""
        pulse = self.degrade_snr(pulse, self.target_snrs_db[snr_idx])
 
        pulse = pulse.float()

        if self.random_freq:
            # Shift range (Hz) that keeps start_freq..stop_freq inside +/- (BASEBAND_MHZ - GUARD_MHZ)
            limit = self.BASEBAND_MHZ - self.GUARD_MHZ
            low = min(labels["start_freq"], labels["stop_freq"])
            high = max(labels["start_freq"], labels["stop_freq"])
            min_shift_hz, max_shift_hz = (-limit - low) * 1e6, (limit - high) * 1e6
            pulse = RandomFrequencyShift(
                max_shift_hz=(max_shift_hz - min_shift_hz) / 2,
                offset_hz=(max_shift_hz + min_shift_hz) / 2
            )(pulse, fs=WaveformDataset.SAMPLE_RATE)

        if self.random_time:
            # Pulse is centred, so the gap after it is (frame - pulse) / 2. Shift by up to that, minus the margin.
            pulse_len = float(labels["pulse_width"].replace("us", "")) * self.SAMPLES_PER_US
            max_shift = int((pulse.shape[-1] - pulse_len) / 2 - self.TIME_GUARD_US * self.SAMPLES_PER_US)
            pulse = RandomTimeShift(max_shift_samples=max(max_shift, 0))(pulse)

        if self.random_phase:
            pulse = RandomPhaseShift()(pulse)
            
        if self.normalise:
            pulse = NormaliseByPeakMagnitude()(pulse)
 
        if self.transform is not None:
            pulse = self.transform(pulse)
 
        return pulse
 
    def _augment_views(self, base_pulse, snr_idxs, labels):
        """Augments every (ADC, SNR) pair. Returns (n_views, 2, L), ADC-major then SNR."""
        adc_views = base_pulse if self.multi_adc else base_pulse.unsqueeze(0)
 
        views = []
        for adc_pulse in adc_views:
            for snr_idx in snr_idxs:
                views.append(self._augment_pulse(adc_pulse, snr_idx, labels))
        return torch.stack(views, dim=0)
 
    def __getitem__(self, idx):
        if self.mode == "contrastive":
            base_idx, snr_idxs = idx, range(self.snr_copies)
        else:  # single
            base_idx, snr_idx = divmod(idx, self.snr_copies)
            snr_idxs = [snr_idx]
 
        base_pulse, labels = self.base_dataset[base_idx]
        labels = dict(labels)
 
        views = self._augment_views(base_pulse, snr_idxs, labels)
 
        if self.mode == "single":
            labels["snr_idx"] = snr_idx
            if not self.multi_adc:
                # One ADC, one SNR: plain (2, L) pulse
                views = views[0]
 
        return views, labels

