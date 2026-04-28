# RadarML
A GIT repo for the Radar_ML dataset, read-me and code

This GIT repo links to the dataset release of the UCL RadarML dataset.
Reference:
Ritchie, Matt; White, Ryan; Hosford, Adam (2025). RadarML Dataset. University College London. Dataset. https://doi.org/10.5522/04/30752767.v1

URL for dataset download
https://rdr.ucl.ac.uk/articles/dataset/RadarML_Dataset/30752767?file=60005198

## Python scripts

### `make_dataset.py`
Creates the full RadarML dataset with a range of SNRs using additive noise to degrade the initial full SNR examples. These are created from the raw `.npy` files. The script expects input files in an `unprocessed/` directory and writes processed outputs to `processed/`. For each raw file, it loads three-channel interleaved I/Q radar data, generates noisy versions across predefined SNR levels from +30 dB to -30 dB, and stores the result as a 4D NumPy array with shape:

`[channel, SNR index, pulse/repeat, interleaved samples]`

Key settings:
- `repeats_per_snr = 300`
- Sampling frequency: `fs = 3.84e9 / 32`
- Channels: `ADC0`, `ADC2`, `ADC4`
- SNR levels: `30, 27, ..., -30 dB`
- Uses deterministic per-file/per-SNR/per-pulse random seeds for reproducibility.

### plot_raw_data.py

Visualises raw RadarML .npy files before degradation. The script searches for .npy files in the same directory as the script, loads each file, and plots the final acquisition/pulse for each radar channel.

For each selected pulse it generates:

Power versus time
I, Q, magnitude, and phase versus time
Spectrogram

This script is useful for checking the structure and quality of the raw radar waveform files before creating the degraded dataset.

### plot_dataset.py

Visualises the degraded/processed RadarML dataset. The script expects processed .npy files in a degraded/ directory, then iterates through channels and SNR levels, plotting the first pulse for each SNR.

For each channel/SNR combination it generates:

Power versus time
I, Q, magnitude, and phase versus time
Spectrogram

This script is useful for inspecting how different SNR levels affect the radar waveform data.



