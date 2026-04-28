# RadarML
A GIT repo for the Radar_ML dataset, read-me and code

This GIT repo links to the dataset release of the UCL RadarML dataset.
Reference:
Ritchie, Matt; White, Ryan; Hosford, Adam (2025). RadarML Dataset. University College London. Dataset. https://doi.org/10.5522/04/30752767.v1

URL for dataset download
https://rdr.ucl.ac.uk/articles/dataset/RadarML_Dataset/30752767?file=60005198

# The UCLRESM.zip file password is "UCLRESM"

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


### Note: make_dataset.py currently writes output files to processed/, while plot_dataset.py reads from degraded/. Rename one directory or update the path in plot_dataset.py so the scripts use the same processed-data folder.


### `requirements.txt`
Pinned `pip` dependency list for running the scripts and notebook. Includes NumPy, SciPy, Matplotlib, OpenCV, scikit-learn, PyTorch, TorchVision, CUDA-related PyTorch packages, and tqdm.

### MultiChannelRESM.yml

Conda environment file for reproducing the development environment. It defines a Python 3.10 environment named MultiChannelRESM, with Jupyter/IPython support and the same core ML, plotting, and signal-processing packages used by the project.

Create the environment with:

conda env create -f MultiChannelRESM.yml
conda activate MultiChannelRESM


### VGG13-Waveform-Classification-Example.ipynb

Example Jupyter notebook showing how to train and evaluate a VGG13-style PyTorch classifier on RadarML spectrograms.

The notebook:

- converts processed RadarML .npy files into spectrogram image datasets;
- creates train/test folders organised by ADC channel and waveform class;
- uses SNR values ≥ 0 dB for training and all SNR values for testing;
- defines a custom PyTorch SpectrogramDataset;
- implements a single-channel VGG13-style CNN;
- trains the model using cross-entropy loss and Adam;
- saves best and latest model weights;
- evaluates performance with per-SNR confusion matrices.

Expected input folder:

processed/

Generated output folders/files include:

SpectrogramData/
weights/
adc*_confusion_matrix_snr_*dB.png


