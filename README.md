# Write-up in progress - contact ryan.white.23@ucl.ac.uk with questions 

---

# RadarML 

RadarML dataset a publicly available dataset of experimentally captured modulated radar pulses recorded over-the-air across multiple independent receive channels, enabling researching into multi-channel waveform classification.

The dataset was captured on the UCL ARESTOR platform which is a Radio-Frequency-System-on-a-chip (RFSoC). Each waveform was designed based on its central frequency, bandwidth, duration and modulation type. The overall dataset contains over 2 million waveforms across 7 different modulation types. The goal of this dataset is to enable comparative analysis of Radar Modulation Classification techniques on real data and stimulate research into multi-channel signal detection methods.

---

## Repository Structure

```
Radar_ML/
├── plot_raw_data.py                                     # Visualisation script (see below)
├── make_dataset.py                                      # Dataset construction script (see below)
├── plot_dataset.py                                      # Visualisation script (see below)
├── VGG13-Waveform-Classification-Example.ipynb          # Worked example notebook (see below)
├── environment.yml                                      # Conda environment specification
└── LICENSE.md                                           # License terms
```

## Getting Started

### 1. Download the dataset

The dataset can be downloaded from the following link:

- https://doi.org/10.5522/04/30752767

The password to extract the zip file is: **UCLRESM**

### 2. Clone the repository into the dataset folder

After extracing the zip file, run:

```bash
cd UCLRESM
git clone https://github.com/UCLRadarGroup/Radar_ML.git
cd Radar_ML
```

### 3. Set up the Conda environment

A pre-configured Conda environment with all the required libraries is provided to ensure reproducibility.

Assuming conda is available, this can be installed by running the following:

```bash
conda env create -f MultiChannelRESM.yml
conda activate MultiChannelRESM
```

---

## Building the Dataset

The raw dataset is collected at a high signal-to-noise ratio (SNR) to allow arbitrary levels of additive white Gaussian noise (AWGN) to be added, enabling controlled testing across a wide range of SNR conditions.

### `make_dataset.py`

This script is the most important. It builds the SNR degraded dataset by first loading the raw ".npy" data and applying additive white gaussian noise (generated with deterministic randomness for repeatability).

If desired, the range of SNRs and the repeats per SNR can be modified in the python script.

This will create an additional folder labelled "processed".

### `plot_dataset.py`

This script enables visualisation of the degraded RadarML dataset. The script expects processed ".npy" files in the "./processed" directory, it then iterates through channels and SNR levels, plotting the first pulse for each SNR.

- The following plots are generated:

  - Power in decibels against time
  - In-phase (I), Quadrature (Q), Magnitude and phase components against time
  - Spectrogram

## Training a baseline model

### `VGG13-Waveform-Classification-Example.ipynb`

Provided is an example "Jupyter notebook" showing how to generate spectrograms from the degraded dataset and then use this to train and evaluate a custom VGG13-style PyTorch classifier from scratch.

The notebook performs the following steps when executed sequentially:

- Converts processed RadarML .npy files into spectrogram image datasets;
- Creates train/test folders organised by ADC channel and waveform class;
- Uses SNR values ≥ 0 dB for training and all SNR values for testing;
- Defines a custom PyTorch SpectrogramDataset;
- Implements a single-channel VGG13-style CNN;
- Trains the model using cross-entropy loss and Adam optimiser;
- Saves the best and latest model weights;
- Evaluates performance with per-SNR confusion matrices.

---

## License

This dataset and code are released under the terms described in [`LICENSE.md`](LICENSE.md).

---

## Contact

For questions or issues, please open a GitHub issue or contact: m.ritchie@ucl.ac.uk.
