# RadarML

RadarML is a publicly available dataset of experimentally captured, modulated radar pulses, recorded over the air across multiple independent receive channels. It is intended to support research into multi-channel radar waveform classification.

The dataset was captured on the UCL ARESTOR platform, which is built on a Radio Frequency System-on-Chip (RFSoC). Each waveform was designed by its centre frequency, bandwidth, pulse width and modulation type. The overall dataset contains over 2 million waveforms across 7 modulation types. The goal of the dataset is to enable comparative analysis of radar modulation classification techniques on real data, and to stimulate research into multi-channel signal detection methods.

This repository provides PyTorch dataloaders for the raw captures, on-the-fly augmentations (SNR degradation, frequency, phase and time shifts), a spectrogram transform, a baseline VGG13 model, and two example notebooks showing how to use them.

---

## Repository Structure

```
Radar_ML/
├── dataloaders/
│   ├── pulse_dataset.py          # WaveformDataset and AugmentedWaveformDataset
│   ├── augmentations.py          # SNR degradation, normalisation, frequency/phase/time shifts
│   └── transforms.py             # Spectrogram transform (IQ -> 1-channel image)
├── models/
│   └── vgg13.py                  # VGG13_net: single-channel VGG13-style CNN
├── dataloader_demo.ipynb         # Demo: loading, viewing and augmenting the data
├── vgg13_training_demo.ipynb     # Demo: training and evaluating the VGG13 model
├── modrec.yml                    # Conda environment specification
├── requirements.txt              # pip requirements (alternative to Conda)
└── LICENSE                       # GPL-3.0 license
```

---

## Getting Started

### 1. Download the dataset

The dataset can be downloaded from:

- https://doi.org/10.5522/04/30752767

The password to extract the zip file is: **UCLRESM**

### 2. Clone the repository into the dataset folder

After extracting the zip file, run:

```bash
cd UCLRESM
git clone https://github.com/UCLRadarGroup/Radar_ML.git
cd Radar_ML
```

### 3. Set up the environment

A pre-configured Conda environment with all the required libraries is provided to ensure reproducibility. Assuming Conda is available, install and activate it with:

```bash
conda env create -f modrec.yml
conda activate modrec
```

Alternatively, install the requirements with pip:

```bash
pip install -r requirements.txt
```

### 4. Run the demos

Start with [`dataloader_demo.ipynb`](dataloader_demo.ipynb), then [`vgg13_training_demo.ipynb`](vgg13_training_demo.ipynb) (see [Demos](#demos) below). In each notebook, set `data_dir` to the folder holding the raw `.npy` captures.

---

## The Dataset

Each raw capture is a NumPy `.npy` file holding one waveform configuration:

| Axis | Size | Meaning |
|---|---|---|
| 0 | 3 | Receive channels, in the order `ADC0`, `ADC2`, `ADC4` |
| 1 | 1000 | Pulses |
| 2 | 33600 | `int16` samples, interleaved I and Q (`IQIQIQ...`), i.e. 16800 complex samples |

- **Sample rate:** 3.84 GHz / 32 = 120 MHz complex baseband, so each frame covers 140 µs and -60 to +60 MHz.
- **Channels:** `ADC0`, `ADC2` and `ADC4` are three independent digitisations of the same physical pulse, so pulse *i* is the same pulse on every channel. ADCs sharing an RF tile on the RFSoC were not used, which is why the numbering is not continuous.
- **Pulse position:** each pulse is centred in its frame.
- **High SNR:** the pulses are captured at a high signal-to-noise ratio, so that noise can be added afterwards to test across a controlled range of SNRs.

Files are named `<pulse_width>_<modulation>_<bandwidth>_<frequency>.npy`, for example:

```
10us_barker13_0Mhz_230Mhz.npy
50us_fmcw_10Mhz_265Mhz-275Mhz.npy       # frequency-modulated: start-stop frequency
10us_FSK#1_5Mhz_240Mhz.npy              # "#<index>" distinguishes variants of the same modulation
```

The frequencies in the file names are not baseband frequencies: the true baseband frequency is `240 MHz - <filename frequency>`. The dataloader applies this correction for you, so the frequency labels it returns are already correct.

---

## Demos

### [`dataloader_demo.ipynb`](dataloader_demo.ipynb): using the dataloaders

A step-by-step tour of the dataloaders, with plots at each step:

- Loading the raw data and checking the pulse counts per waveform
- Viewing a pulse's I/Q, magnitude, FFT and spectrogram
- The same physical pulse as seen by each ADC (`adc_mode="contrastive"`)
- Balancing waveforms with `max_pulses_per_waveform`
- SNR degradation and normalisation, unstacked and stacked
- Random frequency, phase and time shifts, and the bounds that keep each pulse inside the frame and the baseband
- Making results repeatable with a seed

### [`vgg13_training_demo.ipynb`](vgg13_training_demo.ipynb): training the provided model

An end-to-end example of training and evaluating `VGG13_net` on spectrograms of the raw data:

- Builds the dataset and a seeded train/test split
- Trains on pulses degraded from 30 to 0 dB, with random frequency, phase and time shifts; tests on SNR degradation only
- Trains with SGD, a cosine learning-rate schedule and early stopping, keeping the best model
- Saves the model, reloads it from file and evaluates it at each SNR
- Plots per-SNR confusion matrices (with MCC and macro F1) and F1 against SNR

---

## License

This dataset and code are released under the GPL-3.0 license, described in [`LICENSE`](LICENSE).

---

## Contact

For questions or issues, please open a GitHub issue or contact: m.ritchie@ucl.ac.uk.
