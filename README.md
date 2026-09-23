# RadarML Lite

RadarML contains radar waveforms recorded over the air using UCL's ARESTOR
RFSoC platform. This branch supports a reduced, single-receiver dataset for
learning and comparing radar modulation classifiers. Configurations vary in
modulation, pulse duration, bandwidth and centre frequency.

**Taking part in the hackathon? Start with the [Hackathon guide](hackathon/README.md)**
for setup, signal loading, SNR expansion, evaluation and submission guidance.

## The reduced dataset

The compact archive preserves recorded samples without adding synthetic noise.
Participants can generate different signal-to-noise ratio (SNR) conditions locally,
keeping the download smaller than a dataset containing every noise variant.
Original receiver noise remains; these are not noise-free signals.

| Property | RadarML Lite compact release |
| --- | --- |
| Waveform configurations | 660 |
| Receiver | ADC0, one of the original three channels |
| Pulses per configuration | First 100 recorded pulses, source indices 0–99 |
| Total pulses | 66,000 |
| Classes | `barker`, `cw`, `fmcw`, `fsk`, `hyp`, `nlfm`, `quad` |
| Samples per pulse | 16,800 complex samples at 120 MS/s |
| Array format | int16, `(1, 100, 33600)`, interleaved I/Q |
| Compact ZIP | Approximately 2.54 GB |
| Extracted signal arrays | Approximately 4.44 GB |

The original archive has three receivers, ADC0, ADC2 and ADC4, with 1,000 pulses
per receiver for 648 configurations and 8,000 for the remaining 12. RadarML Lite
selects one receiver and 100 pulses from each configuration.

Expanding to 13 requested SNR settings (+30 to −30 dB in 5 dB steps) creates
858,000 examples and approximately 57.66 GB of arrays. The compact arrays use
13 times less storage, a reduction of 92.3%. Noise variants are not additional
independent captures.

## Download and quick start

Obtain **`recorded_dataset.zip` from the hackathon organizers**. A public compact
download link has not yet been added here. The full source archive is available
separately from the [RadarML dataset record](https://doi.org/10.5522/04/30752767.v1);
participants using the compact release do not need to download it.

Extract the compact ZIP and open a terminal inside its `recorded_dataset` folder.
With Python 3.9 or newer installed:

```bash
python -m pip install numpy
python compact_dataset.py expand --input . --output ../expanded
```

The expansion scripts are **included in the compact ZIP**. Use those bundled
scripts together; the repository's original preprocessing scripts expect a
different input format. Full expansion requires **57.66 GB of additional space**.
See the [guide](hackathon/README.md) for environment setup, a small first check
and loading examples before expanding everything.

To clone this branch:

```bash
git clone --branch radar-ml-lite --single-branch https://github.com/UCLRadarGroup/Radar_ML.git
```

## Repository contents

| Path | Purpose |
| --- | --- |
| [hackathon/README.md](hackathon/README.md) | Participant walkthrough and event protocol status |
| [hackathon/](hackathon/) | Reduced-data preparation, plotting and shard-release utilities |
| [make_dataset.py](make_dataset.py) | Original multi-channel SNR preprocessing |
| [plot_raw_data.py](plot_raw_data.py), [plot_dataset.py](plot_dataset.py) | Original dataset visualizations |
| [VGG13-Waveform-Classification-Example.ipynb](VGG13-Waveform-Classification-Example.ipynb) | Original spectrogram/classifier example |
| [MultiChannelRESM.yml](MultiChannelRESM.yml), [requirements.txt](requirements.txt) | Original example environment and dependencies |

The original notebook and scripts are reference material; their data layout and
split handling are not the compact hackathon workflow. NumPy is sufficient for
compact loading and expansion. Install libraries for your method as needed;
the original environment includes platform-specific dependencies.

## Attribution and licensing

Source: Ritchie, White and Hosford (2025),
[RadarML, version 1](https://doi.org/10.5522/04/30752767.v1).
See [LICENSE](LICENSE) for repository code terms and the dataset record for
dataset licensing. Code licensing does not replace dataset terms.

## Contact

Open a GitHub issue or contact m.ritchie@ucl.ac.uk.
