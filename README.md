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

At the start of the hackathon:

1. Download **`recorded_dataset.zip`** from the
   [UCL Radar ML - Lite dataset page](https://rdr.ucl.ac.uk/articles/dataset/Radar_ML_-_Lite/33977674).
2. Download `split_manifest.csv` and `split_manifest.json` separately from that
   page. If they are not yet publicly visible, use the repository copies:
   [CSV](hackathon/split_manifest.csv) and [JSON](hackathon/split_manifest.json).
   On GitHub, use **Download raw file** to save the actual file, not an HTML page.
3. Extract the ZIP and put both manifest files inside the extracted
   `recorded_dataset` folder, alongside the `.npy` files. No re-zipping is needed;
   the original ZIP and its checksum remain unchanged.

**Do not download, inspect or use `blind_test_participants.zip` until the
organizers announce final evaluation after model freeze**, even if it is visible
on the same download page. Train and tune only on the supplied 80/20 split.
The original full RadarML archive is not needed.

Open a terminal inside the extracted `recorded_dataset` folder.
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

## Blind final evaluation

A separate blind test has been created and verified: **13,200 held-out source
pulses across 660 configurations**, each at **13 requested SNRs from +30 to
−30 dB**, giving **171,600 examples**. All seven modulation classes appear at
every SNR. Its participant ZIP is approximately **11.55 GB**; allow about
**23.1 GB** to keep both the ZIP and extracted files.

The [seven-day event timetable](hackathon/EVENT_RULES.md) uses relative days:
kick-off on Day 0 at 09:00, model freeze on Day 4 at 17:00, blind-test release
on Day 5 at 09:00, and final submissions on Day 6 at 17:00 (Europe/London time).
The separate blind-test record stays private until release. Organizers will
announce the submission destination; the test
signals are not hosted in Git. The answer key and source mappings remain private.
See the [blind test guide](hackathon/README.md#7-final-evaluation-and-submission)
for loading, submission format, verification checksum and preprocessing caveats,
including **5.38% I/Q-value clipping at −30 dB** under the same rules as the main
dataset.

## Repository contents

| Path | Purpose |
| --- | --- |
| [hackathon/README.md](hackathon/README.md) | Participant walkthrough and event protocol |
| [hackathon/EVENT_RULES.md](hackathon/EVENT_RULES.md) | Relative-day timetable, data/model rules, submission limits and tie-breaking |
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
