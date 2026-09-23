# RadarML Lite hackathon guide

[Back to the dataset overview](../README.md)

## Your challenge

Predict a radar pulse's modulation family and measure how classification changes
as noise increases. Use the exact labels `barker`, `cw`, `fmcw`, `fsk`, `hyp`,
`nlfm`, `quad`. All Barker-code subtypes belong to the single `barker` class.

| Track | Inputs and processing |
| --- | --- |
| A: time domain | Ordered I/Q; magnitude, phase differences, normalization, time-domain statistics and learned 1D convolutions are allowed. No explicit FFT, STFT, wavelet or spectrogram representations. |
| B: time-frequency | Representations computed from the same I/Q, such as STFT spectrograms, with your chosen model. Include representation computation in inference timing. |

Compare tracks using identical split assignments and evaluation settings.

## Release and event status

The compact release includes recorded arrays, JSON metadata, expansion scripts,
a verification report and code license. It does **not** currently include a
train/validation split manifest, test set, submission template or scoring key.

The proposed protocol uses **80 training and 20 validation pulses per
configuration**, with additional undistributed captures reserved for final
testing. Organizers must publish the shared split manifest and final submission
arrangements before competitive results can be compared. The inspection and
expansion exercises below work now; a personal development split is not an
official event split.

## 1. Download and set up

Obtain `recorded_dataset.zip` from the organizers and extract it. Open a terminal
inside `recorded_dataset`, where `compact_dataset.py` is visible. Keep each
`*_recorded.npy` beside its matching JSON. Use Python 3.9 or newer.

On Windows, create an environment without needing PowerShell activation:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install numpy
```

On macOS/Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install numpy
```

Below, `python` means that environment's interpreter: use
`.\.venv\Scripts\python.exe` on Windows or `.venv/bin/python` on macOS/Linux.
Add libraries for your classifier and plots separately.

Allow about 4.44 GB for compact arrays and another 57.66 GB for full expansion.
Keeping the ZIP too uses approximately 64.63 GB, before models and features.

## 2. Inspect one recorded pulse

Each compact array has shape `(1, 100, 33600)`: receiver, pulse, interleaved I/Q.
A pulse contains 16,800 complex samples at 120 MS/s, spanning 140 µs including
surrounding noise. The active pulse may be shorter.

Save this as `inspect_pulse.py` in the extracted folder and run
`python inspect_pulse.py`:

```python
from pathlib import Path
import numpy as np

path = sorted(Path('.').glob('*_recorded.npy'))[0]
recording = np.load(path, mmap_mode='r', allow_pickle=False)
row = recording[0, 0]
iq = row[0::2].astype(np.float32) + 1j * row[1::2].astype(np.float32)
time_us = np.arange(iq.size) / 120_000_000 * 1e6
print('Stored array:', recording.shape, recording.dtype)
print('Complex pulse:', iq.shape, iq.dtype)
```

Expect `(1, 100, 33600) int16` and `(16800,) complex64`. I and Q are components
of one signal, not receiver channels. Amplitudes are ADC values, not calibrated
RF power. Recorded arrays have no requested SNR axis or synthetic noise.

## 3. Try a small SNR expansion

Create a `pilot` folder inside the extracted dataset folder. Copy one
`*_recorded.npy` and its matching JSON into it, then run:

```bash
python compact_dataset.py expand --input pilot --output ../pilot_expanded --snrs 30 0 -30
```

Expect one `(1, 3, 100, 33600)` array, approximately 20.16 MB, and metadata.
This tests the workflow; one configuration cannot assess seven-class recognition.

When ready for full expansion:

```bash
python compact_dataset.py expand --input . --output ../expanded
```

The default uses seed 42 and requested SNRs
`30, 25, 20, 15, 10, 5, 0, -5, -10, -15, -20, -25, -30` dB. Each
`*_single_channel.npy` has shape `(1, 13, 100, 33600)`. A smaller sweep can use
`--snrs 20 10 0 -10 -20`, but still processes every input recording.

Checksums are verified before processing each recording. Choose a new output
directory for each run. Expansion cannot currently resume: after interruption,
restart in a new directory and manage incomplete output separately. Do not treat
a partly generated folder as a complete release.

Use the scripts bundled with the compact ZIP together. The repository's
`make_dataset.py` and `hackathon/make_dataset_hackathon.py` process original
multi-channel inputs and are not replacements for this expansion command.

## 4. Load an expanded pulse

From the extracted compact folder:

```python
import json
from pathlib import Path
import numpy as np

path = sorted(Path('../expanded').glob('*_single_channel.npy'))[0]
meta = json.loads(path.with_suffix('.json').read_text())
data = np.load(path, mmap_mode='r', allow_pickle=False)
snr_index = meta['snr_db'].index(0)
row = data[0, snr_index, 0]
iq = row[0::2].astype(np.float32) + 1j * row[1::2].astype(np.float32)
print(iq.shape, meta['snr_db'][snr_index])
```

For the pilot, replace `../expanded` with `../pilot_expanded`. Read the SNR order
from metadata. Memory-map arrays and load batches, not the entire collection.

`participant_loader.py` in this repository supports a different format: mixed
`(N, 33600)` shards with `index.csv` files. It does not directly load these
per-configuration compact or expanded arrays.

## 5. Keep training and validation separate

Use the event's shared split manifest when supplied. The proposed 80/20 split
contains 52,800 training and 13,200 validation source pulses. At all 13 SNRs,
these become 686,400 and 171,600 examples respectively.

- Assign splits by source recording and original pulse identity before adding
  noise. Every SNR version of a pulse stays in the same split.
- Fit learned preprocessing only on training data. Use validation to choose
  settings, not to fit normalization, features or model weights.
- Use filenames and metadata for managing examples and labels only. Never use
  filenames, pulse IDs, labels, SNR settings or storage locations as model inputs.
- Crop using a fixed rule or signal-derived detector, not filename pulse widths.
  Keep validation captures out of training augmentation.

This evaluates new pulses from known configurations, not unseen configurations
or recording sessions. Source filenames expose labels, so this compact release
is an open development dataset, not a blind test set.

## 6. Develop and evaluate

Start with a manageable training subset covering all seven classes; record its
selection rule. Build a complete training-to-prediction pipeline and evaluate
on validation before scaling. Record seeds, SNRs, settings, dataset version,
runtime and hardware. Freeze preprocessing and model choices before final testing.

Training may use a subset of SNRs. Full event evaluation is proposed across all
13 settings. At each SNR, average recall across the seven classes. The main
score averages those balanced accuracies across all 13 SNRs. Also report per-SNR
scores and confusion matrices. A reduced class or SNR subset is not a full score.

Report **requested SNR (dB)**. Preprocessing estimates noise over 0–20 µs and
signal power over 21–119 µs after subtracting noise power. That signal window
includes padding for shorter pulses. Adding noise cannot improve source SNR,
so unattainable higher targets receive no additional noise. Values are clipped
to int16 limits; metadata records clipping and unattainable-target counts.
Requested SNR therefore need not equal the achieved active-pulse SNR.

## 7. Final evaluation and submission

Final test captures and submission arrangements will be provided or administered
by the organizers. Do not use the original public archive to recover held-out
labels. Private test-selection details and keys are not part of this guide.

If a prediction template is supplied, preserve every `sample_id` and predict
exactly one permitted label for each test example, using columns:

```csv
sample_id,predicted_label
```

If a local scoring key is supplied, run from the repository root, replacing the
example paths with your actual files:

```bash
python hackathon/score_submission.py --key /path/to/test_labels.csv --submission /path/to/predictions.csv --output score.json
```

The compact ZIP does not include a key or template. A distributed key makes this
a labeled holdout: freeze your method first and disclose any subsequent tuning.
For organizer-run inference, follow the published model-submission interface.

Provide reproduction instructions, code/settings, model, track, training subset
and SNRs, score curves, confusion matrices and a short account of what worked.
Report training time, inference time per pulse, hardware and model size. Include
signal preprocessing in inference timing.

## Attribution

Ritchie, White and Hosford (2025),
[RadarML, version 1](https://doi.org/10.5522/04/30752767.v1).
See the [overview](../README.md#attribution-and-licensing) for licensing references
and [contact details](../README.md#contact) for questions.
