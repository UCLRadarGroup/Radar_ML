# Seven-day timetable and event rules

Day 0 is the opening day. Day 1 is the following calendar day, and the event
ends on Day 6: seven calendar days, inclusive. All times use **Europe/London**
local time. The organizers announce the calendar date corresponding to Day 0
and the submission destination before the event. This schedule does not depend
on a particular start date.

## Timetable

| Day and time | Activity | Required outcome |
| --- | --- | --- |
| Day 0, 09:00 | Kick-off and development-data access | Read the guide, download the compact training ZIP and separate split CSV/JSON, check loading, and choose Track A, Track B or both. |
| Day 1 | First complete pipeline | Train an initial model and produce validation predictions. Check class labels and balanced accuracy. |
| Day 2 | Signal processing and model experiments | Compare preprocessing and model choices using validation only. Record settings and seeds. |
| Day 3 | Robustness and efficiency | Evaluate across requested SNRs, examine class confusions, and measure runtime and memory needs. |
| Day 4, before 17:00 | Final training and validation | Select the method, finish training and save final validation results. |
| **Day 4, 17:00** | **Model freeze deadline** | Submit the frozen model and preprocessing, code/configuration, dependency versions, inference command and artifact hashes. Freeze one pipeline per entered track. |
| **Day 5, 09:00** | **Blind-test release** | Organizers provide the separate blind-test download after checking freeze submissions. Teams verify, extract and run inference with frozen pipelines. |
| Day 5 through Day 6, 17:00 | Final inference and report preparation | Complete predictions, check IDs and file format, and prepare the method summary. No model tuning or test-driven adaptation. |
| **Day 6, 17:00** | **Final submission deadline** | Submit predictions and the final method/reproducibility materials for each track. |
| Day 6, after 17:00 | Organizer scoring and wrap-up | Score accepted entries, discuss methods and announce provisional results. Final rankings follow any reproducibility checks. |

Training and tuning are allowed from Day 0 at 09:00 until Day 4 at 17:00.
The blind inference window is 32 hours. Allow time for the approximately
11.55 GB test download and about 23.1 GB of disk space for ZIP plus extraction.

If organizer-side download or submission infrastructure fails, organizers must
announce any revised deadline to all teams. Individual extensions are not part
of the standard rules.

## Training data and permitted tools

- Use the supplied RadarML Lite recordings and the published 80/20 split.
  Fit models, learned normalization, feature transforms and calibration using
  **training pulses only**. Use validation for evaluation and selecting settings;
  do not add validation pulses to a final training run.
- All SNR variants and augmentations inherit their original pulse's split.
  Training SNR subsets are allowed. Report the subset used; final blind scoring
  covers every supplied test example at all 13 requested SNR settings.
- Train models from scratch. External training datasets, additional recordings
  from the original RadarML archive, pretrained weights and external learned
  feature extractors are not permitted. Do not recover test labels by matching
  signals to the public archive.
- Augmentation derived from supplied training pulses is allowed when it
  preserves the intended class. Additional independently generated labeled
  waveform datasets are excluded from the standard competition.
- Standard libraries, papers, tutorials, public code and AI coding assistants
  may be used. Cite reused methods/code and disclose AI assistance in the method
  summary. These tools do not permit external learned weights or extra data.
  Do not upload blind-test signals to external services for analysis or prediction.
- CPUs, GPUs and cloud compute are allowed; no fixed compute quota is imposed.
  Declare hardware and training time. Runtime is reported but is not a ranking
  tie-break because teams may use different hardware.
- Follow the [track restrictions](README.md#your-challenge). Model inputs are
  permitted signal representations and fixed acquisition constants, never
  filenames, IDs, source metadata or requested-SNR labels.

## Model freeze and blind evaluation

By Day 4 at 17:00, submit one frozen pipeline per track, including all ensemble
members if applicable. Supply the saved weights, preprocessing/configuration,
code archive or fixed Git commit, dependency versions and an inference command.
Record SHA256 hashes of model and configuration files so the organizer can
check that the final run used the frozen artifacts. A commit alone is not a
substitute for supplying accessible weights and settings.

After freeze, no retraining, threshold tuning, model selection, ensemble changes,
test-time training or fitting preprocessing to test data is allowed. Fixed
per-pulse processing defined before freeze, such as normalizing an individual
pulse by its own RMS, remains allowed. Run the model in inference mode.

An execution-only repair, such as a dependency or file-path correction, requires
organizer agreement and a recorded diff. It must not alter weights, preprocessing
or prediction logic. Keep the original frozen artifacts for comparison.

The test is held in a separate dataset record pending release. Do not obtain,
inspect or use test signals before Day 5 at 09:00. Disclose any earlier access
to the organizers before model freeze; they will determine eligibility for the
blind ranking. The answer key remains organizer-only throughout the event.

## Submission limits and required files

Each team may enter both tracks, with **one final entry per track**. Submit:

1. `predictions.csv`: exactly `sample_id,predicted_label`, one row per test ID,
   using only `barker`, `cw`, `fmcw`, `fsk`, `hyp`, `nlfm`, `quad`.
2. The frozen pipeline identifier and artifact hashes, plus any approved
   execution-only repair record.
3. A brief method summary with track, training data/SNR subset, seeds, validation
   scores and confusion matrices, dependencies, code attribution, hardware,
   training time, inference time per pulse and model size. Include representation
   generation in inference timing.

Identify the team and track in the submission folder/message, not by changing
the required prediction columns. Submit through the destination announced at
kick-off; do not send predictions or private files to the public repository.

Before Day 6 at 17:00, a team may replace a submission to correct packaging,
formatting or an incomplete inference run using the same frozen pipeline. The
latest valid submission per team/track received before the deadline is scored.
This does not permit testing multiple models. No late replacements are accepted
unless an event-wide infrastructure extension is announced.

Organizers may confirm file receipt and schema/ID validity before the deadline,
but provide **no blind-test accuracy, per-SNR scores or confusion matrices** until
submissions close. Missing, duplicate or extra IDs and unknown labels make a
prediction file invalid.

## Ranking and tie-breaking

Rank Track A and Track B separately. The primary score is the mean of the
seven-class balanced accuracies across all 13 requested SNR settings, with
equal weight per class and SNR.

Compare scores before display rounding. Differences of at most **0.000001** on
the 0–1 score scale count as a tie. For a primary-score tie, compare mean balanced
accuracy over the three lowest requested SNRs: **−20, −25 and −30 dB**. Apply the
same tolerance. If those also tie, award a shared place; do not break ties using
submission time or self-reported runtime.

Publish overall and per-SNR scores and confusion matrices after the deadline.
Do not publish per-example answers during the event. Rankings are conditional
on compliance with data/track rules and reproducibility of the frozen pipeline.

## Organizer readiness checks

Before Day 0, announce the start-date mapping, submission destination and support
contact. Verify the public training download and separate split files. Keep the
blind-test record private and retain the answer key separately. Account for any
earlier public exposure; removing a file does not revoke existing downloads.

After Day 4 freeze, check that each entry has accessible frozen artifacts before
releasing the test on Day 5. After submissions close, score privately using the
existing scorer and apply the ranking rules above. Its standard overall score
is unchanged; the three-lowest-SNR tie-break is calculated from its per-SNR output.
