"""Build RadarML with ONE receiver channel and 100 pulses per SNR.

Adapted from UCLRadarGroup/Radar_ML make_dataset.py, GPL-3.0.
Hackathon modifications: channel/pulse reduction, CLI, bounded memory and guards.

Run: python make_dataset_hackathon.py --input-dir unprocessed --output-dir processed_hackathon
Output shape per file: (1, 13, 100, 33600), int16, with the default settings.
The singleton channel axis is retained for compatibility with the original layout.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

# Defaults requested for the hackathon.
REPEATS_PER_SNR = 100
TARGET_SNR_DBS = [30, 25, 20, 15, 10, 5, 0, -5, -10, -15, -20, -25, -30]
CHANNEL_NAMES = ['ADC0', 'ADC2', 'ADC4']
FS = 3.84e9 / 32
GLOBAL_SEED = 42


def output_name(filename):
    """Preserve upstream's IF-to-baseband filename conversion."""
    parts = Path(filename).stem.split('_')
    frequencies = parts[-1].split('-')
    try:
        parts[-1] = '-'.join(f'{240 - int(f.removesuffix("Mhz"))}Mhz' for f in frequencies)
    except ValueError as exc:
        raise ValueError(f'Unexpected raw-data frequency field in {filename}') from exc
    return '_'.join(parts) + '_single_channel.npy'


def degrade(row, snr_db, rng):
    """Use the original power-estimation windows and return interleaved int16 IQ.

    Guard against negative added-noise power and int16 wraparound. The SNR
    estimate is averaged over 21..119 us, as upstream; it is not the in-pulse
    SNR for shorter pulses. Returns output, unattainable-target flag, clip count.
    """
    iq = row[0::2].astype(np.float64) + 1j * row[1::2].astype(np.float64)
    noise_power = np.mean(np.abs(iq[:int(20e-6 * FS)]) ** 2)
    signal_power = np.mean(np.abs(iq[int(21e-6 * FS):int(119e-6 * FS)]) ** 2) - noise_power
    if not np.isfinite(signal_power) or signal_power <= 0:
        raise ValueError('Non-positive signal power in the original estimation window')
    added_power = signal_power / (10.0 ** (snr_db / 10.0)) - noise_power
    unattainable = added_power < 0
    sigma = np.sqrt(max(0.0, added_power) / 2.0)
    noisy = iq + rng.normal(0, sigma, iq.size) + 1j * rng.normal(0, sigma, iq.size)
    interleaved = np.empty(row.size, dtype=np.float64)
    interleaved[0::2], interleaved[1::2] = noisy.real, noisy.imag
    clipped = int(np.count_nonzero((interleaved < -32768) | (interleaved > 32767)))
    # Casting without clipping would wrap large values and corrupt low-SNR data.
    return np.clip(interleaved, -32768, 32767).astype(np.int16), unattainable, clipped


def process_file(source, output_dir, channel='ADC0', pulses=REPEATS_PER_SNR,
                 snrs=TARGET_SNR_DBS, seed=GLOBAL_SEED, overwrite=False):
    source, output_dir = Path(source), Path(output_dir)
    channel_idx = CHANNEL_NAMES.index(channel)
    raw = np.load(source, mmap_mode='r', allow_pickle=False)
    if raw.ndim != 3 or raw.shape[0] != 3 or raw.shape[2] != 33600 or raw.dtype != np.int16:
        raise ValueError(f'{source.name}: expected int16 (3, N, 33600), got {raw.dtype} {raw.shape}')
    if not 1 <= pulses <= raw.shape[1]:
        raise ValueError(f'{source.name}: requested {pulses} pulses, available {raw.shape[1]}')
    if not snrs or not np.all(np.isfinite(snrs)):
        raise ValueError('Provide at least one finite SNR value')
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / output_name(source.name)
    metadata_path = target.with_suffix('.json')
    if (target.exists() or metadata_path.exists()) and not overwrite:
        raise FileExistsError(f'{target}: already exists; use --overwrite to replace')
    temporary = target.with_suffix('.partial.npy')
    # Map output to disk; do not allocate all SNRs or load all raw channels in RAM.
    output = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.int16,
                                     shape=(1, len(snrs), pulses, raw.shape[2]))
    unattainable_count = clipped_count = 0
    try:
        for pulse_idx in range(pulses):
            row = raw[channel_idx, pulse_idx]
            for snr_idx, snr_db in enumerate(snrs):
                # Same seed convention as upstream for reproducibility.
                key = f'{seed}_{source.name}_{snr_db}_{pulse_idx}'
                noise_seed = int(hashlib.md5(key.encode()).hexdigest()[:8], 16)
                values, unattainable, clipped = degrade(row, snr_db, np.random.default_rng(noise_seed))
                output[0, snr_idx, pulse_idx] = values
                unattainable_count += int(unattainable)
                clipped_count += clipped
        output.flush()
    finally:
        del output
    temporary.replace(target)
    metadata = dict(source_file=source.name, source_shape=list(raw.shape),
                    channel=channel, source_channel_index=channel_idx, output_channel_index=0,
                    output_shape=[1, len(snrs), pulses, raw.shape[2]],
                    dtype='int16', sample_rate_hz=FS, snr_db=list(snrs), seed=seed,
                    source_pulse_indices=list(range(pulses)),
                    snr_definition='Original 21..119 us signal window minus 0..20 us noise power',
                    unattainable_snr_examples=unattainable_count, clipped_iq_values=clipped_count)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'{source.name} -> {target.name}; {channel}, {pulses} pulses/SNR; '
          f'unattainable SNR targets={unattainable_count}, clipped IQ values={clipped_count}', flush=True)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument('--input-dir', type=Path, default=here / 'unprocessed')
    parser.add_argument('--output-dir', type=Path, default=here / 'processed_hackathon')
    parser.add_argument('--channel', choices=CHANNEL_NAMES, default='ADC0')
    parser.add_argument('--pulses', type=int, default=REPEATS_PER_SNR)
    parser.add_argument('--snrs', type=int, nargs='+', default=TARGET_SNR_DBS)
    parser.add_argument('--seed', type=int, default=GLOBAL_SEED)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if not args.input_dir.is_dir():
        parser.error(f'Input directory does not exist: {args.input_dir}')
    files = sorted(p for p in args.input_dir.glob('*.npy')
                   if not p.name.endswith(('_single_channel.npy', '_degraded.npy', '.partial.npy')))
    if not files:
        parser.error(f'No .npy files in {args.input_dir}')
    if args.pulses <= 0:
        parser.error('--pulses must be positive')
    names = [output_name(f.name) for f in files]
    if len(set(names)) != len(names):
        parser.error('Multiple inputs map to the same output filename')
    if not args.overwrite and any((args.output_dir / n).exists() or (args.output_dir / n).with_suffix('.json').exists() for n in names):
        parser.error('Output files already exist; select a new directory or use --overwrite')
    total_bytes = len(files) * len(args.snrs) * args.pulses * 33600 * 2
    print(f'{len(files)} files; output array payload ~{total_bytes / 1e9:.2f} GB '
          f'({total_bytes / 2**30:.2f} GiB). Processing one file at a time.', flush=True)
    for source in files:
        process_file(source, args.output_dir, args.channel, args.pulses, args.snrs, args.seed, args.overwrite)
    print('All files processed.')


if __name__ == '__main__':
    main()
