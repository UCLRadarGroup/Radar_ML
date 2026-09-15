"""Plot one real example per RadarML modulation family or filename subtype.

Requires numpy and matplotlib. Supports raw (3,N,33600) and reduced
(1,SNR,N,33600) NPY files. Reduced files require their companion JSON.
"""
import argparse
import json
import os
from pathlib import Path
import re

import numpy as np

FAMILIES = {'barker', 'cw', 'fmcw', 'fsk', 'hyp', 'nlfm', 'quad'}
CHANNELS = ['ADC0', 'ADC2', 'ADC4']
FS = 120000000


def subtype(path):
    parts = path.stem.split('_')
    if len(parts) < 2:
        raise ValueError(f'Cannot read modulation label from {path.name}')
    value = parts[1].split('#')[0].lower()
    if not re.fullmatch('[a-z0-9]+', value):
        raise ValueError(f'Unexpected modulation label: {value}')
    return value


def spectrogram(iq, fs, nfft=256):
    """Two-sided STFT, time bin centers, and power relative to this example's peak."""
    hop = nfft // 2
    frames = np.lib.stride_tricks.sliding_window_view(iq, nfft)[::hop]
    power = np.abs(np.fft.fftshift(np.fft.fft(frames * np.hanning(nfft), axis=1), axes=1)) ** 2
    peak = max(float(power.max()), np.finfo(float).tiny)
    db = 10 * np.log10(np.maximum(power / peak, 1e-12))
    times = (np.arange(len(frames)) * hop + (nfft - 1) / 2) / fs
    frequencies = np.fft.fftshift(np.fft.fftfreq(nfft, 1 / fs))
    return times, frequencies, db.T


def load_example(path, pulse, snr, channel):
    array = np.load(path, mmap_mode='r', allow_pickle=False)
    if array.dtype != np.int16 or array.shape[-1] != 33600:
        raise ValueError(f'{path}: expected int16 with 33600 interleaved values')
    details = dict(file=path.name, pulse_index=pulse, subtype=subtype(path))
    if array.ndim == 4:
        metadata = json.loads(path.with_suffix('.json').read_text())
        if array.shape != (1, len(metadata['snr_db']), len(metadata['source_pulse_indices']), 33600):
            raise ValueError(f'{path}: shape and metadata disagree')
        if channel != metadata['channel']:
            raise ValueError(f'{path}: contains {metadata["channel"]}, not {channel}')
        if snr not in metadata['snr_db'] or not 0 <= pulse < array.shape[2]:
            raise ValueError(f'{path}: requested pulse or SNR is unavailable')
        row = array[0, metadata['snr_db'].index(snr), pulse]
        fs = metadata['sample_rate_hz']
        details.update(channel=channel, requested_snr_db=snr, sample_rate_hz=fs,
                       source_pulse_id=metadata['source_pulse_indices'][pulse], kind='reduced')
        caption = f'{channel} | pulse {pulse} | requested SNR {snr:+g} dB'
    elif array.ndim == 3 and array.shape[0] == 3:
        if not 0 <= pulse < array.shape[1]:
            raise ValueError(f'{path}: pulse index out of range')
        row = array[CHANNELS.index(channel), pulse]
        fs = FS
        details.update(channel=channel, sample_rate_hz=fs, kind='raw')
        caption = f'{channel} | pulse {pulse} | raw capture (no added noise)'
    else:
        raise ValueError(f'{path}: unsupported shape {array.shape}')
    return row[0::2].astype(np.float32) + 1j * row[1::2].astype(np.float32), fs, caption, details


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('plots/modulations'))
    p.add_argument('--snr', type=int, default=30, help='Requested SNR for reduced files; ignored for raw files')
    p.add_argument('--pulse', type=int, default=0)
    p.add_argument('--channel', choices=CHANNELS, default='ADC0')
    p.add_argument('--group-by', choices=['family', 'subtype'], default='family', help='Use subtype to show Barker variants separately')
    p.add_argument('--source', choices=['auto', 'raw', 'reduced'], default='auto', help='Auto prefers a reduced example for each group')
    p.add_argument('--nfft', type=int, default=256)
    p.add_argument('--dynamic-range', type=float, default=60)
    p.add_argument('--time-window', type=float, nargs=2, metavar=('START_US', 'END_US'), help='Zoom all panels to this interval')
    args = p.parse_args()
    if args.pulse < 0 or not 4 <= args.nfft <= 16800 or args.nfft % 2 or args.dynamic_range <= 0:
        p.error('Need nonnegative pulse, even nfft in 4..16800 and positive dynamic range')
    if args.time_window and not 0 <= args.time_window[0] < args.time_window[1] <= 140:
        p.error('Time window must lie within 0..140 microseconds')
    selected = {}
    for path in sorted(args.data.glob('*.npy')):
        if path.name.endswith('.partial.npy'):
            continue
        reduced = path.stem.endswith(('_single_channel', '_degraded'))
        if args.source == 'raw' and reduced or args.source == 'reduced' and not reduced:
            continue
        label = subtype(path)
        family = 'barker' if label.startswith('barker') else label
        if family not in FAMILIES:
            print(f'Skipping unknown modulation: {path.name}')
            continue
        group = family if args.group_by == 'family' else label
        if group not in selected or (reduced and not selected[group][1]):
            selected[group] = (path, reduced)
    if not selected:
        p.error('No recognized raw or reduced waveform files found')
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(args.output.resolve() / '.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    report = []
    for label, (path, _) in sorted(selected.items()):
        iq, fs, caption, details = load_example(path, args.pulse, args.snr, args.channel)
        t = np.arange(iq.size) / fs * 1e6
        times, frequencies, db = spectrogram(iq, fs, args.nfft)
        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, layout='constrained')
        fig.suptitle(f'{label.upper()} | {caption}\n{path.name}', fontsize=12)
        axes[0].plot(t, iq.real, label='I', lw=.6)
        axes[0].plot(t, iq.imag, label='Q', lw=.6, alpha=.75)
        axes[0].set_ylabel('I / Q (ADC units)')
        axes[0].legend(loc='upper right')
        axes[1].plot(t, np.abs(iq), color='darkslateblue', lw=.8)
        axes[1].set_ylabel('Magnitude (ADC units)')
        mesh = axes[2].pcolormesh(times * 1e6, frequencies / 1e6, db, shading='nearest',
                                  cmap='magma', vmin=-args.dynamic_range, vmax=0)
        axes[2].set_ylabel('Baseband frequency (MHz)')
        axes[2].set_xlabel('Time from capture start (microseconds)')
        fig.colorbar(mesh, ax=axes[2], label='Power (dB relative to example peak)')
        axes[2].set_xlim(args.time_window or (0, iq.size / fs * 1e6))
        for ax in axes[:2]:
            ax.grid(alpha=.2)
        target = args.output / f'{label}_{args.channel}_pulse{args.pulse}_{"raw" if details["kind"] == "raw" else str(args.snr)+"dB"}.png'
        fig.savefig(target, dpi=160)
        plt.close(fig)
        details.update(group=label, plot=target.name, nfft=args.nfft, time_window_us=args.time_window)
        report.append(details)
        print(target, flush=True)
    missing = sorted(FAMILIES - {'barker' if r['subtype'].startswith('barker') else r['subtype'] for r in report})
    (args.output / 'plot_manifest.json').write_text(json.dumps(dict(examples=report, missing_families=missing), indent=2))
    if missing:
        print('Not available in this folder: ' + ', '.join(missing))


if __name__ == '__main__':
    main()
