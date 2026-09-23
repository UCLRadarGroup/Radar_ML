"""Score anonymous predictions using organizer or distributed holdout labels."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def score(key_path, submission_path):
    with Path(key_path).open(newline='') as stream:
        key = [r for r in csv.DictReader(stream) if r['split'] == 'test']
    expected = {r['sample_id']: r for r in key}
    if not expected or len(expected) != len(key):
        raise ValueError('Answer key must contain unique test samples')
    classes = sorted({r['label'] for r in key})
    predictions = {}
    with Path(submission_path).open(newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['sample_id', 'predicted_label']:
            raise ValueError('Required columns: sample_id,predicted_label')
        for row in reader:
            identity, label = row['sample_id'], row['predicted_label']
            if identity in predictions:
                raise ValueError(f'Duplicate ID: {identity}')
            if label not in classes:
                raise ValueError(f'Unknown or empty label: {label}')
            predictions[identity] = label
    if predictions.keys() != expected.keys():
        raise ValueError(f'ID mismatch: missing={len(expected.keys()-predictions.keys())}, '
                         f'extra={len(predictions.keys()-expected.keys())}')
    results = []
    for snr in sorted({int(r['requested_snr_db']) for r in key}, reverse=True):
        matrix = np.zeros((len(classes), len(classes)), dtype=int)
        for row in key:
            if int(row['requested_snr_db']) == snr:
                matrix[classes.index(row['label']), classes.index(predictions[row['sample_id']])] += 1
        if np.any(matrix.sum(axis=1) == 0):
            raise ValueError('Every SNR must contain every class')
        results.append(dict(requested_snr_db=snr, balanced_accuracy=float(np.mean(matrix.diagonal()/matrix.sum(axis=1))),
                            accuracy=float(matrix.trace()/matrix.sum()), confusion_matrix=matrix.tolist()))
    return dict(classes=classes, samples=len(key),
                mean_balanced_accuracy=float(np.mean([r['balanced_accuracy'] for r in results])), per_snr=results)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--key', required=True)
    p.add_argument('--submission', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    report = score(args.key, args.submission)
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(f'Mean balanced accuracy: {report["mean_balanced_accuracy"]:.4f}')
