"""Participant utility: load a pulse and index information. No ML method included."""
import csv
from pathlib import Path
import numpy as np


class Samples:
    def __init__(self, split_directory):
        self.root = Path(split_directory)
        with (self.root / 'index.csv').open(newline='') as stream:
            self.rows = list(csv.DictReader(stream))
        self._filename = None
        self._array = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        info = dict(self.rows[index])
        if self._filename != info['file']:
            self._array = np.load(self.root / info['file'], mmap_mode='r', allow_pickle=False)
            self._filename = info['file']
        row = self._array[int(info['row'])]
        iq = row[0::2].astype(np.float32) + 1j * row[1::2].astype(np.float32)
        return iq, info
