# MIDI 폴더 → 토큰 시퀀스 Dataset.
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config import MIDI_DIR, SEQ_LEN
from midi_utils import midi_to_token_seq

class RiffDataset(Dataset):
    def __init__(self, midi_dir=MIDI_DIR, seq_len=SEQ_LEN):
        self.seq_len = seq_len
        paths = sorted(glob.glob(os.path.join(midi_dir, "*.mid")))
        self.data = []

        for p in paths:
            try:
                tokens = midi_to_token_seq(p)
                if len(tokens) == seq_len:
                    self.data.append(tokens)
            except Exception as e:
                print("Skip", p, ":", e)

        if len(self.data) == 0:
            raise RuntimeError("No valid MIDI riffs found in " + midi_dir)

        self.data = np.stack(self.data, axis=0).astype(np.int64)
        print("Loaded", self.data.shape[0], "riffs.")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x = self.data[idx]
        return torch.from_numpy(x).long()


def get_dataloader(batch_size=64, shuffle=True):
    dataset = RiffDataset()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)
    return loader, dataset
