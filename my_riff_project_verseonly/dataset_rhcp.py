# dataset_rhcp.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset

from midi_utils import midi_to_token_seq
from config import SEQ_LEN, REST_TOKEN


class RHCPTokensDataset(Dataset):
    """
    RHCP verse MIDI → 토큰 시퀀스 Dataset
    각 MIDI는 1개의 (SEQ_LEN,) 토큰 시퀀스를 반환
    """
    def __init__(self, midi_dir: str, seq_len: int = SEQ_LEN):
        self.midi_dir = midi_dir
        self.seq_len = seq_len

        self.paths = [
            os.path.join(midi_dir, f)
            for f in os.listdir(midi_dir)
            if f.lower().endswith(".mid")
        ]
        self.paths.sort()

        if len(self.paths) == 0:
            raise RuntimeError(f"No MIDI files found in {midi_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        midi_path = self.paths[idx]

        tokens = midi_to_token_seq(midi_path)

        # 길이 보정
        if len(tokens) > self.seq_len:
            tokens = tokens[:self.seq_len]
        elif len(tokens) < self.seq_len:
            pad = np.full(self.seq_len - len(tokens), REST_TOKEN, dtype=np.int64)
            tokens = np.concatenate([tokens, pad])

        return torch.from_numpy(tokens.astype(np.int64))
