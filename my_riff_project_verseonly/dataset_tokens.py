# dataset_tokens.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset


class NpyTokenDataset(Dataset):
    """
    폴더 안의 *.npy(길이=SEQ_LEN 토큰 시퀀스)를 읽어오는 Dataset
    각 npy는 1D array (SEQ_LEN,) 이어야 함
    """
    def __init__(self, token_dir: str):
        self.token_dir = token_dir
        self.paths = [
            os.path.join(token_dir, f)
            for f in os.listdir(token_dir)
            if f.lower().endswith(".npy")
        ]
        self.paths.sort()

        if len(self.paths) == 0:
            raise FileNotFoundError(f"No .npy files found in: {token_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        x = np.load(self.paths[idx]).astype(np.int64)

        # (SEQ_LEN,) 형태 강제
        if x.ndim != 1:
            x = x.reshape(-1)

        return torch.from_numpy(x)


