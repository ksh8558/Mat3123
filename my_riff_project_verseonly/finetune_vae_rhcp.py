# finetune_vae_rhcp.py
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import (
    VOCAB_SIZE,
    SEQ_LEN,
    REST_TOKEN,
)

from models import RiffVAE
from dataset_rhcp import RHCPTokensDataset


# ===============================
# 설정 (경로/하이퍼파라미터)
# ===============================
RHCP_MIDI_DIR = "midi_riffs" # RHCP verse MIDI 폴더
PRETRAIN_PATH = "riff_vae_pretrain.pth"  # GuitarSet pretrain 결과
OUT_PATH = "riff_vae_rhcp.pth"            # finetune 결과

EPOCHS = 20
BATCH_SIZE = 32
LR = 1e-4
KL_BETA = 0.1
GRAD_CLIP = 1.0
SEED = 42


# ===============================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_kl(mu, logvar):
    return 0.5 * torch.mean(
        torch.sum(torch.exp(logvar) + mu**2 - 1.0 - logvar, dim=1)
    )


def forward_vae(vae: RiffVAE, x: torch.Tensor):
    """
    RiffVAE forward(x, x_in) 구조 대응
    """
    B, L = x.shape
    rest = torch.zeros((B, 1), dtype=x.dtype, device=x.device)
    x_in = torch.cat([rest, x[:, :-1]], dim=1)

    out = vae(x, x_in)
    logits, mu, logvar = out[0], out[1], out[2]
    return logits, mu, logvar


# ===============================
def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device)

    # ---------------------------
    # Dataset (RHCP MIDI only)
    # ---------------------------
    dataset = RHCPTokensDataset(
        midi_dir=RHCP_MIDI_DIR,
        seq_len=SEQ_LEN,
)
    

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )

    print(f"Loaded RHCP MIDI samples: {len(dataset)}")

    # ---------------------------
    # Model
    # ---------------------------
    vae = RiffVAE(
        vocab_size=VOCAB_SIZE,
        seq_len=SEQ_LEN,
        z_dim=32,
    ).to(device)

    # 🔑 GuitarSet pretrain 로드
    vae.load_state_dict(
        torch.load(PRETRAIN_PATH, map_location=device),
        strict=True
    )
    print("Loaded pretrain VAE from", PRETRAIN_PATH)

    opt = torch.optim.Adam(vae.parameters(), lr=LR)

    # ---------------------------
    # Finetune
    # ---------------------------
    for ep in range(EPOCHS):
        vae.train()
        total_loss = 0.0
        total_rec = 0.0
        total_kl = 0.0
        steps = 0

        for x in loader:
            x = x.to(device)

            # 길이 보정 (안전)
            if x.shape[1] != SEQ_LEN:
                if x.shape[1] > SEQ_LEN:
                    x = x[:, :SEQ_LEN]
                else:
                    x = F.pad(x, (0, SEQ_LEN - x.shape[1]), value=REST_TOKEN)

            logits, mu, logvar = forward_vae(vae, x)

            rec = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE),
                x.reshape(-1),
                reduction="mean"
            )
            kl = compute_kl(mu, logvar)
            loss = rec + KL_BETA * kl

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), GRAD_CLIP)
            opt.step()

            total_loss += loss.item()
            total_rec += rec.item()
            total_kl += kl.item()
            steps += 1

        print(
            f"[Finetune VAE] Epoch {ep+1}/{EPOCHS} | "
            f"loss={total_loss/steps:.4f} "
            f"rec={total_rec/steps:.4f} "
            f"kl={total_kl/steps:.4f}"
        )

    # ---------------------------
    # Save
    # ---------------------------
    torch.save(vae.state_dict(), OUT_PATH)
    print("Saved finetuned VAE to", OUT_PATH)


if __name__ == "__main__":
    main()
