# train_diffusion_rhcp.py
import os
import math
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import VOCAB_SIZE, SEQ_LEN, REST_TOKEN
from models import RiffVAE, LatentDiffusionMLP, LatentDiffusionTrainer
from dataset_rhcp import RHCPTokensDataset  # 우리가 만든 RHCP dataset


# =========================
# 경로/하이퍼파라미터 (여기만 보면 됨)
# =========================
RHCP_MIDI_DIR = "midi_riffs"

VAE_FINETUNED_PATH = "riff_vae_rhcp.pth"          # ✅ finetune 결과
DIFF_OUT_PATH = "latent_diffusion_rhcp.pth"       # ✅ 새 diffusion 저장

Z_DIM = 32
T = 200

EPOCHS = 20

BATCH_SIZE = 64
LR = 2e-4
GRAD_CLIP = 1.0

# diffusion 학습 안정화용
USE_MU_ONLY = True   # True면 z = mu (결정적)로 사용 -> 더 안정적
SEED = 42
# =========================


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def encode_to_z(vae: RiffVAE, x: torch.Tensor) -> torch.Tensor:
    """
    RHCP 토큰 x -> VAE latent z로 변환
    RiffVAE forward(x, x_in)에서 mu, logvar를 뽑아 z를 만든다.
    """
    B, L = x.shape
    rest = torch.zeros((B, 1), dtype=x.dtype, device=x.device)
    x_in = torch.cat([rest, x[:, :-1]], dim=1)

    out = vae(x, x_in)
    logits, mu, logvar = out[0], out[1], out[2]

    if USE_MU_ONLY:
        return mu
    else:
        eps = torch.randn_like(mu)
        z = mu + torch.exp(0.5 * logvar) * eps
        return z


def ddpm_loss(model, trainer: LatentDiffusionTrainer, z0: torch.Tensor):
    """
    표준 DDPM loss: eps 예측 MSE
    z_t = sqrt(alpha_bar[t])*z0 + sqrt(1-alpha_bar[t])*eps
    """
    device = z0.device
    B = z0.size(0)

    t = torch.randint(0, trainer.T, (B,), device=device, dtype=torch.long)
    eps = torch.randn_like(z0)

    alpha_bar = trainer.alpha_bar[t].unsqueeze(1)  # (B,1)
    zt = torch.sqrt(alpha_bar) * z0 + torch.sqrt(1.0 - alpha_bar) * eps

    eps_hat = model(zt, t)
    return F.mse_loss(eps_hat, eps)


def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device)

    # -------------------------
    # 데이터: RHCP MIDI only
    # -------------------------
    ds = RHCPTokensDataset(RHCP_MIDI_DIR, seq_len=SEQ_LEN)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    print("Loaded RHCP samples:", len(ds))

    # -------------------------
    # VAE: finetuned 로드 (고정)
    # -------------------------
    vae = RiffVAE(vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN, z_dim=Z_DIM).to(device)
    vae.load_state_dict(torch.load(VAE_FINETUNED_PATH, map_location=device))
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    print("Loaded finetuned VAE:", VAE_FINETUNED_PATH)

    # -------------------------
    # Diffusion 모델/스케줄
    # -------------------------
    diff = LatentDiffusionMLP(z_dim=Z_DIM).to(device)
    trainer = LatentDiffusionTrainer(diff, T=T, z_dim=Z_DIM, device=device)
    opt = torch.optim.Adam(diff.parameters(), lr=LR)

    # -------------------------
    # Train
    # -------------------------
    best = float("inf")

    for ep in range(EPOCHS):
        diff.train()
        total = 0.0
        steps = 0

        for x in dl:
            x = x.to(device)

            # 안전: 길이 보정
            if x.shape[1] != SEQ_LEN:
                if x.shape[1] > SEQ_LEN:
                    x = x[:, :SEQ_LEN]
                else:
                    x = F.pad(x, (0, SEQ_LEN - x.shape[1]), value=REST_TOKEN)

            with torch.no_grad():
                z0 = encode_to_z(vae, x)

            loss = ddpm_loss(diff, trainer, z0)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diff.parameters(), GRAD_CLIP)
            opt.step()

            total += float(loss.item())
            steps += 1

        avg = total / max(1, steps)
        print(f"[Diffusion RHCP] Epoch {ep+1}/{EPOCHS} | loss={avg:.6f}")

        if avg < best:
            best = avg
            torch.save(diff.state_dict(), DIFF_OUT_PATH)
            print("  -> saved best:", DIFF_OUT_PATH)

    print("Done. best_loss=", best)
    print("Saved diffusion to:", os.path.abspath(DIFF_OUT_PATH))


if __name__ == "__main__":
    main()
