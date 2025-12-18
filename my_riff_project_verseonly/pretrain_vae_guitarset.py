# pretrain_vae_guitarset.py
import os
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from dataset_tokens import NpyTokenDataset

# 네 프로젝트 파일들
from config import VOCAB_SIZE, SEQ_LEN
from models import RiffVAE


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_kl(mu, logvar):
    # KL(q(z|x) || N(0,1)) for diagonal Gaussian
    # 평균 배치 기준 scalar
    return 0.5 * torch.mean(torch.sum(torch.exp(logvar) + mu**2 - 1.0 - logvar, dim=1))


def forward_vae(vae: RiffVAE, x: torch.Tensor):
    """
    RiffVAE는 forward(x, x_in)을 요구함
    x_in은 teacher forcing용 shift 입력
    """
    # x: (B, L)
    B, L = x.shape

    # x_in = [REST, x[:-1]]
    rest = torch.zeros((B, 1), dtype=x.dtype, device=x.device)
    x_in = torch.cat([rest, x[:, :-1]], dim=1)

    out = vae(x, x_in)

    if isinstance(out, (list, tuple)) and len(out) >= 3:
        logits, mu, logvar = out[0], out[1], out[2]
        return logits, mu, logvar

    raise RuntimeError("VAE forward() must return (logits, mu, logvar)")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token_dir", default="guitarset_tokens_4bars", help="GuitarSet 토큰(.npy) 폴더")
    ap.add_argument("--out_path", default="riff_vae_pretrain.pth", help="저장할 VAE 가중치 파일명")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--kl_beta", type=float, default=0.1, help="KL 가중치 (beta-VAE 느낌)")
    ap.add_argument("--val_ratio", type=float, default=0.05)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--z_dim", type=int, default=32)
    args = ap.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device)

    # Dataset
    ds = NpyTokenDataset(args.token_dir)

    # 토큰 값이 VOCAB_SIZE 범위 안인지 체크 (중요)
    # 네 토큰 정의가 "MIDI pitch 직접 사용"이면 보통 VOCAB_SIZE >= 128이어야 함
    max_tok = 0
    for i in range(min(200, len(ds))):
        max_tok = max(max_tok, int(ds[i].max().item()))
    if max_tok >= VOCAB_SIZE:
        raise ValueError(
            f"[ERROR] token max={max_tok} >= VOCAB_SIZE={VOCAB_SIZE}\n"
            f"-> VOCAB_SIZE를 128 이상으로 올리거나, pitch를 vocab 범위로 매핑해야 함."
        )

    # Split train/val
    n_total = len(ds)
    n_val = max(1, int(n_total * args.val_ratio))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])
    print(f"Loaded tokens: total={n_total}, train={n_train}, val={n_val}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, drop_last=False)

    # Model
    vae = RiffVAE(vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN, z_dim=args.z_dim).to(device)
    opt = torch.optim.Adam(vae.parameters(), lr=args.lr)

    best_val = float("inf")

    for ep in range(args.epochs):
        vae.train()
        total_loss = 0.0
        total_rec = 0.0
        total_kl = 0.0
        steps = 0

        for x in train_loader:
            x = x.to(device)
            # 길이 강제(혹시 이상한 npy가 섞였을 때)
            if x.shape[1] != SEQ_LEN:
                x = x[:, :SEQ_LEN] if x.shape[1] > SEQ_LEN else F.pad(x, (0, SEQ_LEN - x.shape[1]), value=0)

            logits, mu, logvar = forward_vae(vae, x)

            # logits: (B, L, V) 가정
            if logits.dim() != 3:
                raise RuntimeError(f"logits must be (B,L,V). got shape={tuple(logits.shape)}")

            # Reconstruction (CrossEntropy)
            rec = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE),
                x.reshape(-1),
                reduction="mean"
            )
            kl = compute_kl(mu, logvar)
            loss = rec + args.kl_beta * kl

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            opt.step()

            total_loss += float(loss.item())
            total_rec += float(rec.item())
            total_kl += float(kl.item())
            steps += 1

        # division by zero 방지
        if steps == 0:
            raise RuntimeError("No training steps executed (train_loader empty). Check dataset/batch_size.")

        train_loss = total_loss / steps
        train_rec = total_rec / steps
        train_kl = total_kl / steps

        # Validation
        vae.eval()
        v_loss = 0.0
        v_steps = 0
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device)
                if x.shape[1] != SEQ_LEN:
                    x = x[:, :SEQ_LEN] if x.shape[1] > SEQ_LEN else F.pad(x, (0, SEQ_LEN - x.shape[1]), value=0)

                logits, mu, logvar = forward_vae(vae, x)
                rec = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), x.reshape(-1), reduction="mean")
                kl = compute_kl(mu, logvar)
                loss = rec + args.kl_beta * kl

                v_loss += float(loss.item())
                v_steps += 1

        val_loss = v_loss / max(1, v_steps)

        print(f"[Pretrain VAE] Epoch {ep+1}/{args.epochs} | "
              f"train loss={train_loss:.4f} rec={train_rec:.4f} kl={train_kl:.4f} | "
              f"val loss={val_loss:.4f}")

        # Save best
        if val_loss < best_val:
            best_val = val_loss
            torch.save(vae.state_dict(), args.out_path)
            print(f"  -> saved best to {args.out_path}")

    print("Done. Best val loss =", best_val)
    print("VAE checkpoint:", os.path.abspath(args.out_path))


if __name__ == "__main__":
    main()
