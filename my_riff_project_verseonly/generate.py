# generate.py
# 학습된 모델로 새 리프 생성 → (여러 2마디를 이어붙여) MIDI로 저장.

# generate.py
import torch
import numpy as np

from config import VOCAB_SIZE, SEQ_LEN, VAE_PATH, DIFF_PATH, REST_TOKEN
from models import RiffVAE, LatentDiffusionMLP, LatentDiffusionTrainer
from midi_utils import token_seq_to_midi


def sample_latent(ld_trainer, z_dim, T, device):
    model = ld_trainer.model
    betas = ld_trainer.betas
    alphas = ld_trainer.alphas
    alpha_bar = ld_trainer.alpha_bar

    z_t = torch.randn(1, z_dim, device=device)
    with torch.no_grad():
        for t_ in reversed(range(T)):
            t = torch.tensor([t_], device=device, dtype=torch.long)
            eps_hat = model(z_t, t)

            alpha_t = alphas[t_]
            beta_t = betas[t_]
            alpha_bar_t = alpha_bar[t_]

            noise = torch.randn_like(z_t) if t_ > 0 else torch.zeros_like(z_t)

            z_t = (1.0 / torch.sqrt(alpha_t)) * \
                  (z_t - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * eps_hat) + \
                  torch.sqrt(beta_t) * noise
    return z_t


@torch.no_grad()
def decode_autoregressive(vae: RiffVAE, z: torch.Tensor,
                          seq_len: int, temperature: float = 1.0,
                          rest_logit_penalty: float = 2.0) -> np.ndarray:
    """
    GRU 디코더를 1-step씩 굴리면서 토큰 생성.
    rest(0)로 붕괴되는 걸 막기 위해 rest logit을 깎음.
    """
    # hidden init from z
    h = vae.fc_z_to_h(z).unsqueeze(0)  # (1, B=1, hidden)
    prev_tok = torch.tensor([[REST_TOKEN]], device=z.device, dtype=torch.long)  # (1,1)

    tokens = []
    for _ in range(seq_len):
        emb = vae.emb(prev_tok)                 # (1,1,emb_dim)
        out, h = vae.decoder_rnn(emb, h)        # out: (1,1,hidden)
        logits = vae.fc_out(out.squeeze(1))     # (1,vocab)

        # rest 억제
        logits[:, REST_TOKEN] -= rest_logit_penalty

        # temperature sampling
        probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
        tok = torch.multinomial(probs, num_samples=1)  # (1,1)

        tokens.append(int(tok.item()))
        prev_tok = tok

    return np.array(tokens, dtype=np.int64)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device)

    # ===== 조절 파라미터 =====
    num_segments = 8
    out_midi = "generated_riff_long.mid"
    tempo = 120.0

    temperature = 1.2           # 1.0~1.3 추천
    rest_logit_penalty = 2.5    # 1.0~4.0 추천 (클수록 rest 덜 나옴)
    max_tries_per_segment = 10  # 전부 rest면 z 다시 뽑는 횟수
    # =======================

    # VAE
    vae = RiffVAE(vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN, z_dim=32).to(device)
    vae.load_state_dict(torch.load(VAE_PATH, map_location=device))
    vae.eval()
    print("Loaded VAE from", VAE_PATH)

    # Diffusion
    z_dim = 32
    T = 200
    ld_model = LatentDiffusionMLP(z_dim=z_dim).to(device)
    ld_trainer = LatentDiffusionTrainer(ld_model, T=T, z_dim=z_dim, device=device)
    ld_model.load_state_dict(torch.load(DIFF_PATH, map_location=device))
    ld_model.eval()
    print("Loaded diffusion model from", DIFF_PATH)

    all_tokens = []

    for seg in range(num_segments):
        ok = False
        for _ in range(max_tries_per_segment):
            z = sample_latent(ld_trainer, z_dim=z_dim, T=T, device=device)
            tokens = decode_autoregressive(
                vae, z, SEQ_LEN,
                temperature=temperature,
                rest_logit_penalty=rest_logit_penalty
            )
            # 전부 rest면 실패 처리
            if np.any(tokens != REST_TOKEN):
                ok = True
                break

        if not ok:
            print(f"[WARN] segment {seg}: 계속 rest만 나와서 그냥 저장(무음)합니다.")
        all_tokens.append(tokens)

        rest_ratio = float(np.mean(tokens == REST_TOKEN))
        print(f"segment {seg}: rest_ratio={rest_ratio:.2f}")

    long_tokens = np.concatenate(all_tokens, axis=0)
    token_seq_to_midi(long_tokens, out_midi, tempo=tempo)
    print(f"Generated riff saved to {out_midi} (segments={num_segments}, total_steps={len(long_tokens)})")


if __name__ == "__main__":
    main()
