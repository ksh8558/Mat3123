# generate.py
# 리듬은 실행마다 랜덤 1개 선택(strum_pattern_steps=None), 곡 내 고정(keep_rhythm_fixed=True)
# + 첫 마디 리듬만 변주
# + voicing_cycle(1114/2231/2233 등) + hit마다 chord_tones variant 믹스로 "같은 음 반복" 감소

import time
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

            z_t = (1.0 / torch.sqrt(alpha_t)) * (
                z_t - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * eps_hat
            ) + torch.sqrt(beta_t) * noise
    return z_t


@torch.no_grad()
def decode_autoregressive(
    vae: RiffVAE,
    z: torch.Tensor,
    seq_len: int,
    temperature: float = 1.0,
    rest_logit_penalty: float = 2.0,
) -> np.ndarray:
    h = vae.fc_z_to_h(z).unsqueeze(0)
    prev_tok = torch.tensor([[REST_TOKEN]], device=z.device, dtype=torch.long)

    tokens = []
    for _ in range(seq_len):
        emb = vae.emb(prev_tok)
        out, h = vae.decoder_rnn(emb, h)
        logits = vae.fc_out(out.squeeze(1))

        logits[:, REST_TOKEN] -= rest_logit_penalty

        probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
        tok = torch.multinomial(probs, num_samples=1)

        tokens.append(int(tok.item()))
        prev_tok = tok

    return np.array(tokens, dtype=np.int64)


def mutate_tokens_tail(tokens, rng, tail_ratio=0.20, mutate_ratio=0.05, max_shift=1):
    out = tokens.copy()
    L = len(out)
    start = int(L * (1.0 - tail_ratio))
    region = np.arange(L) >= start

    note_mask = (out != REST_TOKEN)
    change = region & note_mask & (rng.random(L) < mutate_ratio)
    if np.any(change):
        shift = rng.integers(-max_shift, max_shift + 1, size=np.sum(change))
        out[change] = np.clip(out[change] + shift, 1, VOCAB_SIZE - 1)
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device)

    num_segments = 8
    out_midi = "verse_to_verse2.mid"
    tempo = 105

    temperature = 1.05
    rest_logit_penalty = 2.2
    max_tries_per_segment = 10

    transpose_semitones = -10
    rest_grace_bars = 1

    # 리듬: 실행마다 랜덤 선택 (곡 내 고정)
    strum_pattern_steps = None
    keep_rhythm_fixed = True
    intro_bar_variation = True
    intro_force_diff_rhythm = True

    # voicing cycle
    voicing_keep_fixed_in_song = True
    voicing_change_prob_per_bar = 0.7
    voicing_cycle_seed = None  # None=매 실행마다 다름

    # hit마다 chord tones variant 섞기
    voicing_variant_mix = True
    triad_bias = 0.85

    hendrix_spice = True
    hendrix_prob = 0.10

    vae = RiffVAE(vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN, z_dim=32).to(device)
    vae.load_state_dict(torch.load(VAE_PATH, map_location=device))
    vae.eval()
    print("Loaded VAE:", VAE_PATH)

    z_dim = 32
    T = 200
    ld_model = LatentDiffusionMLP(z_dim=z_dim).to(device)
    ld_trainer = LatentDiffusionTrainer(ld_model, T=T, z_dim=z_dim, device=device)
    ld_model.load_state_dict(torch.load(DIFF_PATH, map_location=device))
    ld_model.eval()
    print("Loaded diffusion:", DIFF_PATH)

    rng = np.random.default_rng(int(time.time()) % (2**32 - 1))

    all_tokens = []
    prev = None

    for seg in range(num_segments):
        ok = False
        tokens = None

        for _ in range(max_tries_per_segment):
            z = sample_latent(ld_trainer, z_dim=z_dim, T=T, device=device)
            tokens = decode_autoregressive(
                vae, z, SEQ_LEN,
                temperature=temperature,
                rest_logit_penalty=rest_logit_penalty
            )
            if np.any(tokens != REST_TOKEN):
                ok = True
                break

        if not ok:
            print(f"[WARN] segment {seg}: rest만 -> 무음 세그먼트")
            tokens = np.full(SEQ_LEN, REST_TOKEN, dtype=np.int64)

        if prev is not None and rng.random() < 0.55:
            tail = mutate_tokens_tail(prev, rng, tail_ratio=0.18, mutate_ratio=0.05, max_shift=1)
            mix_mask = (rng.random(SEQ_LEN) < 0.14)
            tokens = tokens.copy()
            tokens[mix_mask] = tail[mix_mask]

        all_tokens.append(tokens)
        prev = tokens
        print(f"segment {seg}: rest_ratio={float(np.mean(tokens == REST_TOKEN)):.2f}")

    long_tokens = np.concatenate(all_tokens, axis=0)

    token_seq_to_midi(
        long_tokens,
        out_midi,
        tempo=tempo,
        step_div=4,
        beats_per_bar=4,

        transpose_semitones=transpose_semitones,
        rest_grace_bars=rest_grace_bars,

        strum_pattern_steps=strum_pattern_steps,
        keep_rhythm_fixed=keep_rhythm_fixed,
        intro_bar_variation=intro_bar_variation,
        intro_force_diff_rhythm=intro_force_diff_rhythm,

        voicing_keep_fixed_in_song=voicing_keep_fixed_in_song,
        voicing_change_prob_per_bar=voicing_change_prob_per_bar,
        voicing_cycle_seed=voicing_cycle_seed,

        voicing_variant_mix=voicing_variant_mix,
        triad_bias=triad_bias,

        hendrix_spice=hendrix_spice,
        hendrix_prob=hendrix_prob,

        velocity=86,
        vel_rand=10,
        timing_jitter_ms=5.0,
        overlap_ms=58.0,
    )

    print("Saved:", out_midi)


if __name__ == "__main__":
    main()
