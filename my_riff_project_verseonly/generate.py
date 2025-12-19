# generate.py
# Latent Diffusion + VAE로 tokens 생성
# -> Frusciante Guitar Engine으로 Verse(Frus) + Chorus(코러스) 구성 MIDI 생성
#
# 포함:
# - 코드 진행(progressions) 랜덤 샘플링 + transpose
# - 마디마다 hit_pattern 랜덤
# - 마디마다 voicing 랜덤(Frusciante 스타일) + mutate
# - Verse는 숨(REST 마스크) + ghost 많음
# - Chorus는 빽빽(use_token_mask=False) + money_boost

import time
import random
import torch
import numpy as np

from config import VOCAB_SIZE, SEQ_LEN, VAE_PATH, DIFF_PATH, REST_TOKEN
from models import RiffVAE, LatentDiffusionMLP, LatentDiffusionTrainer
from midi_utils import token_seq_to_midi_strum_guitar_voicing


# =========================================================
# 1) Progression package (major/minor only) + transpose
# =========================================================
PROG_PACKS = {
    "mixolydian_funk": [
        (("A", "G", "D", "A"), 10),
        (("E", "D", "A", "E"), 9),
        (("G", "F", "C", "G"), 8),
        (("D", "C", "G", "D"), 8),
        (("B", "A", "E", "B"), 7),
        (("C", "Bb", "F", "C"), 7),
        (("F", "Eb", "Bb", "F"), 6),
        (("A", "G", "A", "D"), 6),
        (("E", "D", "E", "A"), 6),
    ],
    "minor_modal": [
        (("Am", "G", "F", "G"), 10),
        (("Em", "D", "C", "D"), 9),
        (("Dm", "C", "Bb", "C"), 7),
        (("Bm", "A", "G", "A"), 7),
        (("Cm", "Bb", "Ab", "Bb"), 6),
        (("Am", "G", "Dm", "F"), 5),
        (("Em", "D", "Am", "C"), 5),
    ],
    "major_pop": [
        (("C", "G", "Am", "F"), 10),
        (("G", "D", "Em", "C"), 9),
        (("D", "A", "Bm", "G"), 9),
        (("A", "E", "F#m", "D"), 8),
        (("F", "C", "Dm", "Bb"), 7),
        (("E", "B", "C#m", "A"), 7),
    ],
}

NOTE_ORDER_SHARP = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
ENHARMONIC_FLAT = {"Db":"C#","Eb":"D#","Gb":"F#","Ab":"G#","Bb":"A#"}

def _norm_root(root: str) -> str:
    return ENHARMONIC_FLAT.get(root, root)

def _weighted_choice(items_with_w):
    items = [x for x, _ in items_with_w]
    ws = [w for _, w in items_with_w]
    return random.choices(items, weights=ws, k=1)[0]

def transpose_chord(ch: str, semis: int) -> str:
    ch = ch.strip()
    if len(ch) >= 2 and ch[1] in ("#", "b"):
        root = ch[:2]
        rest = ch[2:]
    else:
        root = ch[:1]
        rest = ch[1:]
    minor = rest.startswith("m")

    root = _norm_root(root)
    if root not in NOTE_ORDER_SHARP:
        raise ValueError(f"Bad chord root: {ch}")

    i = NOTE_ORDER_SHARP.index(root)
    j = (i + semis) % 12
    return NOTE_ORDER_SHARP[j] + ("m" if minor else "")

def sample_progression(style: str, transpose=True, transpose_range=(-3, 3)) -> tuple[str, ...]:
    base = _weighted_choice(PROG_PACKS[style])
    if transpose:
        semis = random.randint(transpose_range[0], transpose_range[1])
        base = tuple(transpose_chord(c, semis) for c in base)
    return base


# =========================================================
# 2) Frusciante voicing per bar package (random + mutate)
#    수정 핵심: "한 마디 안 반복/변주(1114,2231,2233)" + "2마디-2마디 반복 후 변주"
# =========================================================
BASE_VOICING = {
    "core": {
        "patterns": ["1212","2121","1112","1121","1211","2112","1111"],
        "weights":  {"1212":8,"2121":5,"1112":5,"1121":4,"1211":4,"2112":3,"1111":2},
    },
    "funk": {
        "patterns": ["1212","2121","1221","2112","1211","1121"],
        "weights":  {"1212":7,"2121":6,"1221":4,"2112":4,"1211":3,"1121":3},
    },
    "melodic": {
        "patterns": ["1111","1121","1211","1112"],
        "weights":  {"1111":7,"1121":5,"1211":4,"1112":4},
    }
}

# 엔딩에 많이 쓰는 "마지막 한 방" (money note)
ENDING_POOL = ["1113","1213","1123","1114"]
ENDING_W = {"1113":7,"1213":4,"1123":3,"1114":2}

def _weighted_pattern(pool, wdict):
    ws = [wdict.get(p, 1) for p in pool]
    return random.choices(pool, weights=ws, k=1)[0]

# --- 추가: 프루시안테스러운 "마디 내부 사이클" 묶음 ---
# 숫자는 midi_utils 쪽에서 voicing index(또는 변주 레벨)로 쓰인다고 가정.
# (기존 코드도 1/2/3/4를 쓰고 있으니 호환)
BAR_MOTIFS = {
    # verse용: 1114 / 2231 / 2233 류를 자주
    "verse": [
        ("1114", 12),
        ("2231", 10),
        ("2233", 8),
        ("1113", 8),
        ("1123", 6),
        ("1213", 5),
        ("1112", 5),
        ("1121", 4),
        ("1212", 4),
        ("2112", 3),
    ],
    # chorus용: 펑키하게 왕복/스윙 느낌
    "chorus": [
        ("1212", 10),
        ("2121", 9),
        ("1221", 7),
        ("2112", 7),
        ("1213", 6),
        ("1123", 5),
        ("1113", 4),
        ("1114", 3),
    ],
}

def _choose_motif(section: str):
    items = BAR_MOTIFS.get(section, BAR_MOTIFS["verse"])
    pats = [p for p, _ in items]
    ws = [w for _, w in items]
    return random.choices(pats, weights=ws, k=1)[0]

def _mutate_pattern_fru(
    pat: str,
    *,
    p_keep_front=0.85,     # 앞(1~2칸)은 거의 유지
    p_last_change=0.70,    # 마지막은 자주 바꿈
    p_swap12=0.10,
    p_add3_last=0.30,
    p_add4_last=0.08,
):
    """
    Frusciante rule-of-thumb:
    - 같은 느낌 유지(앞부분 유지)
    - 마지막/후반만 살짝 바꿔서 "같은데 변주" 만들기
    """
    s = list(pat)

    # 1) 앞 2칸은 웬만하면 유지, 아니면 1<->2 swap 정도만
    if random.random() > p_keep_front:
        if random.random() < p_swap12:
            for i in range(min(2, len(s))):
                if s[i] == '1': s[i] = '2'
                elif s[i] == '2': s[i] = '1'

    # 2) 뒤 2칸은 변주 확률 높게: 1/2를 조금 흔들고, 마지막에 3/4를 꽂기도
    for i in range(max(2, len(s)-2), len(s)):
        if random.random() < 0.25:
            if s[i] == '1': s[i] = '2'
            elif s[i] == '2': s[i] = '1'

    # 3) 마지막에 money note(3/4) 확률적으로
    if random.random() < p_last_change:
        r = random.random()
        if r < p_add4_last:
            s[-1] = '4'
        elif r < (p_add4_last + p_add3_last):
            s[-1] = '3'
        # else: keep (혹은 위에서 바뀐 1/2 유지)

    return "".join(s)

def _is_ending_bar(bar_idx: int, end_every: int | None):
    return (end_every is not None and (bar_idx % end_every) == end_every - 1)

def sample_voicing_per_bar(
    n_bars: int,
    style="core",
    end_every=4,
    no_repeat=True,
    mutate=True,
    section="verse",
    motif_repeat_unit=2,     # 2마디 단위로 motif를 만들고 다음 2마디에서 변주 반복
):
    """
    기존: 매 마디 독립 샘플링 -> 너무 랜덤
    수정: 2마디 motif를 만들고, 다음 2마디는 '거의 같은데 마지막만 변주'로 반복(verse 구조)
    """
    pool = BASE_VOICING[style]["patterns"]
    w = BASE_VOICING[style]["weights"]

    out = []
    last = None

    # 2마디 motif를 저장해뒀다가, 다음 블록에서 변주 반복
    last_block = None

    bar = 0
    while bar < n_bars:
        # 블록 길이(기본 2마디, 남은 마디 수에 따라 조절)
        block_len = min(motif_repeat_unit, n_bars - bar)

        # 1) 블록 생성
        block = []

        # "repeat block"을 만들 차례인지:
        # 짝수 블록: 새로 motif 생성
        # 홀수 블록: 직전 motif를 기반으로 변주
        block_id = bar // motif_repeat_unit
        is_repeat_block = (last_block is not None and (block_id % 2 == 1))

        for i in range(block_len):
            bi = bar + i
            ending = _is_ending_bar(bi, end_every)

            if ending:
                pat = _weighted_pattern(ENDING_POOL, ENDING_W)
            else:
                if is_repeat_block:
                    # 직전 block의 i번째를 기반으로 "같은데 변주"
                    base_pat = last_block[i] if i < len(last_block) else last_block[-1]
                    pat = _mutate_pattern_fru(base_pat)
                else:
                    # 새 motif: (1) 프루시안테 motif를 우선 시도 (2) 부족하면 기존 pool로 fallback
                    if random.random() < 0.72:
                        pat = _choose_motif(section)
                    else:
                        pat = _weighted_pattern(pool, w)

                    if mutate:
                        # 너무 망가뜨리지 않게 프루시안테식 변주만
                        pat = _mutate_pattern_fru(pat)

            # no_repeat: 직전과 완전 동일이면 다시 뽑기(짧게)
            if no_repeat and last is not None:
                tries = 0
                while pat == last and tries < 8:
                    if ending:
                        pat = _weighted_pattern(ENDING_POOL, ENDING_W)
                    else:
                        pat = _choose_motif(section) if random.random() < 0.75 else _weighted_pattern(pool, w)
                        if mutate:
                            pat = _mutate_pattern_fru(pat)
                    tries += 1

            block.append(pat)
            last = pat

        out.extend(block)

        # 새 motif 블록이면 저장 (다음 블록에서 repeat 변주에 사용)
        if not is_repeat_block:
            last_block = block

        bar += block_len

    return tuple(out)


# =========================================================
# 3) Token generation (LD + VAE)
# =========================================================
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
def decode_autoregressive(vae: RiffVAE, z: torch.Tensor, seq_len: int,
                          temperature=1.0, rest_logit_penalty=2.0) -> np.ndarray:
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

def build_tokens_for_steps(vae, ld_trainer, total_steps, device,
                           temperature, rest_logit_penalty,
                           max_tries=10):
    chunks = []
    z_dim = ld_trainer.z_dim
    T = ld_trainer.T

    while sum(len(x) for x in chunks) < total_steps:
        ok = False
        tok = None
        for _ in range(max_tries):
            z = sample_latent(ld_trainer, z_dim=z_dim, T=T, device=device)
            tok = decode_autoregressive(vae, z, SEQ_LEN, temperature, rest_logit_penalty)
            if np.any(tok != REST_TOKEN):
                ok = True
                break
        if not ok:
            tok = np.full(SEQ_LEN, REST_TOKEN, dtype=np.int64)
        chunks.append(tok)

    return np.concatenate(chunks)[:total_steps]


def concat_two_midis(mid_a_path: str, mid_b_path: str, out_path: str, tempo: float):
    import pretty_midi
    a = pretty_midi.PrettyMIDI(mid_a_path)
    b = pretty_midi.PrettyMIDI(mid_b_path)

    out = pretty_midi.PrettyMIDI(initial_tempo=float(tempo))

    for inst in a.instruments:
        out.instruments.append(inst)

    shift = a.get_end_time()
    for inst in b.instruments:
        for n in inst.notes:
            n.start += shift
            n.end += shift
        out.instruments.append(inst)

    out.write(out_path)


# =========================================================
# 4) Main: Verse -> Chorus render
# =========================================================
def main():
    # reproducibility를 원하면 아래 seed를 고정하면 됨
    # random.seed(42); np.random.seed(42); torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device)

    # ---------- structure ----------
    tempo = 105
    step_div = 2
    beats_per_bar = 4

    bars_verse = 8
    bars_chorus = 8

    out_midi = "verse_to_verse2.mid"

    steps_per_bar = step_div * beats_per_bar
    verse_steps = bars_verse * steps_per_bar
    chorus_steps = bars_chorus * steps_per_bar

    # ---------- choose chord progressions ----------
    chord_prog_verse = sample_progression("mixolydian_funk", transpose=True)
    chord_prog_chorus = sample_progression("major_pop", transpose=True)

    # ---------- hit pattern pools ----------
    hit_pool_verse = [
        ("1","1&","2&","4"),
        ("1","1&","3","4"),
        ("1","2&","3","4"),
        ("1","2","2&","4"),
    ]
    hit_pool_chorus = [
        ("1","1&","2&","4"),
        ("1","2","2&","4"),
        ("1","1&","2","4"),
        ("1","2&","3","4"),
    ]

    # ---------- voicing per bar ----------
    # Verse: 2마디 motif -> 2마디 변주 반복(프루시안테 느낌)
    verse_voicing = sample_voicing_per_bar(
        bars_verse,
        style="core",
        end_every=4,
        no_repeat=True,
        mutate=True,
        section="verse",
        motif_repeat_unit=2
    )

    # Chorus: 좀 더 펑키한 왕복/리듬감, 그래도 2마디 단위 반복은 유지
    chorus_voicing = sample_voicing_per_bar(
        bars_chorus,
        style="funk",
        end_every=4,
        no_repeat=True,
        mutate=True,
        section="chorus",
        motif_repeat_unit=2
    )

    # ---------- load models ----------
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

    # 안전용 속성
    ld_trainer.z_dim = z_dim
    ld_trainer.T = T

    # ---------- token generation per section ----------
    verse_tokens = build_tokens_for_steps(
        vae, ld_trainer, verse_steps, device,
        temperature=1.00, rest_logit_penalty=2.3, max_tries=10
    )
    chorus_tokens = build_tokens_for_steps(
        vae, ld_trainer, chorus_steps, device,
        temperature=1.1, rest_logit_penalty=2.8, max_tries=10
    )

    # ---------- render verse ----------
    verse_mid = "_tmp_verse.mid"
    token_seq_to_midi_strum_guitar_voicing(
        verse_tokens, verse_mid,
        tempo=tempo, step_div=step_div, beats_per_bar=beats_per_bar,

        chord_prog=chord_prog_verse,
        chord_change_bars=2,            # Verse는 코드 2마디 유지(기타스럽게)

        hit_pattern_pool=hit_pool_verse,
        voicing_per_bar=verse_voicing,

        use_token_mask=True,            # Verse는 REST로 숨
        ghost_on_rests=True,
        ghost_prob=0.22,
        ghost_prob_on_hits=0.05,

        money_boost=False,

        velocity=90, vel_rand=10,
        timing_jitter_ms=6.0,
        overlap_ms=55.0,
    )

    # ---------- render chorus ----------
    chorus_mid = "_tmp_verse2.mid"
    token_seq_to_midi_strum_guitar_voicing(
        chorus_tokens, chorus_mid,
        tempo=tempo, step_div=step_div, beats_per_bar=beats_per_bar,

        chord_prog=chord_prog_chorus,
        chord_change_bars=1,            # Chorus는 1마디마다 바꿈(진행감↑)

        hit_pattern_pool=hit_pool_chorus,
        voicing_per_bar=chorus_voicing,

        use_token_mask=False,           # Chorus는 빽빽
        ghost_on_rests=True,
        ghost_prob=0.10,
        ghost_prob_on_hits=0.03,

        money_boost=True,
        money_boost_prob=0.28,

        velocity=106, vel_rand=14,
        timing_jitter_ms=5.0,
        overlap_ms=78.0,
    )

    # ---------- concatenate ----------
    concat_two_midis(verse_mid, chorus_mid, out_midi, tempo=tempo)
    print("Saved final MIDI:", out_midi)


if __name__ == "__main__":
    main()
