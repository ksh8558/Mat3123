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
# midi_utils.parse_chord()가 확장코드(7, sus4, add9, dim 등)를 지원하지 않으면
# 렌더 직전에 root + (optional m)만 남겨 단순화한다.
def simplify_chord_for_parser(ch: str) -> str:
    ch = ch.strip()
    if not ch:
        return ch
    # root 추출 (A, Bb, C#, etc.)
    if len(ch) >= 2 and ch[1] in ("#", "b"):
        root = ch[:2]
        rest = ch[2:]
    else:
        root = ch[:1]
        rest = ch[1:]
    # minor 여부만 유지
    is_minor = rest.startswith("m")
    root = _norm_root(root)
    return root + ("m" if is_minor else "")


def sample_progression(style: str, transpose=True, transpose_range=(-3, 3)) -> tuple[str, ...]:
    base = _weighted_choice(PROG_PACKS[style])
    if transpose:
        semis = random.randint(transpose_range[0], transpose_range[1])
        base = tuple(transpose_chord(c, semis) for c in base)
    return base


# --- MIDI utils chord parser is often limited to triads (major/minor only).
#     So we "simplify" extended chords (m7, sus4, add9, dim, etc.) to
#     plain major/minor roots before rendering.
def simplify_chord_for_renderer(ch: str) -> str:
    ch = ch.strip()
    if not ch:
        return ch
    # root extraction
    if len(ch) >= 2 and ch[1] in ("#", "b"):
        root, rest = ch[:2], ch[2:]
    else:
        root, rest = ch[:1], ch[1:]
    # minor if any 'm' quality exists at the beginning (m, m7, m9, m7b5 ...)
    if rest.startswith("m"):
        return root + "m"
    # otherwise just major triad root (also collapses sus/add/dim)
    return root

# =========================================================
# 1.5) Scale-degree progression generator (Frusciante-ish)
#      기존 PROG_PACKS 진행을 "안전망"으로 두고,
#      스케일 기반 진행을 섞어서 harmonic 다양성 ↑
# =========================================================

def _pc(note: str) -> int:
    note = _norm_root(note)
    return NOTE_ORDER_SHARP.index(note)

def _note_from_pc(pc: int) -> str:
    return NOTE_ORDER_SHARP[pc % 12]

# scale degrees (intervals from tonic)
SCALE_INTERVALS = {
    "major":      [0, 2, 4, 5, 7, 9, 11],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "dorian":     [0, 2, 3, 5, 7, 9, 10],
    "aeolian":    [0, 2, 3, 5, 7, 8, 10],
}

# degree -> chord flavor pool (간단 버전)
# (midi_utils 쪽 chord parser가 복잡한 심볼을 못 받으면 여기서 더 단순화하면 됨)
DEGREE_FLAVOR = {
    0: ["", "sus2", "sus4", "add9"],
    1: ["m", "m7"],
    2: ["m", "sus4"],
    3: ["", "add9"],
    4: ["", "sus4"],
    5: ["m", "m7"],
    6: ["dim"],
}

def _parse_chord_root(ch: str) -> str:
    ch = ch.strip()
    if len(ch) >= 2 and ch[1] in ("#", "b"):
        root = ch[:2]
    else:
        root = ch[:1]
    return _norm_root(root)

def _degree_to_chord(tonic: str, scale: str, deg: int) -> str:
    tonic_pc = _pc(tonic)
    intervals = SCALE_INTERVALS[scale]
    root_pc = tonic_pc + intervals[deg % 7]
    root = _note_from_pc(root_pc)

    flavor_pool = DEGREE_FLAVOR.get(deg % 7, [""])
    flavor = random.choice(flavor_pool)

    # "dim"은 표기를 'Bdim'처럼
    if flavor == "dim":
        return root + "dim"
    return root + flavor

def sample_progression_scale(
    tonic: str,
    scale: str,
    length: int = 4,
    step_choices=(-2, -1, 1, 2),
) -> tuple[str, ...]:
    # I 또는 V에서 시작하는 느낌(Frus-ish)
    deg = random.choice([0, 4])
    degs = [deg]
    for _ in range(length - 1):
        deg = (deg + random.choice(step_choices)) % 7
        degs.append(deg)
    return tuple(_degree_to_chord(tonic, scale, d) for d in degs)

def sample_progression_mixed(
    style_pack: str,
    *,
    mix_prob: float = 0.55,     # scale 진행으로 "전체 치환" 확률
    mutate_prob: float = 0.35,  # pack 진행에서 일부만 scale로 바꾸는 확률
    section: str = "verse",
    transpose: bool = True,
) -> tuple[str, ...]:
    """기존 PROG_PACKS 진행을 유지하면서 스케일 기반 진행을 섞는다."""
    # 1) 기존 pack 진행 (안전망)
    base = sample_progression(style_pack, transpose=transpose)

    # 2) 섹션별 추천 스케일
    if section == "verse":
        scale = random.choices(
            ["mixolydian", "dorian", "aeolian"],
            weights=[6, 4, 2],
            k=1
        )[0]
    else:
        scale = random.choices(
            ["major", "mixolydian", "aeolian"],
            weights=[6, 3, 2],
            k=1
        )[0]

    # 3) tonic: base 첫 코드 루트로 통일감
    tonic = _parse_chord_root(base[0])

    scale_prog = sample_progression_scale(tonic, scale, length=len(base))

    r = random.random()
    if r < mix_prob:
        # 전체를 scale 기반으로 교체 (다양성↑)
        return scale_prog

    if r < (mix_prob + mutate_prob):
        # 부분 치환 (1~2개만 갈아끼워 "새로운데 안전" 느낌)
        base = list(base)
        k = random.choice([1, 2])
        idxs = random.sample(range(len(base)), k=k)
        for i in idxs:
            base[i] = scale_prog[i]
        return tuple(base)

    # 그대로 사용
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
# 2.5) Rhythm state machine (per-bar hit patterns)
#    - 리듬을 "패턴 랜덤"이 아니라 "상태 전이"로 움직이게 해서
#      에너지(밀도/비움/밀어치기)가 마디마다 변하도록 만든다.
#
#    * midi_utils.py에 아래처럼 옵션을 추가해야 함:
#      def token_seq_to_midi_strum_guitar_voicing(..., hit_pattern_pool=None, hit_pattern_per_bar=None, ...):
#          ...
#          if hit_pattern_per_bar is not None:
#              hit_pat = hit_pattern_per_bar[bar_idx]
#          else:
#              hit_pat = random.choice(hit_pattern_pool)
# =========================================================

# =========================================================
# 2.5) Rhythm state machine (per-bar hit patterns)
#    - 리듬을 "패턴 랜덤"이 아니라 "상태 전이"로 움직이게 해서
#      에너지(밀도/비움/밀어치기)가 마디마다 변하도록 만든다.
#
#    NOTE:
#    - "한 번만 치는 마디"가 너무 많으면 처음/끝이 휑해져서 별로임.
#      그래서 기본은 2~4 hits로 두고, pickup(1 hit)은 아주 가끔만 허용.
#    - 시작 2마디는 최소 2 hits 보장, 마지막 마디는 drive/push로 밀도 보장.
# =========================================================

RHYTHM_STATES = ("sparse", "push", "drop", "drive")

RHYTHM_TRANSITION = {
    # verse: sparse/push 중심 + drop은 "숨" 정도로만
    "verse": {
        "sparse": (("sparse", 6), ("push", 4), ("drop", 1)),
        "push":   (("sparse", 4), ("push", 3), ("drive", 3), ("drop", 1)),
        "drop":   (("sparse", 6), ("push", 3), ("drive", 1)),
        "drive":  (("push", 6), ("sparse", 3), ("drop", 1)),
    },
    # chorus: drive/push 위주
    "chorus": {
        "sparse": (("push", 6), ("drive", 4), ("sparse", 1)),
        "push":   (("drive", 6), ("push", 3), ("drop", 1)),
        "drop":   (("push", 6), ("drive", 4), ("sparse", 1)),
        "drive":  (("drive", 6), ("push", 4), ("drop", 1)),
    },
}

# --- pickup(1 hit) 패턴: "진짜 가끔"만 ---
HIT_PICKUP = [
    ("4&",),
    ("4",),
]

# 상태별 hit pattern 풀
# - sparse/drop에서도 기본은 2~3 hits로 유지 (숨은 "덜 치기"이지 "한 번만 치기"가 아님)
HIT_CELLS = {
    "verse": {
        "sparse": [
            ("1","3"),
            ("1","4"),
            ("1","2&"),
            ("1&","3"),
            ("1&","4"),
            ("1","3&"),
        ],
        "push": [
            ("1","2&","4"),
            ("1","1&","3"),
            ("1&","2&","4"),
            ("1","3","4"),
            ("1","2&","3"),
            ("1","2&","4&"),
        ],
        "drop": [
            # 숨: 2 hits 위주로
            ("1","4&"),
            ("1","4"),
            ("1&","4"),
            ("1","3"),
        ],
        "drive": [
            ("1","2","3","4"),
            ("1","1&","2&","4"),
            ("1","2&","3","4"),
            ("1","2","2&","4"),
            ("1","2&","3&","4"),
        ],
    },
    "chorus": {
        "sparse": [
            ("1","3"),
            ("1","4"),
            ("1","3","4"),
            ("1","2&","4"),
        ],
        "push": [
            ("1","1&","2&","4"),
            ("1","2","2&","4"),
            ("1","1&","2","4"),
            ("1","2&","3","4"),
            ("1","2&","4&"),
        ],
        "drop": [
            ("1","4&"),
            ("1","2&","4"),
            ("1","3","4"),
        ],
        "drive": [
            ("1","2","3","4"),
            ("1","2&","3","4"),
            ("1","1&","2&","3&","4"),
            ("1","2","2&","3","4"),
        ],
    },
}

def _weighted_next_state(section: str, cur: str) -> str:
    trans = RHYTHM_TRANSITION[section][cur]
    states = [s for s,_ in trans]
    ws = [w for _,w in trans]
    return random.choices(states, weights=ws, k=1)[0]

def sample_hit_patterns_per_bar(
    n_bars: int,
    *,
    section: str,
    motif_repeat_unit: int = 2,    # 2마디 단위로 리듬 motif를 만들고 다음 2마디에서 변주
    variation_prob: float = 0.40,  # repeat 블록에서 셀을 바꿀 확률
    pickup_prob: float = 0.05,     # 1-hit pickup 확률(연속 금지)
    start_min_hits: int = 2,       # 시작 2마디 최소 hit 수
):
    """
    반환: 길이 n_bars의 hit pattern list.
    - 블록 0(2마디): 상태 머신으로 리듬 motif 생성
    - 블록 1(2마디): 직전 motif 기반으로 약간 변주
    - ...
    추가 규칙:
    - pickup(1 hit)은 매우 낮은 확률로만, 연속 금지, 시작 2마디/마지막 마디 금지
    - 마지막 마디는 drive/push로 강제 (엔딩 밀도)
    """
    sec = "chorus" if section == "chorus" else "verse"
    cells = HIT_CELLS[sec]

    out = []
    last_block = None
    # 시작 state: verse는 sparse, chorus는 push
    cur_state = "push" if sec == "chorus" else "sparse"
    last_was_pickup = False

    bar = 0
    while bar < n_bars:
        block_len = min(motif_repeat_unit, n_bars - bar)
        block_id = bar // motif_repeat_unit
        is_repeat_block = (last_block is not None and (block_id % 2 == 1))

        block = []
        for i in range(block_len):
            bar_idx = bar + i
            is_start_zone = (bar_idx < 2)
            is_last_bar = (bar_idx == n_bars - 1)

            def choose_pat_for_state(st: str):
                # 기본 셀에서 선택
                cand = random.choice(cells[st])

                # 시작 2마디: 최소 hit 수 보장
                if is_start_zone:
                    tries = 0
                    while len(cand) < start_min_hits and tries < 12:
                        cand = random.choice(cells[st])
                        tries += 1

                # 마지막 마디: 밀도 보장 (drive/push 우선)
                if is_last_bar and st in ("sparse", "drop"):
                    st2 = "drive" if random.random() < 0.65 else "push"
                    cand2 = random.choice(cells[st2])
                    return st2, cand2

                return st, cand

            # 1) repeat block이면 이전 motif 기반
            if is_repeat_block and i < len(last_block):
                prev_state, prev_pat = last_block[i]
                if random.random() < variation_prob:
                    st, pat = choose_pat_for_state(prev_state)
                else:
                    st, pat = prev_state, prev_pat
            else:
                cur_state = _weighted_next_state(sec, cur_state)
                st, pat = choose_pat_for_state(cur_state)

            # 2) pickup(1 hit) 아주 가끔: 연속 금지 + 시작/마지막 금지
            allow_pickup = (not is_start_zone) and (not is_last_bar) and (not last_was_pickup)
            if allow_pickup and random.random() < pickup_prob:
                pat = random.choice(HIT_PICKUP)
                last_was_pickup = True
            else:
                last_was_pickup = (len(pat) == 1)

            block.append((st, pat))

        out.extend([pat for _, pat in block])

        if not is_repeat_block:
            last_block = block

        bar += block_len

    return out

def render_with_optional_hit_per_bar(*args, hit_pattern_pool=None, hit_pattern_per_bar=None, **kwargs):
    """
    midi_utils가 hit_pattern_per_bar 인자를 지원하면 per-bar 리듬을 쓰고,
    지원하지 않으면 기존 hit_pattern_pool 방식으로 fallback.
    """
    try:
        return token_seq_to_midi_strum_guitar_voicing(
            *args,
            hit_pattern_pool=hit_pattern_pool,
            hit_pattern_per_bar=hit_pattern_per_bar,
            **kwargs
        )
    except TypeError:
        # older midi_utils.py (no hit_pattern_per_bar)
        return token_seq_to_midi_strum_guitar_voicing(
            *args,
            hit_pattern_pool=hit_pattern_pool,
            **kwargs
        )

# ---------------------------------------------------------
# voicing length safety:
# midi_utils.py가 voicing 길이 == hit 개수를 강제하는 경우가 있어,
# generate 단계에서 미리 맞춰서(ValueError 방지) 전달한다.
# ---------------------------------------------------------
def _fit_voicing_to_hits(vp: str, n_hits: int) -> str:
    vp = str(vp)
    if n_hits <= 0:
        return ""
    if len(vp) == n_hits:
        return vp
    if len(vp) > n_hits:
        return vp[:n_hits]
    return vp + (vp[-1] * (n_hits - len(vp)))

def _n_hits_of_pattern(pat) -> int:
    # pat: tuple[str,...] like ("1","2&","4")
    try:
        return len(pat)
    except Exception:
        return 0

def adapt_voicing_per_bar_to_hits(voicing_per_bar, hit_pattern_per_bar):
    if voicing_per_bar is None or hit_pattern_per_bar is None:
        return voicing_per_bar
    out = []
    L = min(len(voicing_per_bar), len(hit_pattern_per_bar))
    for i in range(L):
        out.append(_fit_voicing_to_hits(voicing_per_bar[i], _n_hits_of_pattern(hit_pattern_per_bar[i])))
    # if voicing list longer, keep tail; if shorter, repeat last
    if len(voicing_per_bar) > L:
        out.extend(list(voicing_per_bar[L:]))
    elif len(voicing_per_bar) < len(hit_pattern_per_bar) and len(out) > 0:
        last = out[-1]
        for j in range(L, len(hit_pattern_per_bar)):
            out.append(_fit_voicing_to_hits(last, _n_hits_of_pattern(hit_pattern_per_bar[j])))
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
    chord_prog_verse = sample_progression_mixed("mixolydian_funk", section="verse", transpose=True)
    chord_prog_chorus = sample_progression_mixed("major_pop", section="chorus", transpose=True)
    # midi_utils의 parse_chord 호환을 위해 확장코드를 단순화
    chord_prog_verse = tuple(simplify_chord_for_parser(c) for c in chord_prog_verse)
    chord_prog_chorus = tuple(simplify_chord_for_parser(c) for c in chord_prog_chorus)

    # ---------- hit patterns (state machine per bar) ----------
    verse_hit_per_bar = sample_hit_patterns_per_bar(bars_verse, section="verse", motif_repeat_unit=2)
    chorus_hit_per_bar = sample_hit_patterns_per_bar(bars_chorus, section="chorus", motif_repeat_unit=2)

    # (backward-compat) midi_utils가 per_bar를 아직 지원 안 하면, 아래 pool을 사용하도록 남겨둠
    hit_pool_verse = list({tuple(p) for p in verse_hit_per_bar}) or [("1",)]
    hit_pool_chorus = list({tuple(p) for p in chorus_hit_per_bar}) or [("1","2","3","4")]

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

    
# ---------- render verse (intro to avoid "땅-쉼" problem) ----------
# Verse 초반에 use_token_mask=True면 첫 마디가 REST로 비어 "땅 하고 쉼"처럼 들릴 수 있어서,
# 앞 1~2마디는 마스크 없이(Intro) 렌더 후 본 Verse와 붙인다.

    intro_bars = min(2, bars_verse)  # 1~2 추천
    intro_steps = intro_bars * steps_per_bar
    body_steps = (bars_verse - intro_bars) * steps_per_bar

    intro_tokens = verse_tokens[:intro_steps] if intro_bars > 0 else None
    body_tokens = verse_tokens[intro_steps:intro_steps + body_steps] if body_steps > 0 else None

    # intro는 무조건 4-hit 위주로(그루브/진입감)
    intro_drive_pool = [
        ("1","1&","2&","4"),
        ("1","2","2&","4"),
        ("1","2&","3","4"),
    ]
    intro_hit_per_bar = [random.choice(intro_drive_pool) for _ in range(intro_bars)]

    # body는 state machine 결과 사용
    body_hit_per_bar = verse_hit_per_bar[intro_bars:bars_verse]

    # voicing 길이 mismatch 방지(구버전 midi_utils에서도 안전)
    intro_voicing = adapt_voicing_per_bar_to_hits(verse_voicing[:intro_bars], intro_hit_per_bar) if intro_bars > 0 else None
    body_voicing = adapt_voicing_per_bar_to_hits(verse_voicing[intro_bars:bars_verse], body_hit_per_bar) if body_steps > 0 else None

    intro_mid = "_tmp_intro.mid"
    body_mid = "_tmp_verse_body.mid"
    verse_mid = "_tmp_verse.mid"

    if intro_bars > 0:
        render_with_optional_hit_per_bar(
            intro_tokens, intro_mid,
            tempo=tempo, step_div=step_div, beats_per_bar=beats_per_bar,

            chord_prog=chord_prog_verse,
            chord_change_bars=2,

            hit_pattern_pool=hit_pool_verse,
            hit_pattern_per_bar=intro_hit_per_bar,
            voicing_per_bar=intro_voicing,

            use_token_mask=False,           # ★ intro는 마스크 OFF
            ghost_on_rests=True,
            ghost_prob=0.10,
            ghost_prob_on_hits=0.04,

            money_boost=False,

            velocity=105, vel_rand=10,
            timing_jitter_ms=6.0,
            overlap_ms=78.0,
        )

    if body_steps > 0:
        render_with_optional_hit_per_bar(
            body_tokens, body_mid,
            tempo=tempo, step_div=step_div, beats_per_bar=beats_per_bar,

            chord_prog=chord_prog_verse,
            chord_change_bars=2,            # Verse는 코드 2마디 유지(기타스럽게)

            hit_pattern_pool=hit_pool_verse,
            hit_pattern_per_bar=body_hit_per_bar,
            voicing_per_bar=body_voicing,

            use_token_mask=True,            # 본 Verse는 REST로 숨
            ghost_on_rests=True,
            ghost_prob=0.22,
            ghost_prob_on_hits=0.05,

            money_boost=False,

            velocity=90, vel_rand=10,
            timing_jitter_ms=6.0,
            overlap_ms=55.0,
        )

    import os
    if intro_bars > 0 and body_steps > 0:
        concat_two_midis(intro_mid, body_mid, verse_mid, tempo=tempo)
    elif intro_bars > 0:
        os.replace(intro_mid, verse_mid)
    else:
        os.replace(body_mid, verse_mid)



    # ---------- render chorus ----------
    # voicing 길이 mismatch 방지(구버전 midi_utils에서도 안전)
    chorus_voicing_safe = adapt_voicing_per_bar_to_hits(chorus_voicing, chorus_hit_per_bar)

    chorus_mid = "_tmp_verse2.mid"
    render_with_optional_hit_per_bar(
        chorus_tokens, chorus_mid,
        tempo=tempo, step_div=step_div, beats_per_bar=beats_per_bar,

        chord_prog=chord_prog_chorus,
        chord_change_bars=1,            # Chorus는 1마디마다 바꿈(진행감↑)

        hit_pattern_pool=hit_pool_chorus,
        hit_pattern_per_bar=chorus_hit_per_bar,
        voicing_per_bar=chorus_voicing_safe,

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
