# frusciante_voicing.py
import random

# 1=low power, 2=high power, 3=money(sus4), 4=full(rare)
# 패턴은 hit_pattern 길이(보통 4)에 맞춰 4자리 문자열

BASE_POOL = {
    "core": {
        "base": [
            "1212", "2121", "1112", "1121", "1211", "2112",
            "1111",  # 완전 기본도 섞어야 '사람' 같아짐
        ],
        "weights": {
            "1212": 8, "2121": 5,
            "1112": 5, "1121": 4, "1211": 4, "2112": 3,
            "1111": 2,
        }
    },
    "funk": {
        "base": [
            "1212", "2121", "1221", "2112", "1211", "1121"
        ],
        "weights": {
            "1212": 7, "2121": 6, "1221": 4, "2112": 4, "1211": 3, "1121": 3
        }
    },
    "melodic": {
        "base": [
            "1111", "1121", "1211", "1112"
        ],
        "weights": {
            "1111": 7, "1121": 5, "1211": 4, "1112": 4
        }
    }
}

# 엔딩용(마지막 hit에 3을 두는 패턴이 많음)
ENDING_POOL = ["1113", "1213", "1123", "1114"]
ENDING_W = {"1113": 7, "1213": 4, "1123": 3, "1114": 1}


def _weighted_choice(patterns, weights_dict):
    ws = [weights_dict.get(p, 1) for p in patterns]
    return random.choices(patterns, weights=ws, k=1)[0]


def _mutate_pattern(pat: str,
                    p_swap12: float = 0.12,
                    p_drop2_to1: float = 0.10,
                    p_add_money_last: float = 0.18,
                    allow_4: bool = False):
    """
    미세 변형으로 '매번 똑같이' 들리는 걸 깨기.
    - swap12: 1<->2 스왑(포지션 바뀐 느낌)
    - drop2_to1: 어떤 hit의 2를 1로 내려서 단순하게(사람 손)
    - add_money_last: 마지막 hit을 3으로 바꿀 확률(엔딩 느낌)
    """
    s = list(pat)

    # 1<->2 스왑
    if random.random() < p_swap12:
        s = ['2' if c == '1' else ('1' if c == '2' else c) for c in s]

    # 2를 1로 다운그레이드(너무 하이가 많으면 피로해짐)
    if random.random() < p_drop2_to1:
        idxs = [i for i, c in enumerate(s) if c == '2']
        if idxs:
            i = random.choice(idxs)
            s[i] = '1'

    # 마지막 hit에 money 살짝
    if random.random() < p_add_money_last:
        s[-1] = '3'

    # 4는 정말 드물게
    if allow_4 and random.random() < 0.03:
        s[-1] = '4'

    return "".join(s)


def sample_voicing_per_bar(
    n_bars: int,
    style: str = "core",
    end_every: int | None = 4,     # 4마디마다 엔딩 패턴 주기
    no_repeat: bool = True,
    mutate: bool = True,
):
    """
    '사이클 하나'가 아니라 '마디마다' 뽑아줌 -> 훨씬 덜 똑같이 들림.
    반환: tuple[str] 길이 n_bars
    """
    if style not in BASE_POOL:
        raise ValueError(f"Unknown style: {style}")

    base = BASE_POOL[style]["base"]
    w = BASE_POOL[style]["weights"]

    out = []
    last = None

    for bar in range(n_bars):
        is_ending_bar = (end_every is not None and (bar % end_every) == end_every - 1)

        if is_ending_bar:
            pat = _weighted_choice(ENDING_POOL, ENDING_W)
        else:
            pat = _weighted_choice(base, w)

        if no_repeat and last is not None:
            # 같은 패턴이 연속이면 다시 뽑기
            tries = 0
            while pat == last and tries < 10:
                pat = _weighted_choice(base, w) if not is_ending_bar else _weighted_choice(ENDING_POOL, ENDING_W)
                tries += 1

        if mutate and not is_ending_bar:
            pat = _mutate_pattern(pat)

        out.append(pat)
        last = pat

    return tuple(out)
