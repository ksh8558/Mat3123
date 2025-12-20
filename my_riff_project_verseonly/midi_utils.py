# MIDI ↔ 토큰 변환 함수들.
import numpy as np
import pretty_midi

from config import PITCH_MIN, PITCH_MAX, REST_TOKEN, BARS, BEATS_PER_BAR, STEPS_PER_BEAT


def note_to_token(pitch: int) -> int:
    if pitch < PITCH_MIN or pitch > PITCH_MAX:
        return REST_TOKEN
    return (pitch - PITCH_MIN) + 1


def token_to_pitch(token: int):
    if token == REST_TOKEN:
        return None
    return (token - 1) + PITCH_MIN


def midi_to_token_seq(
    midi_path: str,
    bars: int = BARS,
    beats_per_bar: int = BEATS_PER_BAR,
    steps_per_beat: int = STEPS_PER_BEAT,
    track_idx: int | None = None,
) -> np.ndarray:
    pm = pretty_midi.PrettyMIDI(midi_path)

    if track_idx is None:
        insts = [inst for inst in pm.instruments if not inst.is_drum]
        if len(insts) == 0:
            raise ValueError("No non-drum tracks found.")
        inst = insts[0]
    else:
        inst = pm.instruments[track_idx]

    L = bars * beats_per_bar * steps_per_beat
    tempo = pm.estimate_tempo()
    sec_per_beat = 60.0 / tempo
    total_dur = bars * beats_per_bar * sec_per_beat

    times = np.linspace(0, total_dur, num=L, endpoint=False)
    tokens = np.zeros(L, dtype=np.int64)
    notes = sorted(inst.notes, key=lambda n: n.start)

    for i, t in enumerate(times):
        active = [n.pitch for n in notes if (n.start <= t < n.end)]
        tokens[i] = REST_TOKEN if len(active) == 0 else note_to_token(max(active))

    return tokens


def token_seq_to_midi(
    tokens,
    out_path,
    tempo=92.0,
    step_div=4,                # 16분 그리드 추천(4)
    velocity=86,
    vel_rand=10,               # 악센트 아님: 사람 손맛(미세 변화)
    timing_jitter_ms=5.0,      # 사람 손맛
    overlap_ms=35.0,
    beats_per_bar=4,
    transpose_semitones=-7,

    # bar 단위 sustain(끊김 방지)
    rest_grace_bars=1,

    # ===== Rhythm: 실행마다 1개 랜덤 선택(곡 내 고정) =====
    strum_pattern_steps=None,
    rhythm_pool=None,
    keep_rhythm_fixed=True,

    # ===== Intro 1-bar rhythm variation (bar0만) =====
    intro_bar_variation=True,
    intro_force_diff_rhythm=True,

    # ===== Voicing cycle =====
    voicing_pool=None,
    voicing_keep_fixed_in_song=True,
    voicing_change_prob_per_bar=0.25,
    voicing_cycle_seed=None,

    # ===== NEW: hit마다 chord tones variant 섞기 =====
    # (악센트 없이도 "한 마디에 같은 음만" 반복되는 문제 해결)
    voicing_variant_mix=True,
    triad_bias=0.60,            # triad 비중 (0~1)
    variant_pool=("dyad", "triad", "add9", "sus2", "sus4"),

    # ===== Hendrix spice (선택) =====
    hendrix_spice=True,
    hendrix_prob=0.10,
):
    """
    ✅ 악센트/ghost/skip 없음 (강세 설계 제거)
    ✅ 리듬은 실행마다 랜덤 1개 선택, 곡 내 고정
    ✅ 첫 마디는 리듬만 변주
    ✅ voicing_cycle(1114/2231/2233 등)로 한 마디 안에서도 '소리' 반복/변주
    ✅ (NEW) 같은 root라도 hit마다 chord_tones variant를 섞어서
        "한 마디에 똑같은 음만 계속" 문제를 크게 줄임
    ✅ '4'는 단순 반복이 아니라 상단 2음 더블스탑으로 처리
    """
    tokens = np.array(tokens, dtype=np.int64)
    pm = pretty_midi.PrettyMIDI(initial_tempo=float(tempo))

    inst = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program("Electric Guitar (clean)"))
    inst_mute = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program("Electric Guitar (muted)"))

    rng = np.random.default_rng()
    rng_vc = np.random.default_rng(voicing_cycle_seed if voicing_cycle_seed is not None else rng.integers(0, 2**32 - 1))

    step_dur = (60.0 / float(tempo)) / float(step_div)
    steps_per_bar = int(step_div * beats_per_bar)
    L = len(tokens)

    # -----------------------------
    # Rhythm pool (16th grid 기준)
    # -----------------------------
    if rhythm_pool is None:
        rhythm_pool = [
            (0, 2, 6, 12),          # 1, 1&, 2&, 4
            (0, 4, 6, 12),          # 1, 2, 2&, 4
            (0, 2, 8, 12),          # 1, 1&, 3, 4
            (0, 6, 8, 12),          # 1, 2&, 3, 4
            (0, 2, 6, 10, 12),      # 1, 1&, 2&, 3&, 4
            (0, 4, 8, 12),          # 1, 2, 3, 4
            (0, 2, 4, 6, 12),       # 1, 1&, 2, 2&, 4
            (0, 3, 6, 11, 12),      # 살짝 펑키
            (0, 2, 7, 12),          # 1, 1&, 2e, 4
        ]

    if strum_pattern_steps is None:
        main_rhythm = rhythm_pool[rng.integers(0, len(rhythm_pool))]
    else:
        main_rhythm = tuple(strum_pattern_steps)

    # intro rhythm (bar0만)
    intro_pool = [
        (0, 2, 6, 12),
        (0, 2, 4, 6, 12),
        (0, 4, 6, 12),
        (0, 2, 8, 12),
        (0, 3, 6, 11, 12),
        (0, 2, 6, 10, 12),
    ]
    if intro_bar_variation:
        if intro_force_diff_rhythm:
            cand = [r for r in intro_pool if tuple(r) != tuple(main_rhythm)]
            intro_rhythm = cand[rng.integers(0, len(cand))] if len(cand) > 0 else main_rhythm
        else:
            intro_rhythm = intro_pool[rng.integers(0, len(intro_pool))]
    else:
        intro_rhythm = main_rhythm

    # -----------------------------
    # Voicing cycle pool
    # -----------------------------
    if voicing_pool is None:
        voicing_pool = [
            ("1114", "2231", "2233"),
            ("1212", "2323", "2312"),
            ("1123", "2213", "2231"),
            ("1232", "2321", "2231"),
            ("1312", "2323", "1212"),
            ("1112", "2231", "2323"),
        ]

    voicing_set = voicing_pool[rng_vc.integers(0, len(voicing_pool))]
    print("[Rhythm] main =", main_rhythm, "| intro(bar0) =", intro_rhythm, "| fixed_in_song =", keep_rhythm_fixed)
    print("[Voicing] set =", voicing_set, "| fixed_in_song =", voicing_keep_fixed_in_song)

    # ---------- helpers ----------
    def clamp_to_guitar_range(p: int, lo=48, hi=76) -> int:
        while p > hi:
            p -= 12
        while p < lo:
            p += 12
        return int(np.clip(p, 0, 127))

    def add_note(inst_, pitch: int, start: float, end: float, base_vel: int):
        jitter = rng.uniform(-timing_jitter_ms, timing_jitter_ms) * 1e-3
        s = max(0.0, start + jitter)
        e = max(s + 0.02, end + jitter + overlap_ms * 1e-3)

        v = int(base_vel + rng.integers(-vel_rand, vel_rand + 1))
        v = int(np.clip(v, 30, 112))

        inst_.notes.append(pretty_midi.Note(
            velocity=v,
            pitch=int(np.clip(pitch, 0, 127)),
            start=float(s),
            end=float(e),
        ))

    def representative_root_for_bar(bar_tokens: np.ndarray, fallback: int | None) -> int | None:
        pitches = []
        for tok in bar_tokens:
            p = token_to_pitch(int(tok))
            if p is not None and p > 0:
                pitches.append(int(p) + int(transpose_semitones))
        if len(pitches) == 0:
            return fallback
        vals, cnts = np.unique(pitches, return_counts=True)
        return int(vals[np.argmax(cnts)])

    def pick_chord_tones_variant(root_pitch: int, variant: str) -> list[int]:
        """
        같은 root라도 variant에 따라 chord_tones 구성을 바꿔서
        한 마디 안에서도 음이 계속 같지 않게 함.
        """
        r = root_pitch
        is_minor = (rng.random() < 0.35)

        if variant == "dyad":
            intervals = [0, 7]
        elif variant == "triad":
            intervals = [0, (3 if is_minor else 4), 7]
        elif variant == "add9":
            intervals = [0, 7, 14]
        elif variant == "sus2":
            intervals = [0, 2, 7]
        elif variant == "sus4":
            intervals = [0, 5, 7]
        else:
            intervals = [0, 7]

        tones = [clamp_to_guitar_range(r + iv) for iv in intervals]
        tones = sorted(set(tones))
        if len(tones) > 3:
            tones = tones[:3]
        return tones

    def apply_voicing_pattern(chord_tones: list[int], pattern: str) -> list[int]:
        """
        pattern 예: "1114"
        - 1,2,3: 해당 인덱스(클램프)
        - 4: 상단 2음 더블스탑(tones[-2], tones[-1])로 처리
        """
        n = len(chord_tones)
        if n <= 0:
            return []

        out = []
        for ch in pattern:
            if ch == "4":
                if n >= 2:
                    out.extend([chord_tones[-2], chord_tones[-1]])
                else:
                    out.append(chord_tones[-1])
                continue

            if ch < "1" or ch > "9":
                continue
            k = int(ch)
            idx = min(max(k - 1, 0), n - 1)
            out.append(chord_tones[idx])

        return out

    def add_hendrix_spice(base_root: int, bar_start_step: int):
        if not hendrix_spice:
            return
        if rng.random() >= hendrix_prob:
            return

        cand = [int(steps_per_bar * 0.45), int(steps_per_bar * 0.85)]
        hit_step = bar_start_step + cand[rng.integers(0, len(cand))]

        start = hit_step * step_dur
        dur = rng.uniform(0.045, 0.085)
        end = start + dur

        r = clamp_to_guitar_range(base_root, lo=48, hi=76)

        if rng.random() < 0.55:
            intervals = [4, 9]   # 3rd + 6th
        else:
            intervals = [3, 10]  # b3 + b7

        p1 = clamp_to_guitar_range(r + intervals[0], lo=52, hi=84)
        p2 = clamp_to_guitar_range(r + intervals[1], lo=52, hi=84)

        base_v = int(np.clip(velocity * 0.62, 30, 95))
        add_note(inst_mute, p1, start, end, base_v)
        add_note(inst_mute, p2, start + 0.012, end, int(base_v * 0.92))

    # -----------------------------
    # bar loop (마디 단위 코드 체인지)
    # -----------------------------
    n_bars = (L + steps_per_bar - 1) // steps_per_bar
    prev_root = None
    empty_bar_run = 0

    variants = list(variant_pool)

    for bar in range(n_bars):
        bar_start = bar * steps_per_bar
        bar_end = min(L, bar_start + steps_per_bar)
        bar_tokens = tokens[bar_start:bar_end]

        root = representative_root_for_bar(bar_tokens, prev_root)

        if np.all(bar_tokens == REST_TOKEN):
            empty_bar_run += 1
            if prev_root is None or empty_bar_run > rest_grace_bars:
                continue
            root = prev_root
        else:
            empty_bar_run = 0

        if root is None:
            continue

        prev_root = root
        root = clamp_to_guitar_range(root)

        # (bar마다 voicing_set을 바꿀지 옵션)
        if not voicing_keep_fixed_in_song and (rng_vc.random() < voicing_change_prob_per_bar):
            voicing_set = voicing_pool[rng_vc.integers(0, len(voicing_pool))]

        cur_rhythm = intro_rhythm if (intro_bar_variation and bar == 0) else main_rhythm

        for i_hit, off in enumerate(cur_rhythm):
            hit_step = bar_start + off
            if hit_step >= bar_end:
                continue

            t0 = hit_step * step_dur

            # ✅ hit마다 chord_tones variant 섞기
            if voicing_variant_mix:
                if rng.random() < float(triad_bias):
                    variant = "triad"
                else:
                    variant = variants[rng.integers(0, len(variants))]
            else:
                variant = "triad"

            chord_tones = pick_chord_tones_variant(root, variant)

            pattern = voicing_set[i_hit % len(voicing_set)]
            pick_seq = apply_voicing_pattern(chord_tones, pattern)

            # sustain는 악센트가 아니라 '끊김 방지/기타 공명' 수준에서만 살짝 랜덤
            sustain_steps = rng.uniform(2.6, 3.4)  # 16분 기준 대략 2~3.5 step
            end = t0 + step_dur * sustain_steps

            # pick_seq는 한 줄씩(아르페지오). 작은 스프레드만.
            spread = rng.uniform(0.010, 0.020)
            delays = np.linspace(0.0, spread, num=max(1, len(pick_seq)), endpoint=True)

            for pch, dly in zip(pick_seq, delays):
                add_note(inst, int(pch), t0 + float(dly), end, velocity)

        add_hendrix_spice(root, bar_start)

    pm.instruments.append(inst)
    pm.instruments.append(inst_mute)
    pm.write(out_path)
    print("Saved MIDI:", out_path)
