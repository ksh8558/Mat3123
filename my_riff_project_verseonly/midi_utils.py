import numpy as np
import pretty_midi

# REST_TOKEN은 네 config에 맞게 import해도 되고, 여기서 0 가정해도 됨
# from config import REST_TOKEN
REST_TOKEN = 0

def token_seq_to_midi_strum_guitar_voicing(
    tokens,
    out_path,
    tempo=100.0,
    step_div=2,                       # "&" 쓰려면 2 추천
    beats_per_bar=4,

    chord_prog=("Em", "C", "G", "D"),
    chord_change_bars=2,

    # ===== 랜덤화 옵션 =====
    hit_pattern=("1","1&","2&","4"),  # 고정 사용
    hit_pattern_pool=None,            # 마디마다 랜덤 사용(아래 기본 풀 제공)
    voicing_cycle=("1212","1212","1112","1113"),  # 고정 사용
    voicing_per_bar=None,             # 마디마다 지정(추천: sample_voicing_per_bar 결과)

    # ===== 스트로크/고스트 =====
    use_token_mask=True,              # tokens[step]==REST면 그 hit은 안침(코러스면 False 권장)
    ghost_on_rests=True,              # REST 위치에서도 고스트 스트럼 가능
    ghost_prob=0.18,                  # 고스트 발생 확률
    ghost_prob_on_hits=0.06,          # hit에서도 가끔 고스트를 섞음(더 사람같음)
    ghost_dur_ms=(45, 85),            # 고스트 길이(짧게)
    ghost_vel=(18, 40),               # 고스트 세기

    # ===== 사운드 =====
    velocity=92,
    vel_rand=12,
    overlap_ms=60.0,
    timing_jitter_ms=6.0,

    # ===== 코러스 느낌 부스트 =====
    money_boost=False,                # 코러스에서 sus4 더 자주/더 크게
    money_boost_prob=0.22,            # (1)->(3)로 승격 확률
):
    pm = pretty_midi.PrettyMIDI(initial_tempo=float(tempo))

    inst_clean = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program("Electric Guitar (clean)")
    )
    inst_mute = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program("Electric Guitar (muted)")
    )

    tokens = np.array(tokens, dtype=np.int64)
    rng = np.random.default_rng()

    step_dur = (60.0 / float(tempo)) / float(step_div)
    steps_per_bar = int(step_div * beats_per_bar)

    # ---------------- chord parse ----------------
    NOTE2PC = {"C":0,"C#":1,"Db":1,"D":2,"D#":3,"Eb":3,"E":4,"F":5,"F#":6,"Gb":6,"G":7,"G#":8,"Ab":8,"A":9,"A#":10,"Bb":10,"B":11}
    def parse_chord(ch: str):
        ch = ch.strip()
        qual = "maj"
        if ch.endswith("m") and not ch.endswith("maj"):
            qual = "min"
            root_name = ch[:-1]
        else:
            root_name = ch
        if root_name not in NOTE2PC:
            raise ValueError(f"Unsupported chord: {ch}")
        return NOTE2PC[root_name], qual

    prog = [parse_chord(c) for c in chord_prog]

    # tag -> step (supports "&"; also supports e/a if step_div==4)
    def beat_to_step(tag: str):
        tag = tag.strip()
        # ex: "2&", "3", "1e", "4a"
        if len(tag) == 1 and tag.isdigit():
            b = int(tag)
            return (b - 1) * step_div
        b = int(tag[0])
        suf = tag[1:]
        base = (b - 1) * step_div

        if suf == "&":
            return base + int(step_div / 2)
        if step_div >= 4:
            # 16th grid: e=+1, &=+2, a=+3
            if suf == "e":
                return base + 1
            if suf == "&":
                return base + 2
            if suf == "a":
                return base + 3
        # fallback
        return base

    def is_upstroke(tag: str):
        tag = tag.strip()
        return (tag.endswith("&") or tag.endswith("e") or tag.endswith("a"))

    def fit_voicing_to_hits(vp: str, n_hits: int) -> str:
        """
        voicing 문자열 길이를 hit 개수에 맞춤.
        - 길면: 마지막 문자를 살리고 앞부분을 자름 (accent 유지)
        - 짧으면: 마지막 문자를 반복해서 늘림
        """
        if n_hits <= 0:
            return ""
        vp = str(vp)
        if len(vp) == n_hits:
            return vp
        if len(vp) > n_hits:
            if n_hits == 1:
                return vp[-1]
            return vp[:n_hits-1] + vp[-1]
        # len(vp) < n_hits
        return vp + (vp[-1] * (n_hits - len(vp)))

    # 기본 hit pattern pool (전부 길이 4로 맞춤)
    if hit_pattern_pool is None:
        hit_pattern_pool = [
            ("1","1&","2&","4"),   # 기본 펑키
            ("1","2&","3","4"),    # 띄엄띄엄
            ("1","1&","3","4"),    # 2를 비워서 공간
            ("1","2","2&","4"),    # 2에서 몰아치기
            ("1","1&","2","4"),    # 안정 + 업스트로크
        ]

    # ---------------- voicing ----------------
    # 1=low power, 2=high power, 3=money(sus4), 4=full(rare)
    # -12는 절대 안 씀.
    def choose_root_midi(root_pc, base):
        best = None
        for m in range(40, 80):
            if (m % 12) == root_pc:
                if best is None or abs(m - base) < abs(best - base):
                    best = m
        return best if best is not None else base

    def voicing_pitches(root_pc, quality, vid: int):
        if vid == 1:
            r = choose_root_midi(root_pc, base=52)     # low
            return [r, r + 7]
        if vid == 2:
            r = choose_root_midi(root_pc, base=64)     # high
            return [r, r + 7]
        if vid == 3:
            r = choose_root_midi(root_pc, base=57)     # mid
            return [r, r + 7, r + 5]
        if vid == 4:
            r = choose_root_midi(root_pc, base=52)
            return [r, r + 7, r + 5]
        r = choose_root_midi(root_pc, base=52)
        return [r, r + 7]

    # downstroke: low->high, upstroke: high->low
    def add_strum(inst, pitches, start, end, v, direction="down", strength=1.0):
        ps = list(pitches)
        if direction == "up":
            ps = list(reversed(ps))

        t = start
        for i, p in enumerate(ps):
            if 0 <= p <= 127:
                dt = rng.uniform(0.006, 0.016)  # strum delay
                vv = int(np.clip(v * strength * (0.95 - 0.06*i), 1, 127))
                inst.notes.append(pretty_midi.Note(
                    velocity=vv, pitch=int(p), start=float(t), end=float(end)
                ))
                t += dt

    # ghost strum = muted, 짧게 긁기
    def add_ghost(start, direction="down"):
        dur = rng.uniform(ghost_dur_ms[0], ghost_dur_ms[1]) * 1e-3
        end = start + dur
        v = rng.integers(ghost_vel[0], ghost_vel[1] + 1)

        # muted는 너무 화음처럼 들리면 안 돼서 1~2음만 (root/fifth 느낌)
        # 여기선 "E3(52) 근처" 같은 고정 보다는 현재 코드 기반으로 넣어줌 (아래에서 호출 시 인자로 pitches 줄 수도 있지만 간단히)
        # 일단 무난하게 '짧게 두 음'만:
        # (실제로는 inst_mute program이 muted 톤이라 충분히 기타 느낌 남)
        return end, int(v)

    # ---------------- main loop ----------------
    n_steps = len(tokens)
    n_bars = int(np.ceil(n_steps / steps_per_bar))

    for bar in range(n_bars):
        bar_start_step = bar * steps_per_bar

        # chord select
        prog_idx = (bar // int(chord_change_bars)) % len(prog)
        root_pc, quality = prog[prog_idx]

        # pattern select (bar마다 랜덤)
        if hit_pattern_pool is not None:
            pat = hit_pattern_pool[rng.integers(0, len(hit_pattern_pool))]
        else:
            pat = hit_pattern

        hit_steps = [beat_to_step(x) for x in pat]
        up_flags = [is_upstroke(x) for x in pat]

        # voicing select (bar마다)
        if voicing_per_bar is not None:
            vp = voicing_per_bar[bar % len(voicing_per_bar)]
        else:
            vp = voicing_cycle[bar % len(voicing_cycle)]

        # voicing 길이가 hit 수와 다르면 자동으로 맞춰서 진행(리듬 다양성 허용)
        if len(vp) != len(hit_steps):
            vp = fit_voicing_to_hits(vp, len(hit_steps))

        for hs, is_up, ch in zip(hit_steps, up_flags, vp):
            step = bar_start_step + hs
            if step >= n_steps:
                continue

            hit_is_rest = (tokens[step] == REST_TOKEN)

            # 고스트를 hit에서도 섞고, rest에서도 섞음(옵션)
            do_ghost = False
            if hit_is_rest and ghost_on_rests and rng.random() < ghost_prob:
                do_ghost = True
            if (not hit_is_rest) and rng.random() < ghost_prob_on_hits:
                do_ghost = True

            # 리듬 마스크(REST면 안 치기) — 단, 고스트는 가능
            if use_token_mask and hit_is_rest and (not do_ghost):
                continue

            start = step * step_dur
            # timing jitter
            jitter = rng.uniform(-timing_jitter_ms, timing_jitter_ms) * 1e-3
            start = max(0.0, start + jitter)

            if do_ghost:
                end, gv = add_ghost(start, direction=("up" if is_up else "down"))
                # 현재 코드 기반의 "짧은 뮤트" 1~2음
                r = choose_root_midi(root_pc, base=56)
                pitches = [r, r+7]
                if is_up:
                    pitches = list(reversed(pitches))
                t = start
                for p in pitches:
                    dt = rng.uniform(0.004, 0.010)
                    inst_mute.notes.append(pretty_midi.Note(
                        velocity=int(gv),
                        pitch=int(np.clip(p, 0, 127)),
                        start=float(t),
                        end=float(end),
                    ))
                    t += dt
                continue

            # normal strum end time
            dur = (step_div * 0.85) * step_dur
            end = start + dur + overlap_ms * 1e-3

            # velocity humanize + down/up dynamics
            v = int(velocity + rng.integers(-vel_rand, vel_rand + 1))
            if hs == 0:
                v += 10  # beat1 강조
            v = int(np.clip(v, 40, 124))

            if is_up:
                v = int(v * 0.78)   # upstroke 약하게
                direction = "up"
            else:
                direction = "down"

            # money boost: 코러스에서 가끔 1을 3으로 승격(색채)
            vid = int(ch)
            if money_boost and vid == 1 and rng.random() < money_boost_prob:
                vid = 3

            pitches = voicing_pitches(root_pc, quality, vid)

            # voicing별 강도(기타스럽게)
            strength = 1.00
            if vid == 2: strength = 0.88
            if vid == 3: strength = 0.80
            if vid == 4: strength = 1.06

            add_strum(inst_clean, pitches, start, end, v, direction=direction, strength=strength)

    pm.instruments.append(inst_clean)
    pm.instruments.append(inst_mute)
    pm.write(out_path)
    print("Saved MIDI:", out_path)