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
    step_sec = sec_per_beat / steps_per_beat

    total_dur = bars * beats_per_bar * sec_per_beat
    times = np.linspace(0, total_dur, num=L, endpoint=False)

    tokens = np.zeros(L, dtype=np.int64)
    notes = sorted(inst.notes, key=lambda n: n.start)

    for i, t in enumerate(times):
        active = [n.pitch for n in notes if (n.start <= t < n.end)]
        if len(active) == 0:
            tokens[i] = REST_TOKEN
        else:
            pitch = max(active)
            tokens[i] = note_to_token(pitch)

    return tokens


def token_seq_to_midi(tokens, out_path, tempo=120.0, step_div=4, velocity=90):
    import pretty_midi
    import numpy as np

    pm = pretty_midi.PrettyMIDI(initial_tempo=float(tempo))
    inst = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program("Electric Guitar (clean)"))

    tokens = np.array(tokens, dtype=np.int64)

    # 16분음표 step 기준: 4/4에서 1 beat를 step_div(=4)로 쪼갠다고 가정
    # 1 beat duration = 60/tempo
    step_dur = (60.0 / float(tempo)) / float(step_div)

    # ===== HOLD 흉내(연속 pitch 합치기) =====
    cur_pitch = None
    cur_start = 0.0
    cur_end = 0.0

    def flush_note():
        nonlocal cur_pitch, cur_start, cur_end
        if cur_pitch is None:
            return
        # MIDI 안전 범위
        p = int(max(0, min(127, cur_pitch)))
        v = int(max(1, min(127, velocity)))
        if cur_end > cur_start:
            inst.notes.append(pretty_midi.Note(
                velocity=v, pitch=p, start=float(cur_start), end=float(cur_end)
            ))
        cur_pitch = None

    t = 0.0
    for tok in tokens:
        # 너 코드에 token_to_pitch가 있으면 그걸 그대로 쓰고 clamp만 하자
        pitch = token_to_pitch(int(tok))  # <= 기존 그대로 유지
        # pitch가 None/<=0이면 rest 취급
        if pitch is None or pitch <= 0:
            # rest가 나오면 지금까지 노트를 확정하고 끊기
            flush_note()
            t += step_dur
            continue

        pitch = int(pitch)

        if cur_pitch is None:
            # 새 노트 시작
            cur_pitch = pitch
            cur_start = t
            cur_end = t + step_dur
        else:
            if pitch == cur_pitch:
                # 같은 pitch 연속 -> end만 늘려서 HOLD처럼
                cur_end += step_dur
            else:
                # pitch 바뀜 -> 이전 노트 확정 후 새 노트 시작
                flush_note()
                cur_pitch = pitch
                cur_start = t
                cur_end = t + step_dur

        t += step_dur

    # 마지막 노트 확정
    flush_note()

    pm.instruments.append(inst)
    pm.write(out_path)
    print("Saved MIDI:", out_path)

