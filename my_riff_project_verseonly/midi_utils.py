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


def token_seq_to_midi(
    tokens: np.ndarray,
    out_path: str,
    bars: int = BARS,
    beats_per_bar: int = BEATS_PER_BAR,
    steps_per_beat: int = STEPS_PER_BEAT,
    tempo: float = 120.0,
    program: int = 27,  # Electric Guitar (Jazz) 정도
):
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=program)

    sec_per_beat = 60.0 / tempo
    step_sec = sec_per_beat / steps_per_beat

    L = len(tokens)
    time = 0.0
    current_pitch = None
    note_start = None

    def flush_note():
        nonlocal current_pitch, note_start, time
        if current_pitch is not None and note_start is not None:
            note = pretty_midi.Note(
                velocity=90,
                pitch=current_pitch,
                start=note_start,
                end=time
            )
            inst.notes.append(note)

    for tok in tokens:
        pitch = token_to_pitch(int(tok))
        if pitch is not None:
            if current_pitch is None:
                current_pitch = pitch
                note_start = time
            elif current_pitch != pitch:
                flush_note()
                current_pitch = pitch
                note_start = time
        else:
            if current_pitch is not None:
                flush_note()
                current_pitch = None
                note_start = None
        time += step_sec

    flush_note()
    pm.instruments.append(inst)
    pm.write(out_path)
    print("Saved MIDI:", out_path)
