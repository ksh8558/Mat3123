import pretty_midi

midi_path = "generated_riff_long.mid"

pm = pretty_midi.PrettyMIDI(midi_path)

print("=== BASIC ===")
print("instruments:", len(pm.instruments))
print("estimated tempo:", pm.estimate_tempo() if len(pm.get_tempo_changes()[1]) > 1 else "(tempo changes too small / unknown)")
print("end time (sec):", pm.get_end_time())

total_notes = 0
pitches = []
durations = []

for i, inst in enumerate(pm.instruments):
    n = len(inst.notes)
    total_notes += n
    print(f"\n--- Instrument {i} ---")
    print("program:", inst.program, "is_drum:", inst.is_drum, "notes:", n)
    for note in inst.notes[:10]:
        pitches.append(note.pitch)
        durations.append(note.end - note.start)
    if n > 0:
        pitches.extend([note.pitch for note in inst.notes])
        durations.extend([(note.end - note.start) for note in inst.notes])

print("\n=== NOTES SUMMARY ===")
print("total notes:", total_notes)

if total_notes == 0:
    print(">>> MIDI에 노트가 0개야. (token->midi 변환 or 생성 토큰이 전부 rest일 가능성)")
else:
    print("pitch min/max:", min(pitches), max(pitches))
    print("avg duration:", sum(durations)/len(durations))
    # 같은 pitch 비율이 높은지 대충 보기
    from collections import Counter
    c = Counter(pitches)
    most_pitch, most_cnt = c.most_common(1)[0]
    print("most common pitch:", most_pitch, "count:", most_cnt, f"({most_cnt/len(pitches)*100:.1f}%)")
