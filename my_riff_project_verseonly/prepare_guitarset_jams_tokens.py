import os
import json
import argparse
import numpy as np
import jams


REST_TOKEN = 0

def load_notes_from_jams(jam: jams.JAMS):
    """GuitarSet JAMS의 note_midi(6 strings)를 합쳐 note events로 반환."""
    note_events = []
    # GuitarSet은 string별 note_midi annotation이 있음. :contentReference[oaicite:1]{index=1}
    anno_arr = jam.search(namespace="note_midi")
    for s in range(6):
        anns = anno_arr.search(data_source=str(s))
        if len(anns) == 0:
            continue
        ann = anns[0]
        intervals, values = ann.to_interval_values()
        # values: fractional midi note numbers 가능 :contentReference[oaicite:2]{index=2}
        for (st, ed), v in zip(intervals, values):
            if ed <= st:
                continue
            pitch = float(v)
            note_events.append((float(st), float(ed), pitch))
    note_events.sort(key=lambda x: x[0])
    return note_events


def load_beats_from_jams(jam: jams.JAMS):
    """beat_position에서 (time, position) 읽기. position==1이면 downbeat."""
    # mirdata GuitarSet loader도 beat_position namespace를 사용 :contentReference[oaicite:3]{index=3}
    anns = jam.search(namespace="beat_position")
    if len(anns) == 0:
        raise RuntimeError("beat_position annotation not found in JAMS")
    ann = anns[0]
    times, values = ann.to_event_values()
    positions = []
    for v in values:
        # v는 dict이고 "position" 필드를 가짐 :contentReference[oaicite:4]{index=4}
        positions.append(int(v["position"]))
    return list(map(float, times)), positions


def build_16th_grid(times, positions, seg_start, seg_end, bars=4, subdiv=4):
    """
    seg_start~seg_end 구간의 beat_position 이벤트를 이용해
    4/4 기준 bar 내부 beat를 16분(subdiv=4)로 쪼개 grid time을 만든다.
    """
    # segment 내 beat만 뽑기
    bt = [(t, p) for (t, p) in zip(times, positions) if (seg_start <= t < seg_end)]
    bt.sort(key=lambda x: x[0])

    # downbeat(=position 1)들 추출
    downbeats = [t for (t, p) in bt if p == 1]
    # seg_start가 downbeat가 아니면, seg_start를 downbeat로 취급(안전장치)
    if len(downbeats) == 0 or abs(downbeats[0] - seg_start) > 1e-3:
        downbeats = [seg_start] + downbeats

    # bar별 beat time dict 만들기: bar i의 position 1~4 time
    # bt에는 bar들을 순서대로 포함되어 있다고 가정(클릭 트랙 기반) :contentReference[oaicite:5]{index=5}
    grid_times = []
    # seg_start 기준으로 bars개 bar를 만들기 위해 downbeat 경계가 필요
    # seg_end는 이미 seg_start 이후 bars개의 downbeat로 잡을 예정이므로 여기선 bt 기반으로 생성
    # 각 bar마다: beat1~beat4 time을 모으고, beat5는 다음 bar downbeat로 대체
    current_bar_start = seg_start
    for bar_idx in range(bars):
        bar_start = current_bar_start
        bar_end = None

        # 다음 downbeat를 bar_end로 사용
        cand = [t for t in downbeats if t > bar_start + 1e-6]
        if len(cand) == 0:
            # fallback: 균등하게 2초짜리 bar 가정(tempo 불명일 때)
            bar_end = bar_start + 2.0
        else:
            bar_end = cand[0]

        # bar 안의 beat1~beat4 time 찾기 (없으면 균등 분할 fallback)
        beats_in_bar = [(t, p) for (t, p) in bt if (bar_start <= t < bar_end)]
        # position별로
        beat_times = {p: t for (t, p) in beats_in_bar if p in (1, 2, 3, 4)}
        if 1 not in beat_times:
            beat_times[1] = bar_start

        # beat2~4가 없으면 균등 분할로 채움
        for p in (2, 3, 4):
            if p not in beat_times:
                beat_times[p] = bar_start + (p-1) * (bar_end - bar_start) / 4.0

        # beat5는 다음 downbeat(=bar_end)
        beat_times[5] = bar_end

        # 각 beat를 subdiv로 쪼개 16분 그리드 생성
        for p in (1, 2, 3, 4):
            b0 = beat_times[p]
            b1 = beat_times[p+1]
            dur = b1 - b0
            for k in range(subdiv):
                grid_times.append(b0 + dur * (k / subdiv))

        current_bar_start = bar_end

    return grid_times  # 길이 = bars * 4 * subdiv = 64


def notes_to_tokens(note_events, grid_times, choose="highest"):
    """
    각 grid step 시작 시점에서 active note를 찾아 토큰화.
    choose="highest": 여러 개면 최고 pitch를 선택(모노화)
    """
    tokens = np.zeros(len(grid_times), dtype=np.int16)

    # 간단 O(steps*notes) (steps=64라 충분히 빠름)
    for i, t in enumerate(grid_times):
        active = [p for (st, ed, p) in note_events if (st <= t < ed)]
        if not active:
            tokens[i] = REST_TOKEN
            continue
        if choose == "highest":
            pitch = max(active)
        else:
            pitch = active[-1]
        # fractional midi note number -> int로 반올림 :contentReference[oaicite:6]{index=6}
        midi_pitch = int(np.rint(pitch))
        if midi_pitch < 0 or midi_pitch > 127:
            tokens[i] = REST_TOKEN
        else:
            tokens[i] = midi_pitch  # 0은 rest로 쓰고, 실제 pitch는 1+ 영역이라 충돌 거의 없음
    return tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jams_dir", required=True, help="GuitarSet의 .jams 파일들이 있는 폴더")
    ap.add_argument("--out_dir", default="guitarset_tokens_4bars", help="출력 폴더(.npy)")
    ap.add_argument("--bars", type=int, default=4, help="한 샘플 bar 수(기본 4)")
    ap.add_argument("--subdiv", type=int, default=4, help="beat 당 subdivision(기본 4=16분)")
    ap.add_argument("--hop_bars", type=int, default=2, help="슬라이딩 hop(bar). 기본 2면 (A 2마디 + A' 2마디) 느낌으로 겹침")
    ap.add_argument("--max_per_file", type=int, default=20, help="각 jams에서 최대 몇 샘플 뽑을지")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    meta_path = os.path.join(args.out_dir, "meta.jsonl")

    jams_files = [f for f in os.listdir(args.jams_dir) if f.lower().endswith(".jams")]
    jams_files.sort()

    total = 0
    with open(meta_path, "w", encoding="utf-8") as mf:
        for jf in jams_files:
            jpath = os.path.join(args.jams_dir, jf)
            try:
                jam = jams.load(jpath)
                note_events = load_notes_from_jams(jam)
                times, positions = load_beats_from_jams(jam)
            except Exception as e:
                print(f"[SKIP] {jf}: {e}")
                continue

            # downbeat(=position 1)로 bar 경계 만들기
            downbeats = [t for (t, p) in zip(times, positions) if p == 1]
            downbeats.sort()
            if len(downbeats) < args.bars + 1:
                print(f"[SKIP] {jf}: not enough downbeats")
                continue

            # 4-bar segment를 hop_bars씩 슬라이딩하며 추출
            count_this = 0
            for i in range(0, len(downbeats) - (args.bars + 0), args.hop_bars):
                if i + args.bars >= len(downbeats):
                    break
                seg_start = downbeats[i]
                seg_end = downbeats[i + args.bars]

                grid = build_16th_grid(times, positions, seg_start, seg_end,
                                       bars=args.bars, subdiv=args.subdiv)
                if len(grid) != args.bars * 4 * args.subdiv:
                    continue

                tokens = notes_to_tokens(note_events, grid, choose="highest")
                # 전부 rest면 버림
                if np.all(tokens == REST_TOKEN):
                    continue

                base = os.path.splitext(jf)[0]
                out_name = f"{base}_b{i:03d}_{args.bars}bars.npy"
                out_path = os.path.join(args.out_dir, out_name)
                np.save(out_path, tokens)

                mf.write(json.dumps({
                    "src_jams": jf,
                    "segment_index": i,
                    "bars": args.bars,
                    "subdiv": args.subdiv,
                    "seg_start": seg_start,
                    "seg_end": seg_end,
                    "out": out_name,
                    "nonrest": int(np.sum(tokens != REST_TOKEN)),
                }, ensure_ascii=False) + "\n")

                total += 1
                count_this += 1
                if count_this >= args.max_per_file:
                    break

            print(f"[OK] {jf}: {count_this} samples")

    print(f"\nDONE. total_samples={total}")
    print(f"Saved to: {args.out_dir}")
    print(f"Meta: {meta_path}")


if __name__ == "__main__":
    main()
