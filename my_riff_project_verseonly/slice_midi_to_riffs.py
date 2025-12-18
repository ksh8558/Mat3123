# slice_midi_to_riffs.py
import os
import argparse
import numpy as np
import pretty_midi

# 네 프로젝트 토큰화 함수/상수 재사용
from midi_utils import midi_to_token_seq
from config import SEQ_LEN, REST_TOKEN

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="guitarmidi", help="입력 MIDI 폴더")
    ap.add_argument("--out_dir", default="midi_riffs_sliced", help="출력 폴더(.npy)")
    ap.add_argument("--seq_len", type=int, default=SEQ_LEN, help="샘플 길이(step 단위)")
    ap.add_argument("--hop", type=int, default=None, help="슬라이딩 hop(step). 기본=seq_len//2")
    ap.add_argument("--min_note_ratio", type=float, default=0,
                    help="샘플 안에서 rest가 아닌 비율 최소값(너무 텅 빈 샘플 제거)")
    ap.add_argument("--max_samples_per_file", type=int, default=999999,
                    help="MIDI 1개에서 최대 몇 개 샘플까지 뽑을지 제한")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    hop = args.hop if args.hop is not None else max(1, args.seq_len // 2)

    midi_files = [f for f in os.listdir(args.in_dir) if f.lower().endswith(".mid")]
    midi_files.sort()
    if not midi_files:
        raise RuntimeError(f"No .mid files found in {args.in_dir}")

    total_written = 0
    total_skipped = 0

    for fn in midi_files:
        path = os.path.join(args.in_dir, fn)

        # midi_to_token_seq는 "파일 전체를 토큰 시퀀스로" 뽑는다고 가정
        tokens = np.array(midi_to_token_seq(path), dtype=np.int64)

        if len(tokens) < args.seq_len:
            # 너무 짧으면 pad해서 1개만
            pad = np.full(args.seq_len - len(tokens), REST_TOKEN, dtype=np.int64)
            chunk = np.concatenate([tokens, pad])
            note_ratio = float(np.mean(chunk != REST_TOKEN))
            if note_ratio >= args.min_note_ratio:
                out_name = os.path.splitext(fn)[0] + f"_s0.npy"
                np.save(os.path.join(args.out_dir, out_name), chunk)
                total_written += 1
            else:
                total_skipped += 1
            continue

        base = os.path.splitext(fn)[0]
        c = 0
        for start in range(0, len(tokens) - args.seq_len + 1, hop):
            chunk = tokens[start:start + args.seq_len]
            note_ratio = float(np.mean(chunk != REST_TOKEN))
            if note_ratio < args.min_note_ratio:
                total_skipped += 1
                continue

            out_name = f"{base}_s{start}.npy"
            np.save(os.path.join(args.out_dir, out_name), chunk)
            total_written += 1
            c += 1
            if c >= args.max_samples_per_file:
                break

    print("DONE")
    print("out_dir:", os.path.abspath(args.out_dir))
    print("written:", total_written)
    print("skipped:", total_skipped)
    print("seq_len:", args.seq_len, "hop:", hop, "min_note_ratio:", args.min_note_ratio)

if __name__ == "__main__":
    main()
