import os
import numpy as np
import matplotlib.pyplot as plt

from midi_utils import midi_to_token_seq, token_to_pitch
from config import SEQ_LEN, REST_TOKEN

def tokens_to_pitch_array(tokens: np.ndarray) -> np.ndarray:
    """
    token(0=rest, 1..)=pitch로 바꾼 배열.
    rest는 -1로 둬서 그래프에서 빈칸처럼 보이게.
    """
    pitches = []
    for tok in tokens:
        p = token_to_pitch(int(tok))
        pitches.append(-1 if p is None else p)
    return np.array(pitches, dtype=np.int64)

def save_pianoroll_like_plot(pitches: np.ndarray, out_png: str):
    """
    아주 단순한 시각화:
    x축=step(0..31), y축=pitch
    rest(-1)는 찍지 않음.
    """
    xs = np.arange(len(pitches))
    mask = pitches >= 0

    plt.figure()
    plt.title("Tokenized 2-bar verse (step vs pitch)")
    plt.xlabel("Step (16th-note grid)")
    plt.ylabel("MIDI Pitch")

    plt.scatter(xs[mask], pitches[mask], s=20)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--midi", required=True, help="2-bar MIDI file path")
    ap.add_argument("--png", default=None, help="output png path (optional)")
    args = ap.parse_args()

    tokens = midi_to_token_seq(args.midi)
    if len(tokens) != SEQ_LEN:
        print(f"[WARN] seq_len={len(tokens)} (expected {SEQ_LEN}). "
              f"잘린 구간/템포/그리드 설정을 확인해봐.")
    print("Tokens (len={}):".format(len(tokens)))
    print(tokens.tolist())

    pitches = tokens_to_pitch_array(tokens)
    print("\nPitches (-1 = rest):")
    print(pitches.tolist())

    if args.png is None:
        base = os.path.splitext(os.path.basename(args.midi))[0]
        args.png = base + "_tokens.png"

    save_pianoroll_like_plot(pitches, args.png)
    print("\nSaved plot:", args.png)

if __name__ == "__main__":
    main()
