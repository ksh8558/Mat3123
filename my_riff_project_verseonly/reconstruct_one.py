import torch
import numpy as np

from config import VOCAB_SIZE, SEQ_LEN, VAE_PATH
from models import RiffVAE
from midi_utils import midi_to_token_seq, token_seq_to_midi

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

midi_in = r"midi_riffs\californication_verse1a.mid"
midi_out = "recon_californication_verse1a.mid"

# load tokens
tokens = midi_to_token_seq(midi_in)
x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)  # (1, L)

# load vae
vae = RiffVAE(vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN, z_dim=32).to(device)
vae.load_state_dict(torch.load(VAE_PATH, map_location=device))
vae.eval()

with torch.no_grad():
    mu, logvar = vae.encode(x)
    z = mu
    x_in = torch.zeros(1, SEQ_LEN, dtype=torch.long, device=device)
    logits = vae.decode(z, x_in)
    out_tokens = torch.softmax(logits, dim=-1).argmax(dim=-1).squeeze(0).cpu().numpy().astype(np.int64)

token_seq_to_midi(out_tokens, midi_out, tempo=120.0)
print("Saved reconstruction:", midi_out)
