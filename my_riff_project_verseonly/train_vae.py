# VAE만 학습하는 거
import torch

from config import VOCAB_SIZE, SEQ_LEN, VAE_PATH
from dataset import get_dataloader
from models import RiffVAE, vae_loss_fn

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device)

    dataloader, _ = get_dataloader(batch_size=4, shuffle=True)

    vae = RiffVAE(vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN, z_dim=32).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

    epochs = 20
    beta = 0.1

    for ep in range(epochs):
        vae.train()
        total_loss = total_rec = total_kl = 0.0
        n = 0
        for x in dataloader:
            x = x.to(device)
            x_in = x  # teacher forcing input

            optimizer.zero_grad()
            logits, mu, logvar = vae(x, x_in)
            loss, rec, kl = vae_loss_fn(logits, x, mu, logvar, beta=beta)
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            total_loss += loss.item() * bs
            total_rec += rec.item() * bs
            total_kl += kl.item() * bs
            n += bs

        print(f"[VAE] Epoch {ep+1}/{epochs} | loss={total_loss/n:.4f} rec={total_rec/n:.4f} kl={total_kl/n:.4f}")

    torch.save(vae.state_dict(), VAE_PATH)
    print("Saved VAE to", VAE_PATH)

if __name__ == "__main__":
    main()
