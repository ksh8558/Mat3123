# VAE encoder로 latent 뽑아서 diffusion 학습.
import torch

from config import VOCAB_SIZE, SEQ_LEN, VAE_PATH, DIFF_PATH
from dataset import get_dataloader
from models import RiffVAE, LatentDiffusionMLP, LatentDiffusionTrainer

def extract_latents(vae, dataloader, device):
    vae.eval()
    zs = []
    with torch.no_grad():
        for x in dataloader:
            x = x.to(device)
            mu, logvar = vae.encode(x)
            z = mu  # or reparameterize
            zs.append(z.cpu())
    return torch.cat(zs, dim=0)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device)

    dataloader, dataset = get_dataloader(batch_size=64, shuffle=False)

    vae = RiffVAE(vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN, z_dim=32).to(device)
    vae.load_state_dict(torch.load(VAE_PATH, map_location=device))
    print("Loaded VAE from", VAE_PATH)

    z_dataset = extract_latents(vae, dataloader, device)
    print("Latent dataset shape:", z_dataset.shape)
    z_dim = z_dataset.shape[1]

    T = 200
    ld_model = LatentDiffusionMLP(z_dim=z_dim).to(device)
    ld_trainer = LatentDiffusionTrainer(ld_model, T=T, z_dim=z_dim, device=device)
    optimizer = torch.optim.Adam(ld_model.parameters(), lr=1e-3)

    epochs = 30
    batch_size = 64
    N = z_dataset.size(0)

    for ep in range(epochs):
        perm = torch.randperm(N)
        total_loss = 0.0
        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            z0 = z_dataset[idx].to(device)
            t = torch.randint(0, T, (z0.size(0),), device=device)

            optimizer.zero_grad()
            loss = ld_trainer.p_losses(z0, t)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * z0.size(0)

        print(f"[Diffusion] Epoch {ep+1}/{epochs} | loss={total_loss/N:.4f}")

    torch.save(ld_model.state_dict(), DIFF_PATH)
    print("Saved diffusion model to", DIFF_PATH)

if __name__ == "__main__":
    main()
