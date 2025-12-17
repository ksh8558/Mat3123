# VAE + Latent Diffusion 전부 한 파일에.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RiffVAE(nn.Module):
    def __init__(self, vocab_size, seq_len=32,
                 emb_dim=64, hidden_dim=128, z_dim=32):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.emb = nn.Embedding(vocab_size, emb_dim)

        self.encoder_rnn = nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        self.fc_mu = nn.Linear(hidden_dim, z_dim)
        self.fc_logvar = nn.Linear(hidden_dim, z_dim)

        self.decoder_rnn = nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.fc_z_to_h = nn.Linear(z_dim, hidden_dim)

    def encode(self, x):
        emb = self.emb(x)
        _, h_n = self.encoder_rnn(emb)
        h_n = h_n.squeeze(0)
        mu = self.fc_mu(h_n)
        logvar = self.fc_logvar(h_n)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, x_in):
        emb = self.emb(x_in)
        h0 = self.fc_z_to_h(z).unsqueeze(0)
        out, _ = self.decoder_rnn(emb, h0)
        logits = self.fc_out(out)
        return logits

    def forward(self, x, x_in):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z, x_in)
        return logits, mu, logvar


def vae_loss_fn(logits, target, mu, logvar, beta=0.1):
    rec_loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        target.view(-1),
        reduction="mean"
    )
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl = kl.mean()
    return rec_loss + beta * kl, rec_loss, kl


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb_scale)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb


class LatentDiffusionMLP(nn.Module):
    def __init__(self, z_dim, time_dim=64, hidden_dim=128):
        super().__init__()
        self.time_mlp = TimeEmbedding(time_dim)
        self.fc1 = nn.Linear(z_dim + time_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, z_dim)

    def forward(self, z_t, t):
        emb_t = self.time_mlp(t)
        x = torch.cat([z_t, emb_t], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        eps_hat = self.fc3(x)
        return eps_hat


class LatentDiffusionTrainer:
    def __init__(self, model, T=200, z_dim=32, device="cpu"):
        self.model = model
        self.T = T
        self.device = device
        betas = torch.linspace(1e-4, 0.02, T)
        self.betas = betas.to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, z0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(z0)
        alpha_bar_t = self.alpha_bar[t].view(-1, 1)
        return torch.sqrt(alpha_bar_t) * z0 + torch.sqrt(1 - alpha_bar_t) * noise

    def p_losses(self, z0, t):
        noise = torch.randn_like(z0)
        z_t = self.q_sample(z0, t, noise)
        eps_hat = self.model(z_t, t)
        return F.mse_loss(eps_hat, noise)
