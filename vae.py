import torch, math, copy
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from scipy.stats import kde
import matplotlib.pyplot as plt

class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(2, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU()
        )

        self.mu = nn.Linear(50, latent_dim)
        self.logvar = nn.Linear(50, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 2)
        )

    def encode(self, x):
        encoded = self.encoder(x)
        mu = self.mu(encoded)
        logvar = self.logvar(encoded)
        return mu, logvar

    def sample(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + (eps * std)
        return z

    def decode(self, z):
        out = self.decoder(z)
        return out

    def forward(self, x):
        mu, logvar = self.encode(x) #dist params for each latent variable
        z = self.sample(mu, logvar) #sample latent vector from latent dist
        out = self.decode(z) #decode latent vector
        return mu, logvar, out

    def generate(self, n):
        z = torch.randn(n, self.latent_dim)
        samples = self.decode(z)
        return samples

def loss(x, out, mu, logvar, beta):

    diff = x - out
    latent_dim = len(logvar)

    #Compute reconstruction loss
    mse = nn.MSELoss()
    recons_loss = 0.5*(latent_dim*np.log(2*np.pi) + mse(x, out))

    #Compute KL loss
    kld_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

    #Compute total loss
    loss = recons_loss + beta * kld_loss

    return recons_loss, kld_loss, loss

def sample(n):

    mean = torch.randint(1, 4, (n,2), dtype=torch.float32)
    std = torch.ones(n,2)/100
    s = torch.normal(mean,std)

    return s

def plot_density(data):
    data = data.detach().numpy()
    nbins = 50
    x, y = data.T
    k = kde.gaussian_kde(data.T)
    xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.BuGn_r)

    plt.tight_layout()
    plt.show()
    plt.clf()

vae = VAE(100)
opt = torch.optim.Adam(vae.parameters(), lr=5e-4)

#training loop
beta = 0.01
for i in range(20000):
    s = sample(128)
    mu, logvar, out = vae(s)
    rl, kl, l = loss(s, out, mu, logvar, beta)
    opt.zero_grad()
    l.backward()
    opt.step()
    if i % 1000 == 0:
        data = vae.generate(5000)
        plot_density(data)
