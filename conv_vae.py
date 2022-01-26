import torch, math, copy
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
from torchvision import datasets, transforms
from scipy.stats import kde

class ConvVAE(nn.Module):
    def __init__(self, latent_dim, hidden_dim=1024):
        super(ConvVAE, self).__init__()

        self.latent_dim = latent_dim
        nc=1; ndf=8; ngf=8

        self.encoder = nn.Sequential(

            #Layer 1
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            #Layer 2
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            #Layer 3
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            #Layer 4
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.decoder = nn.Sequential(

            #Layer 1
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            #Layer 2
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            #Layer 3
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            #Layer 4
            nn.ConvTranspose2d(ngf, 1, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(True),
        )

        self.mu = nn.Linear(1024, latent_dim)
        self.logvar = nn.Linear(1024, latent_dim)
        self.hidden = nn.Linear(latent_dim, 1024)

    def sample(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + (eps * std)
        return z

    def encode(self, input):
        conv = self.encoder(input)
        conv = conv.view(-1, 1024)
        mu, logvar = self.mu(conv), self.logvar(conv)
        return mu, logvar

    def decode(self, z):
        x = self.hidden(z)
        x = x.view(-1,64,4,4)
        out = self.decoder(x)
        return out

    def forward(self, input):

        mu, logvar = self.encode(input)
        z = self.sample(mu, logvar)
        decoded = self.decode(z)

        return mu, logvar, decoded

    def generate(self, n):
        z = torch.randn(n, self.latent_dim).cuda()
        samples = self.decode(z)
        return samples

transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(64), transforms.CenterCrop(64), transforms.Normalize((0.5,), (0.5,))])
dataset = datasets.MNIST("data", download=True, transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

def show(x):
    img = x.data.cpu().permute(1, 2, 0).numpy() * 255
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow(img)

#training loop
beta = 2
for epoch in range(100):
    for i, x in enumerate(loader):
        if len(x) == 2:
            x = x[0]
        x = x.cuda()

        mu, logvar, out = vae(x)
        rl, kl, l = loss(x, out, mu, logvar, beta)

        opt.zero_grad()
        l.backward()
        opt.step()

        if i == 0:
            vae.eval()
            data = vae.generate(8)
            grid_img = torchvision.utils.make_grid(data, nrow=8, normalize=True)
            show(grid_img)
            vae.train()
