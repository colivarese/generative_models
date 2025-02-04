import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torch.nn as nn
import torch



class VanillaVAE(nn.Module):
    def __init__(self, in_dim, latent_dim, hidden_dims):
        super(VanillaVAE, self).__init__()

        if not isinstance(hidden_dims, list):
            raise TypeError("hidden_dims must be a list")

        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims

        self._build_encoder()
        self._build_decoder()

    def _build_encoder(self):
        layers = []
        in_channels = self.in_dim
        for h_dim in self.hidden_dims:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=5, stride=2),
                    nn.BatchNorm2d(h_dim),
                    #nn.LeakyReLU())
                    nn.ReLU())
                )
            in_channels = h_dim
        
        self.encoder = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(self.hidden_dims[-1]*16, self.latent_dim)
        self.fc_sigma = nn.Linear(self.hidden_dims[-1]*16, self.latent_dim)

    def _build_decoder(self):

        
        self.decoder_input = nn.Linear(self.latent_dim, self.hidden_dims[-1] * 16)

        hidden_dims = self.hidden_dims[::-1]
        hidden_dims.append(1)
        layers = [] 
        for i in range(len(hidden_dims)-1):
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i+1], kernel_size=5, stride=2,),
                    nn.BatchNorm2d(hidden_dims[i+1]),
                    #nn.LeakyReLU())
                    nn.ReLU())
            )

        self.decoder = nn.Sequential(*layers)
        self.out_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=4),
            nn.BatchNorm2d(hidden_dims[-1]),
            #nn.ReLU(),
            #nn.Conv2d(hidden_dims[-2], out_channels=1, kernel_size=4),
            #nn.Tanh()
            nn.Sigmoid()
        )


    def encode(self, x):
            z = self.encoder(x)
            z = z.view(z.size(0), -1)
            mu = self.fc_mu(z)
            log_var = self.fc_sigma(z)
            return mu, log_var

    def reparametrize(self, mu, log_var):
            std = torch.exp(0.5*log_var)
            eps = torch.randn_like(std)
            return mu + (std * eps)

    def decode(self, z):
            z = self.decoder_input(z)
            z = z.view(z.size(0), self.hidden_dims[-1], 4, 4)
            out = self.decoder(z)
            out = self.out_layer(out)
            return out


    def forward(self, x):
            mu, log_var = self.encode(x)
            z = self.reparametrize(mu, log_var)
            out = self.decode(z)
            return out, mu, log_var

