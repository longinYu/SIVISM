import torch
import torch.nn as nn

class SIMINet(nn.Module):
    def __init__(self, namedict, device):
        """
        the optimizer paprameters for sigma is the log_var
        """
        super(SIMINet, self).__init__()
        self.z_dim = namedict.z_dim
        self.h_dim = namedict.h_dim
        self.out_dim = namedict.out_dim

        self.mu = nn.Sequential(
            nn.Linear(self.z_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.out_dim)
        )
        self.log_var = nn.Parameter(torch.zeros(namedict.out_dim) + namedict.log_var_ini, requires_grad = True)
        self.device = device
        self.log_var_min = namedict.log_var_min
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(mu)
        return mu + std * eps, eps/std
    
    def forward(self, Z):
        mu = self.mu(Z)
        log_var = self.log_var.clamp(self.log_var_min)
        X, neg_score_implicit = self.reparameterize(mu, log_var)
        return X, neg_score_implicit

    def sampling(self, num = 1000, sigma = 1):
        with torch.no_grad():
            Z = torch.randn([num, self.z_dim], ).to(self.device)
            Z = Z * sigma
            X, _ = self.forward(Z)
        return X



class Fnet(nn.Module):
    def __init__(self, namedict):
        super(Fnet, self).__init__()
        self.fnet = nn.Sequential(
                    nn.Linear(namedict.out_dim, namedict.f_dim),
                    nn.ReLU(),
                    nn.Linear(namedict.f_dim, namedict.f_dim),
                    nn.ReLU(),
                    nn.Linear(namedict.f_dim, namedict.out_dim),
                )
    def forward(self, X):
        return self.fnet(X)