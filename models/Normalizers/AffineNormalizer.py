import torch
from .Normalizer import Normalizer
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal



class AffineNormalizer(Normalizer):
    def __init__(self, mu=None, sigma=None, cat_dims=None):
        super(AffineNormalizer, self).__init__()
        self.cat_dims = cat_dims
        self.mu = mu
        self.sigma = sigma
        if mu is not None or sigma is not None:
            self.mu = nn.Parameter(mu, requires_grad=False)
            self.sigma = nn.Parameter(sigma, requires_grad=False)
        
        if self.cat_dims is not None:
            self.U_noise = MultivariateNormal(torch.zeros(len(self.cat_dims)), torch.eye(len(self.cat_dims))/(6*4))

    def forward(self, x, h, context=None):
        x0 = torch.zeros(x.shape).to(x.device)
        if self.cat_dims is not None:
#             keys = self.cat_dims.keys()
            keys = list(self.cat_dims)#.keys()
#             print(keys)
            with torch.no_grad():
                u_noise = torch.zeros(x.shape).to(x.device)
#                 if self.training:
# #                     u_noise[:,keys] = torch.rand(torch.Size([x.shape[0],len(keys)])).to(x.device)#.float()
#                     u_noise[:,keys] = torch.clamp(torch.randn(torch.Size([x.shape[0],len(keys)])).to(x.device)/6, max=1/2, min=-1/2)#.float()
#                     u_noise[:,keys] = torch.clamp(self.U_noise.sample(torch.Size([x.shape[0]])).to(x.device), max=1/2, min=-1/2)#.float()
    
#                 else:
#                     u_noise[:,keys] = torch.ones(torch.Size([x.shape[0],len(keys)])).to(x.device)*0.0#.float()
#             print(u_noise[:2,:])
            x = x + u_noise
        else:
            x = x
        mu, sigma = h[:, :, 0].clamp_(-5., 5.), torch.exp(h[:, :, 1].clamp_(-5., 2.))
        z = x * sigma + mu
        return z, sigma

    def inverse_transform(self, z, h, context=None):
        mu, sigma = h[:, :, 0].clamp_(-5., 5.), torch.exp(h[:, :, 1].clamp_(-5., 2.))
        x = (z - mu)/sigma
        return x
