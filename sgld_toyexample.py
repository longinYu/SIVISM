import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from models.target_models import target_distribution
import os


def SGLD_toy(loop = 10000, Z = torch.zeros([1,7850]), epsilon_0 = 1e-1, alpha = 0):
    for t in tqdm(range(0, loop)):
        compu_targetscore = model.score(Z)
        learn_rate = epsilon_0/(1+t)**alpha
        Z = Z + learn_rate/2 * compu_targetscore + np.sqrt(learn_rate) * torch.randn([Z.shape[0],2]).to(device)
    return Z.cpu()


if __name__ == "__main__":
    os.makedirs(os.path.join("SGLD_TOY"),exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataname = "banana"
    model = target_distribution[dataname](device)
    Z = torch.zeros([100000,2]).to(device) 
    trace_SGLD = SGLD_toy(loop = 500000, Z = Z, epsilon_0 = 5 * 1e-4, alpha = 0)
    torch.save(trace_SGLD, "SGLD_TOY/parallel_SGLD_{}.pt".format(dataname))
    