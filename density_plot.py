import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import yaml
from utils.density_estimation import density_estimation

seednow = 2022
torch.manual_seed(seednow)
torch.cuda.manual_seed_all(seednow)
np.random.seed(seednow)
torch.backends.cudnn.deterministic = True

def parse_config():
    def dict2namespace(config):
        namespace = argparse.Namespace()
        for key, value in config.items():
            if isinstance(value, dict):
                new_value = dict2namespace(value)
            else:
                new_value = value
            setattr(namespace, key, new_value)
        return namespace

    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    parser.add_argument(
            "--config", type=str, default = "LRwaveform.yml", help="Path to the config file"
        )
    args = parser.parse_args()
    with open(os.path.join("configs", args.config), "r") as f:
            config = yaml.safe_load(f)
    new_config = dict2namespace(config)
    new_config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return new_config


config = parse_config()
device = config.device
datetimelabel = "0720"
plotarrange = 1 # plot 0--6 arange 

def scatterplot(method = "SIVI-SM"):
    """
    plot the comparison of SGLD for each three method.
    """
    figpos, axpos = plt.subplots(5, 5,figsize = (15,15), constrained_layout=False)
    X_sgld = torch.load("SGLD_LR/parallel_SGLD_LRwaveform.pt")
    if method == "SIVI-SM":
        X = torch.load('exp/{}/traceplot{}/SISM_sample.pt'.format(config.target_score, datetimelabel))
        col = "#ff7f0e"
    elif method == "SIVI":
        X = torch.load("exp/LRwaveform/SIVI/SIVI_sampling_SIVILR.pt")
        col = "#1f77b4"
    elif method == "UIVI":
        X = torch.load("exp/LRwaveform/UIVI/UIVI_sampling.pt")
        col = "#2ca02c"
    # 1-5 dimension
    for plotx in range(plotarrange, plotarrange+5):
        for ploty in range(plotarrange, plotarrange+5):
            if ploty != plotx:
                X1, Y1, Z = density_estimation(X[:,plotx].cpu().numpy(), X[:,ploty].cpu().numpy())
                axpos[plotx-plotarrange,ploty-plotarrange].contour(X1, Y1, Z,colors=col, alpha = 1)
                X1, Y1, Z = density_estimation(X_sgld[:,plotx].cpu().numpy(), X_sgld[:,ploty].cpu().numpy())
                axpos[plotx-plotarrange,ploty-plotarrange].contour(X1, Y1, Z,colors="black", alpha = 1)
                
                mean_x = X_sgld[:,plotx].mean()
                mean_y = X_sgld[:,ploty].mean()
                width_density = 1
                axpos[plotx-plotarrange,ploty-plotarrange].set_xlim(mean_x - width_density, mean_x + width_density)
                axpos[plotx-plotarrange,ploty-plotarrange].set_ylim(mean_y - width_density, mean_y + width_density)
                axpos[plotx-plotarrange,ploty-plotarrange].tick_params(labelsize = 17)
            else:
                sns.kdeplot(X[:,plotx].cpu().numpy(),shade=False,color=col,alpha = 1, ax = axpos[plotx-plotarrange, ploty-plotarrange], label = method).set(ylabel=None)
                sns.kdeplot(X_sgld[:,plotx].cpu().numpy(),shade=False,color="black",ax = axpos[plotx-plotarrange, ploty-plotarrange], label = "SGLD").set(ylabel=None)
                axpos[plotx-plotarrange,ploty-plotarrange].set_xlim(-2.0, 1.5)
                axpos[plotx-plotarrange,ploty-plotarrange].set_xticks([-2, -1, 0, 1])
                axpos[plotx-plotarrange,ploty-plotarrange].set_ylim(0, 1.9)
                axpos[plotx-plotarrange,ploty-plotarrange].tick_params(labelsize = 17)
                    
    figpos.tight_layout()
    plt.suptitle(method, fontsize= 70, y = 1.07)
    plt.savefig('exp/{}/traceplot{}/{}scatterplot_range{}-{}—new.jpg'.format(config.target_score, datetimelabel, method,plotarrange, plotarrange+5),dpi=120, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    scatterplot(method = "SIVI-SM")
    # scatterplot(method = "SIVI")
    # scatterplot(method = "UIVI")