import argparse
import os
import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from models.networks import Fnet, SIMINet
from models.target_models import target_distribution
from tqdm import tqdm
from utils.annealing import annealing


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
            "--config", type=str, default = "multimodal.yml", help="Path to the config file"
        )
    args = parser.parse_args()
    # args.config = "{}.yml".format(dataname)
    with open(os.path.join("configs", args.config), "r") as f:
            config = yaml.safe_load(f)
    new_config = dict2namespace(config)
    new_config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return new_config


class SIVISM(object):
    def __init__(self, date = "", config = None):
        self.config = parse_config() if not config else config
        self.datetimelabel = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') if not date else date
        self.device = self.config.device
        self.target = self.config.target_score
        self.trainpara = self.config.train
        self.num_iters = self.trainpara.num_perepoch * self.config.train.num_epochs
        self.iter_idx = 0
        
    def preprocess(self):
        os.makedirs(os.path.join("exp", self.target, "fig{}".format(self.datetimelabel)),exist_ok=True)
        
    def learn(self):
        self.preprocess()

        self.target_model = target_distribution[self.target](self.device)
        self.SemiVInet = SIMINet(self.trainpara, self.device).to(self.device)
        self.fnet = Fnet(self.trainpara).to(self.device)

        annealing_coef = lambda t: annealing(t, warm_up_interval = self.num_iters//self.trainpara.warm_ratio, anneal = self.trainpara.annealing)

        # Select the optimizer
        optimizer_VI = torch.optim.Adam(self.SemiVInet.parameters(), lr = self.trainpara.lr_SIMI, betas=(.9, .99))
        optimizer_f = torch.optim.Adam(self.fnet.parameters(), lr = self.trainpara.lr_f, betas=(.9, .99))

        scheduler_VI = torch.optim.lr_scheduler.StepLR(optimizer_VI, step_size=self.trainpara.gamma_step, gamma=self.trainpara.gamma)
        scheduler_f = torch.optim.lr_scheduler.StepLR(optimizer_f, step_size=self.trainpara.gamma_step, gamma=self.trainpara.gamma)


        for epoch in tqdm(range(1, self.trainpara.num_epochs+1)):

            if (epoch - 1)%self.config.sampling.visual_time ==0 or epoch ==self.trainpara.num_epochs:
                X = self.SemiVInet.sampling(num = self.config.sampling.num)
                if self.target in ["banana", "multimodal", "x_shaped"]:
                    plt.cla()
                    fig, ax = plt.subplots(figsize=(5, 5))
                    save_to_path = 'exp/{}/fig{}/test{}.jpg'.format(self.target, self.datetimelabel, self.iter_idx)
                    bbox = {"multimodal":[-5, 5, -5, 5], "banana":[-3.5,3.5,-6,1], "x_shaped":[-5,5,-5,5]}
                    quiver_plot = True if epoch < self.trainpara.num_epochs else False
                    self.target_model.contour_plot(bbox[self.target], ax, self.fnet, ngrid=100, samples=X.cpu().numpy(), save_to_path=save_to_path, quiver = quiver_plot, t = self.iter_idx)

            for i in range(1, self.trainpara.num_perepoch+1):
                self.iter_idx = (epoch-1) * self.trainpara.num_perepoch + i
                # ============================================================== #
                #                      Train the SemiVInet                       #
                # ============================================================== #
                Z = torch.randn([self.trainpara.batchsize, self.trainpara.z_dim]).to(self.device)
                X, neg_score_implicit = self.SemiVInet(Z)
                f_opt = self.fnet(X)
                compu_targetscore = self.target_model.score(X) * annealing_coef(self.iter_idx)
                g_opt = f_opt
                loss = torch.mean(torch.sum(g_opt * (2.0 * compu_targetscore + 2.0 * neg_score_implicit - g_opt), -1))
                optimizer_VI.zero_grad()
                loss.backward()
                optimizer_VI.step()
                scheduler_VI.step()
                # ============================================================ #
                #                        Train the fnet                        #
                # ============================================================ #
                for _ in range(self.trainpara.ftimes):
                    Z = torch.randn([self.trainpara.batchsize, self.trainpara.z_dim]).to(self.device)
                    X, neg_score_implicit = self.SemiVInet(Z)
                    f_opt = self.fnet(X)
                    compu_targetscore = self.target_model.score(X) * annealing_coef(self.iter_idx)
                    g_opt = f_opt
                    loss = - torch.mean(torch.sum(g_opt * (2.0 * compu_targetscore + 2.0 * neg_score_implicit.detach() - g_opt), -1))
                    optimizer_f.zero_grad()
                    loss.backward()
                    optimizer_f.step()
                    scheduler_f.step()
if __name__ == "__main__":
    seednow = 2022
    torch.manual_seed(seednow)
    torch.cuda.manual_seed_all(seednow)
    np.random.seed(seednow)
    random.seed(seednow)
    torch.backends.cudnn.deterministic = True
    config = parse_config()
    task = SIVISM("",config=config)
    task.learn()


