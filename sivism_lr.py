import argparse
import os
import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import seaborn as sns
import torch
import torch.nn.functional as F
import yaml
from models.networks import Fnet, SIMINet
from models.target_models import target_distribution
from tqdm import tqdm
from utils.annealing import annealing
from utils.density_estimation import density_estimation


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
    parser.add_argument(
            "--baseline_sample", type=str, default = "SGLD_LR/parallel_SGLD_LRwaveform.pt", help="Path to the estimated samples generated from SGLD."
        )
    args = parser.parse_args()
    with open(os.path.join("configs", args.config), "r") as f:
            config = yaml.safe_load(f)
    new_config = dict2namespace(config)
    new_config.device = torch.device('cpu') # double precision
    new_config.baseline_sample = args.baseline_sample
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
        os.makedirs(os.path.join("exp", self.target, "traceplot{}".format(self.datetimelabel)),exist_ok=True)
        os.makedirs(os.path.join("exp", self.target, "model{}".format(self.datetimelabel)),exist_ok=True)
        
    def loaddata(self):
        # load the datasets
        if self.target in ["LRwaveform"]:
            data = scipy.io.loadmat('datasets/waveform.mat')
            
            X_train = data["X_train"]
            X_test = data["X_test"]
            y_train = data["y_train"]
            y_test = data["y_test"]
            self.X_train = torch.from_numpy(X_train).to(self.device).float()
            self.X_test = torch.from_numpy(X_test).to(self.device).float()
            self.y_train = torch.from_numpy(y_train).to(self.device).reshape(-1,1).float()
            self.y_test = torch.from_numpy(y_test).to(self.device).reshape(-1,1).float()

            self.size_train = X_train.shape[0]
            self.scale_sto = X_train.shape[0]/self.trainpara.sto_batchsize
            self.baseline_sample = torch.load("{}".format(self.config.baseline_sample))

    def learn(self):
        self.preprocess()
        self.loaddata()

        self.target_model = target_distribution[self.target](self.device)
        self.SemiVInet = SIMINet(self.trainpara, self.device).to(self.device)
        self.fnet = Fnet(self.trainpara).to(self.device)

        annealing_coef = lambda t: annealing(t, warm_up_interval = self.num_iters//self.trainpara.warm_ratio, anneal = self.trainpara.annealing)

        optimizer_VI = torch.optim.Adam(self.SemiVInet.parameters(), lr = self.trainpara.lr_SIMI, betas=(.9, .99))
        optimizer_f = torch.optim.Adam(self.fnet.parameters(), lr = self.trainpara.lr_f, betas=(.9, .99))

        scheduler_VI = torch.optim.lr_scheduler.StepLR(optimizer_VI, step_size=self.trainpara.gamma_step, gamma=self.trainpara.gamma)
        scheduler_f = torch.optim.lr_scheduler.StepLR(optimizer_f, step_size=self.trainpara.gamma_step, gamma=self.trainpara.gamma)

        fnetnorm_list = []
        loss_list = []
        test_loglik_list = []
  
        param_psi = 1.004
        psi = param_psi
        for epoch in tqdm(range(1, self.trainpara.num_epochs+1)):
            psi = psi/param_psi if self.trainpara.TransTrick else 0
            for i in range(1, self.trainpara.num_perepoch+1):
                self.iter_idx = (epoch-1) * self.trainpara.num_perepoch + i
                # ============================================================== #
                #                      Train the SemiVInet                       #
                # ============================================================== #
                Z = torch.randn([self.trainpara.batchsize, self.trainpara.z_dim]).to(self.device)
                X, neg_score_implicit = self.SemiVInet(Z)
                f_opt = self.fnet(X)
                compu_targetscore = self.target_model.score(X, self.X_train, self.y_train, self.scale_sto) * annealing_coef(self.iter_idx)
                g_opt = f_opt + compu_targetscore * psi
                loss = torch.mean(torch.sum(g_opt * (2.0 * compu_targetscore + 2.0 * neg_score_implicit - g_opt), -1))
                optimizer_VI.zero_grad()
                loss.backward()
                optimizer_VI.step()
                scheduler_VI.step()
                
                if epoch%10 == 0 and i % self.trainpara.train_vis_inepoch == 0:
                    print(("Epoch [{}/{}], min score matching[{}/{}], loss: {:.4f}").format(epoch, self.trainpara.num_epochs, i, self.trainpara.num_perepoch, loss.item()))
                # ============================================================ #
                #                        Train the fnet                        #
                # ============================================================ #
                for _ in range(self.trainpara.ftimes):
                    Z = torch.randn([self.trainpara.batchsize, self.trainpara.z_dim]).to(self.device)
                    X, neg_score_implicit = self.SemiVInet(Z)
                    f_opt = self.fnet(X.detach())
                    compu_targetscore = self.target_model.score(X.detach(), self.X_train, self.y_train, self.scale_sto) * annealing_coef(self.iter_idx)
                    g_opt = f_opt + compu_targetscore * psi
                    loss = - torch.mean(torch.sum(g_opt * (2.0 * compu_targetscore + 2.0 * neg_score_implicit.detach() - g_opt), -1))
                    
                    optimizer_f.zero_grad()
                    loss.backward()
                    optimizer_f.step()
                    scheduler_f.step()
                    
                    if epoch%10 == 0 and i % self.trainpara.train_vis_inepoch == 0:
                        print(("Epoch [{}/{}], max score matching[{}/{},{}], loss: {:.4f}, fnetnorm: {:.4f}").format(epoch, self.trainpara.num_epochs, i, self.trainpara.num_perepoch, _, -loss.item(), g_opt.norm(2, dim = 1).mean().item()))

            # compute some object in the trainging
            fnetnorm_list.append(np.array([self.iter_idx, (g_opt*g_opt).sum(1).mean().item()]))
            loss_list.append(np.array([self.iter_idx, -loss.item()]))   

            if epoch%self.config.sampling.visual_time ==0:
                # plot X scatter
                X = self.SemiVInet.sampling(num = self.config.sampling.num)
                
                # plot the scatter plot
                if epoch%(self.config.sampling.visual_time * 5) ==0:
                    plt.cla()
                    figpos, axpos = plt.subplots(5, 5,figsize = (15,15), constrained_layout=False)
                    for plotx in range(1,6):
                        for ploty in range(1,6):
                            if ploty != plotx:
                                X1, Y1, Z = density_estimation(X[:,plotx].cpu().numpy(), X[:,ploty].cpu().numpy())
                                axpos[plotx-1,ploty-1].contour(X1, Y1, Z,colors= "#ff7f0e")
                                X1, Y1, Z = density_estimation(self.baseline_sample[:,plotx].cpu().numpy(), self.baseline_sample[:,ploty].cpu().numpy())
                                axpos[plotx-1,ploty-1].contour(X1, Y1, Z,colors= 'black')
                            else:
                                sns.kdeplot(X[:,plotx].cpu().numpy(),fill=True,color= "#ff7f0e",ax = axpos[plotx-1, ploty-1], label="SIVISM").set(ylabel=None)
                                sns.kdeplot(self.baseline_sample[:,plotx].cpu().numpy(),fill=True,color= "black",ax = axpos[plotx-1, ploty-1], label="SGLD").set(ylabel=None)
                                axpos[plotx-1,ploty-1].legend()
                    figpos.tight_layout()
                    plt.savefig('exp/{}/traceplot{}/sample_scatterplot{}.jpg'.format(self.target, self.datetimelabel, self.iter_idx))
                    plt.close()
                    torch.save(X.cpu(), 'exp/{}/traceplot{}/SIVISM_sample.pt'.format(self.target, self.datetimelabel))
                
                # calculate the test_loglik
                with torch.no_grad():
                    test_loglik = self.target_model.logp(X.to(self.device), self.X_test, self.y_test).item()
                test_loglik_list.append(np.array([self.iter_idx, test_loglik]))
                print(("######### Epoch [{}/{}], loss: {:.4f}, test_loglik {:.4f}").format(epoch, self.trainpara.num_epochs, -loss, test_loglik_list[-1][1]))
            
        fnetnorm_list = np.array(fnetnorm_list)
        loss_list = np.array(loss_list)
        test_loglik_list = np.array(test_loglik_list)
        X = self.SemiVInet.sampling(num = self.config.sampling.num)
        torch.save(X.cpu().numpy(), 'exp/{}/traceplot{}/sample{}.pt'.format(self.target, self.datetimelabel, self.config.sampling.num))
        torch.save(fnetnorm_list, 'exp/{}/traceplot{}/fnetnorm_list.pt'.format(self.target, self.datetimelabel))
        torch.save(loss_list, 'exp/{}/traceplot{}/loss_list.pt'.format(self.target, self.datetimelabel))
        torch.save(test_loglik_list, 'exp/{}/traceplot{}/test_loglik_list.pt'.format(self.target, self.datetimelabel))

        torch.save(self.SemiVInet.state_dict(), "exp/{}/model{}/SemiVInet.ckpt".format(self.target, self.datetimelabel))
        torch.save(self.fnet.state_dict(), "exp/{}/model{}/fnet.ckpt".format(self.target, self.datetimelabel))
        return loss_list

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