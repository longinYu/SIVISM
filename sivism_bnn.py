import argparse
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import yaml
from tqdm import tqdm
from utils.annealing import annealing
from models.target_models import target_distribution
from models.networks import Fnet, SIMINet
import torch

from sklearn.model_selection import train_test_split


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
            "--config", type=str, default = "Bnn_boston.yml", help="Path to the config file"
        )
    args = parser.parse_args()
    with open(os.path.join("configs", args.config), "r") as f:
            config = yaml.safe_load(f)
    new_config = dict2namespace(config)
    new_config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return new_config


class SIVISM(object):
    def __init__(self, date = "True", config = None, permu_i=6):
        self.config = parse_config() if not config else config

        self.datetimelabel = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') if not date else date
        self.device = self.config.device
        self.target = self.config.target_score
        self.trainpara = self.config.train
        self.num_iters = self.trainpara.num_perepoch * self.config.train.num_epochs
        self.iter_idx = 0
        self.permu_i = permu_i
        
        
    def preprocess(self):
        os.makedirs(os.path.join("exp", self.target, "traceplot{}".format(self.datetimelabel)),exist_ok=True)
        os.makedirs(os.path.join("exp", self.target, "model{}".format(self.datetimelabel)),exist_ok=True)
        os.makedirs(os.path.join("exp", self.target, "logfile{}".format(self.datetimelabel)),exist_ok=True)
        
    def loaddata(self):
        # load the datasets
        
        data = np.loadtxt('datasets/boston_housing.txt')
        X_input = torch.from_numpy(data[ :, range(data.shape[1] - 1) ]).to(self.device).float()
        y_input = torch.from_numpy(data[ :, data.shape[1] - 1 ]).to(self.device).float()
        
        ## select the permutaion by train_test_split, 
        train_ratio = 0.9
        X_train, X_test, y_train, y_test = train_test_split(X_input, y_input, test_size= 1-train_ratio, random_state=42)
        # The hyperparameter loggamma and loglambda is selected  by MCMC method, like SGLD [1] or SVGD [2].
        # [1] Welling, Max, and Yee W. Teh. "Bayesian learning via stochastic gradient Langevin dynamics." Proceedings of the 28th international conference on machine learning (ICML-11). 2011.
        # [2] Liu, Qiang, and Dilin Wang. "Stein variational gradient descent: A general purpose bayesian inference algorithm." Advances in neural information processing systems 29 (2016). 
        self.loglambda_hyp = -0.9809319716071633
        self.loggamma_hyp = -2.55156665423909

        # from svgd_bnn_hyperparam import svgd_bayesnn
        # svgd = svgd_bayesnn(X_train.cpu().numpy(), y_train.cpu().numpy(), M = 100, batch_size = 100, n_hidden = 50, max_iter = 2000, master_stepsize = 1e-3)
        # self.loggamma_hyp = np.mean(svgd.theta[:,-2])
        # self.loglambda_hyp = np.mean(svgd.theta[:,-1])
        # print(self.loglambda_hyp, self.loggamma_hyp)
        
        

        
        y_train = y_train[:,None]
        y_test = y_test[:,None]

        size_dev = min(int(np.round(0.1 * X_train.shape[0])), 500)
        X_dev, y_dev = X_train[-size_dev:], y_train[-size_dev:]
        X_train, y_train = X_train[:-size_dev], y_train[:-size_dev]

        X_train_mean = X_train.mean(0)
        y_train_mean = y_train.mean(0)
        X_train_std = X_train.std(0)
        y_train_std = y_train.std(0)

        # normalization
        self.X_train = (X_train - X_train_mean)/X_train_std
        self.y_train = (y_train - y_train_mean)/y_train_std
        self.X_test = (X_test - X_train_mean)/X_train_std
        self.y_test = y_test
        self.X_dev = (X_dev - X_train_mean)/X_train_std
        self.y_dev = y_dev

        self.y_train_mean = y_train_mean
        self.y_train_std = y_train_std
        
        self.size_train = X_train.shape[0]
        self.scale_sto = self.X_train.shape[0]/self.trainpara.sto_batchsize

    def learn(self, model_selection = False):
        self.preprocess()
        self.loaddata()

        # Select the style of networks and the device to trainging 
        self.target_model = target_distribution[self.target](self.device, self.X_train.shape[1], loglambda = self.loglambda_hyp, loggamma = self.loggamma_hyp)
        self.SemiVInet = SIMINet(self.trainpara, self.device).to(self.device)
        self.fnet = Fnet(self.trainpara).to(self.device)

        # Choose a training strategy
        annealing_coef = lambda t: annealing(t, warm_up_interval = self.num_iters//self.trainpara.warm_ratio, anneal = self.trainpara.annealing)

        # Select the optimizer
        optimizer_VI = torch.optim.Adam([{'params':self.SemiVInet.mu.parameters(),'lr': self.trainpara.lr_SIMI},
                              {'params':self.SemiVInet.log_var,'lr': self.trainpara.lr_SIMI_var}])
        optimizer_f = torch.optim.Adam(self.fnet.parameters(), lr = self.trainpara.lr_f, betas=(.9, .99))

        scheduler_VI = torch.optim.lr_scheduler.StepLR(optimizer_VI, step_size=self.trainpara.gamma_step, gamma=self.trainpara.gamma)
        scheduler_f = torch.optim.lr_scheduler.StepLR(optimizer_f, step_size=self.trainpara.gamma_step, gamma=self.trainpara.gamma)

        param_psi = 1.05
        psi = param_psi # trans trick for fnet.

        '''initialize the phinet'''
        optimizer_pre = torch.optim.Adam([{'params':self.SemiVInet.mu.parameters(),'lr': 1e-2},
                            {'params':self.SemiVInet.log_var,'lr': 1e-2}])
        for epoch in range(1, 100):
            Z = torch.randn([self.trainpara.batchsize, self.trainpara.z_dim]).to(self.device) * self.trainpara.sigma_ini
            X, _ = self.SemiVInet(Z)

            ## Use the X_dev dataset to initialize phinet
            # predicty_z = self.target_model.predict_y(X, self.X_dev, self.y_train_mean, self.y_train_std)
            # loss = ((predicty_z.mean(0) - self.y_dev)**2).mean()

            # Use the X_train dataset to initialize phinet
            batch_idexseq = [bat % self.size_train for bat in range((epoch - 1) * self.trainpara.sto_batchsize, epoch * self.trainpara.sto_batchsize)]
            batch_X = self.X_train[batch_idexseq,:]
            batch_y = self.y_train[batch_idexseq,:]
            predicty_z = self.target_model.predict_y(X, batch_X, self.y_train_mean, self.y_train_std)
            loss = ((predicty_z.mean(0) - (batch_y * self.y_train_std + self.y_train_mean))**2).mean()

            optimizer_pre.zero_grad()
            loss.backward()
            optimizer_pre.step()

        '''initialize the fnet'''
        optimizer_pre = torch.optim.Adam(self.fnet.parameters(), lr = 1e-3, betas=(.9, .99))
        psi_pre = 1 if self.trainpara.TransTrick else 0
        batch_idexseq = [bat % self.size_train for bat in range((1 - 1) * self.trainpara.sto_batchsize, 1 * self.trainpara.sto_batchsize)]
        for epoch in range(1, self.trainpara.fnet_ini_num+1):
            batch_X = self.X_train[batch_idexseq,:]
            batch_y = self.y_train[batch_idexseq,:]
            Z = torch.randn([self.trainpara.batchsize, self.trainpara.z_dim]).to(self.device) * self.trainpara.sigma_ini
            X, neg_score_implicit = self.SemiVInet(Z)
            f_opt = self.fnet(X.detach())
            compu_targetscore = self.target_model.score(X.detach(), batch_X, batch_y, self.scale_sto)
            g_opt = f_opt + compu_targetscore * psi_pre
            loss = - torch.mean(torch.sum(g_opt * (2.0 * compu_targetscore + 2.0 * neg_score_implicit.detach() - g_opt), -1))
            optimizer_pre.zero_grad()
            loss.backward()
            optimizer_pre.step()
            if epoch%200 == 0:
                print("loss: {:.4f}".format(loss.item()))
        
        for epoch in tqdm(range(1, self.trainpara.num_epochs+1)):
            
            psi = psi/param_psi if self.trainpara.TransTrick else 0
            for i in range(1, self.trainpara.num_perepoch+1):
                self.iter_idx = (epoch-1) * self.trainpara.num_perepoch + i
                batch_idexseq = [bat % self.size_train for bat in range((self.iter_idx - 1) * self.trainpara.sto_batchsize, self.iter_idx * self.trainpara.sto_batchsize)]
                batch_X = self.X_train[batch_idexseq,:]
                batch_y = self.y_train[batch_idexseq,:]
                # ============================================================== #
                #                      Train the SemiVInet                       #
                # ============================================================== #
                Z = torch.randn([self.trainpara.batchsize, self.trainpara.z_dim]).to(self.device)
                '''change the scale of z'''
                Z = Z * self.trainpara.sigma_ini
                X, neg_score_implicit = self.SemiVInet(Z)
                f_opt = self.fnet(X)
                compu_targetscore = self.target_model.score(X, batch_X, batch_y, self.scale_sto) * annealing_coef(self.iter_idx)
                g_opt = f_opt + compu_targetscore * psi
                loss = torch.mean(torch.sum(g_opt * (2.0 * compu_targetscore + 2.0 * neg_score_implicit - g_opt), -1))

                optimizer_VI.zero_grad()
                loss.backward()
                optimizer_VI.step()
                scheduler_VI.step()
                
                if epoch%self.config.sampling.visual_time == 0 and i % self.trainpara.train_vis_inepoch == 0:
                    print(("Epoch [{}/{}], min score matching[{}/{}], loss: {:.4f}").format(epoch, self.trainpara.num_epochs, i, self.trainpara.num_perepoch, loss.item()))
                # ============================================================ #
                #                        Train the fnet                        #
                # ============================================================ #
                for _ in range(self.trainpara.ftimes):
                    Z = torch.randn([self.trainpara.batchsize, self.trainpara.z_dim]).to(self.device)
                    '''change the scale of z'''
                    Z = Z * self.trainpara.sigma_ini
                    X, neg_score_implicit = self.SemiVInet(Z)

                    f_opt = self.fnet(X.detach())
                    compu_targetscore = self.target_model.score(X.detach(), batch_X, batch_y, self.scale_sto) * annealing_coef(self.iter_idx)
                    g_opt = f_opt + compu_targetscore * psi
                    loss = - torch.mean(torch.sum(g_opt * (2.0 * compu_targetscore + 2.0 * neg_score_implicit.detach() - g_opt), -1))
                    optimizer_f.zero_grad()
                    loss.backward()
                    optimizer_f.step()
                    scheduler_f.step()
                    if epoch%self.config.sampling.visual_time == 0 and i % self.trainpara.train_vis_inepoch == 0:
                        print(("Epoch [{}/{}], min score matching[{}/{}], loss: {:.4f}").format(epoch, self.trainpara.num_epochs, i, self.trainpara.num_perepoch, loss.item()))
            if epoch%self.config.sampling.visual_time ==0 or epoch == self.trainpara.num_epochs:
                X = self.SemiVInet.sampling(num = self.config.sampling.num, sigma=self.trainpara.sigma_ini)
                with torch.no_grad():
                    test_rmse, test_loglik = self.target_model.rmse_llk(X.to(self.device), self.X_test, self.y_test, self.y_train_mean, self.y_train_std)
                print(("######### Epoch [{}/{}], loss: {:.4f}, test_loglik {:.4f}, rmse {:.4f}").format(epoch, self.trainpara.num_epochs, -loss, test_loglik, test_rmse))
        ## Model selection
        if model_selection:
            X = self.SemiVInet.sampling(num = self.config.sampling.num, sigma=self.trainpara.sigma_ini)
            self.target_model.model_selection(X, self.X_dev, self.y_dev, self.y_train_mean, self.y_train_std)
            test_rmse, test_loglik = self.target_model.rmse_llk(X, self.X_test, self.y_test, self.y_train_mean, self.y_train_std)
            print(("######### After selection, test_loglik {:.4f}, rmse {:.4f}").format(test_loglik, test_rmse))
        torch.save(self.SemiVInet.state_dict(), "exp/{}/model{}/SemiVInet_permu_{}.ckpt".format(self.target, self.datetimelabel, self.permu_i))
        torch.save(self.fnet.state_dict(), "exp/{}/model{}/fnet_permu_{}.ckpt".format(self.target, self.datetimelabel, self.permu_i))


if __name__ == "__main__":
    seednow = 2022
    torch.manual_seed(seednow)
    torch.cuda.manual_seed_all(seednow)
    np.random.seed(seednow)
    torch.backends.cudnn.deterministic = True

    config = parse_config()
    task = SIVISM("",config=config)
    task.learn(model_selection = False)

    # for i in range(4):
    #     config = parse_config()
    #     task = SIVISM("",config=config)
    #     task.learn(model_selection = False)
    


