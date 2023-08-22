import argparse
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import yaml
from models.networks import Fnet, SIMINet
from models.target_models import target_distribution
from torchvision import transforms
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
            "--config", type=str, default = "mnist.yml", help="Path to the config file"
        )
    args = parser.parse_args()
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
        os.makedirs(os.path.join("exp", self.target, "traceplot{}".format(self.datetimelabel)),exist_ok=True)
        os.makedirs(os.path.join("exp", self.target, "model{}".format(self.datetimelabel)),exist_ok=True)

       
    def loaddata(self):
        # load the datasets
        if self.target == "mnist":
            train_dataset = torchvision.datasets.MNIST(root='datasets', 
                                            train=True, 
                                            transform=transforms.ToTensor(),  
                                            download=True)
            test_dataset = torchvision.datasets.MNIST(root='datasets', 
                                                train=False, 
                                                transform=transforms.ToTensor(),download=False)
            train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                                batch_size=self.trainpara.sto_batchsize, 
                                                shuffle=True)
            self.test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                                batch_size= self.config.sampling.test_batch, 
                                                shuffle=False)
            self.batchlabels = {}
            self.batchdatasets = {}
            self.scale_sto = 30
            for i, (batchdataset, batchlabel) in enumerate(train_loader):
                self.batchlabels[i+1] = F.one_hot(batchlabel, num_classes = 10).to(self.device)
                self.batchdatasets[i+1] = torch.cat((torch.ones([self.trainpara.sto_batchsize, 1]), batchdataset.reshape(-1, 784)), 1).to(self.device)
        elif self.target == "hapt":
            self.X_train = torch.load("datasets/hapt/Train/X_train.pt").float().to(self.device)
            self.size_train = self.X_train.shape[0]
            self.X_test = torch.load("datasets/hapt/Test/X_test.pt").float().to(self.device)
            self.size_test = self.X_test.shape[0]
            self.y_train = torch.load("datasets/hapt/Train/y_train.pt").float().to(self.device)
            self.y_test = torch.load("datasets/hapt/Test/y_test.pt").float().to(self.device)
            self.permutation = np.random.permutation(self.size_train)
            self.scale_sto = self.X_train.shape[0]/self.trainpara.sto_batchsize


    def learn(self):
        self.preprocess()
        self.loaddata()

        self.target_model = target_distribution[self.target](self.device)
        self.SemiVInet = SIMINet(self.trainpara, self.device).to(self.device)
        self.fnet = Fnet(self.trainpara).to(self.device)

        annealing_coef = lambda t: annealing(t, warm_up_interval = self.num_iters//self.trainpara.warm_ratio, anneal = self.trainpara.annealing)

        optimizer_VI = torch.optim.RMSprop(self.SemiVInet.parameters(), lr = self.trainpara.lr_SIMI, alpha=0.9, eps=1, weight_decay=0, momentum=0, centered=False)
        optimizer_f = torch.optim.RMSprop(self.fnet.parameters(), lr = self.trainpara.lr_f, alpha=0.9, eps=1, weight_decay=0, momentum=0, centered=False)
        scheduler_VI = torch.optim.lr_scheduler.StepLR(optimizer_VI, step_size=self.trainpara.gamma_step, gamma=self.trainpara.gamma)
        scheduler_f = torch.optim.lr_scheduler.StepLR(optimizer_f, step_size=self.trainpara.gamma_step, gamma=self.trainpara.gamma)

        fnetnorm_list = []
        loss_list = []
        test_loglik_list = []
        time_iter = []

        if self.target == "hapt":
            param_psi = 1.005 # 1.004 is also acceptable
        else:
            param_psi = 1.004
        psi = param_psi
        for epoch in tqdm(range(1, self.trainpara.num_epochs+1)):
            psi = psi/param_psi if self.trainpara.TransTrick else 0
            for i in range(1, self.trainpara.num_perepoch+1):
                self.iter_idx = (epoch-1) * self.trainpara.num_perepoch + i
                if self.target in ["hapt"]:
                    batch_idexseq = [bat % self.size_train for bat in range((self.iter_idx - 1) * self.trainpara.sto_batchsize, self.iter_idx * self.trainpara.sto_batchsize)]
                    randidx = self.permutation[batch_idexseq]
                    batch_X = self.X_train[randidx,:]
                    batch_y = self.y_train[randidx,:]
                # ============================================================== #
                #                      Train the SemiVInet                       #
                # ============================================================== #
                time_start = time.time()
                Z = torch.randn([self.trainpara.batchsize, self.trainpara.z_dim]).to(self.device)
                X, neg_score_implicit = self.SemiVInet(Z)
                f_opt = self.fnet(X)
                if self.target == "mnist":
                    compu_targetscore = self.target_model.score(X, self.batchdatasets[i], self.batchlabels[i], self.scale_sto) * annealing_coef(self.iter_idx)
                elif self.target == "hapt":
                    compu_targetscore = self.target_model.score(X, batch_X, batch_y, self.scale_sto) * annealing_coef(self.iter_idx)
                g_opt = f_opt + compu_targetscore * psi
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
                    f_opt = self.fnet(X.detach())
                    if self.target == "mnist":
                        compu_targetscore = self.target_model.score(X.detach(), self.batchdatasets[i], self.batchlabels[i], self.scale_sto) * annealing_coef(self.iter_idx)
                    elif self.target == "hapt":
                        compu_targetscore = self.target_model.score(X.detach(), batch_X, batch_y, self.scale_sto) * annealing_coef(self.iter_idx)
                    g_opt = f_opt + compu_targetscore * psi
                    loss = - torch.mean(torch.sum(g_opt * (2.0 * compu_targetscore + 2.0 * neg_score_implicit.detach() - g_opt), -1))
                    optimizer_f.zero_grad()
                    loss.backward()
                    optimizer_f.step()
                    scheduler_f.step()
                time_end = time.time()
                time_iter.append(time_end - time_start)
                
            # compute some object in the trainging
            fnetnorm_list.append(np.array([self.iter_idx, (g_opt*g_opt).sum(1).mean().item()]))
            loss_list.append(np.array([self.iter_idx, -loss.item()]))   
            
            if epoch%self.config.sampling.visual_time ==0:
                X = self.SemiVInet.sampling(num = self.config.sampling.num)
                # calculate the test_loglik
                with torch.no_grad():
                    if self.target == "mnist":
                        for (test_batchdataset, batchlabel) in self.test_loader:
                            batchlabel = F.one_hot(batchlabel, num_classes = 10).to(self.device)
                            test_batchdataset = torch.cat((torch.ones([10000, 1]), test_batchdataset.reshape(-1, 784)), 1).to(self.device)
                            test_loglik = self.target_model.logp(X.to(self.device), test_batchdataset, batchlabel).item()
                    elif self.target in ["hapt"]:
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