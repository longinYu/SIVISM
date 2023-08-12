import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


class Banana_shape(object):
    name = "banana_shape"
    def __init__(self, device):
        self.device = device
    def logp(self, X):
        Y = torch.stack((X[:, 0], X[:, 0]**2 + X[:, 1] + 1), 1)
        sigmasqinv = torch.tensor([[1.0, -0.9], [-0.9, 1.0]]).to(self.device)/0.19
        return -0.5 * 2 * np.log(2 * np.pi) - 0.5 * np.log(0.19) - 0.5 * torch.matmul(torch.matmul(Y[:,None,:],sigmasqinv), Y[:,:,None]).squeeze(-1)

    def score(self, X):
        Y = torch.matmul(torch.stack((X[:, 0], X[:, 0]**2 + X[:, 1] + 1), 1),torch.tensor([[1.,-0.9],[-0.9,1.]]).to(self.device))
        return -torch.stack((Y[:,0] + 2 * X[:,0] * Y[:,1], Y[:, 1]),1)/0.19

    def contour_plot(self, bbox, ax, fnet, ngrid=100, samples=None, save_to_path=None, quiver = True, t = None):
        xx, yy = np.mgrid[bbox[0]:bbox[1]:100j, bbox[2]:bbox[3]:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        f = -np.log(-np.reshape(self.logp(torch.Tensor(positions.T).to(self.device)).cpu().numpy(), xx.shape))
        if samples is None:
            samples = self.sample(10000).cpu().numpy()
        
        cxx, cyy = np.mgrid[bbox[0]:bbox[1]:30j, bbox[2]:bbox[3]:30j]
        
        ax.axis(bbox)
        ax.set_aspect(abs(bbox[1]-bbox[0])/abs(bbox[3]-bbox[2]))
        cfset = ax.contourf(xx, yy, f, cmap='Blues', alpha=0.8, levels = 11)
        ax.plot(samples[:, 0], samples[:,1], '.', markersize= 2, color='#ff7f0e')
        if quiver:
            cpositions = np.vstack([cxx.ravel(), cyy.ravel()])
            scores = np.reshape(fnet(torch.Tensor(cpositions.T).to(self.device)).detach().cpu().numpy(), cpositions.T.shape)
            ax.quiver(cxx, cyy, scores[:, 0], scores[:, 1], width=0.002)
        plt.xticks(fontsize = 15)
        plt.yticks(fontsize = 15)
        if t:
            ax.set_title("t = {}".format(t), fontsize = 30, y=1.04)
        else:
            ax.set_title("X-shaped", fontsize = 20, y=1.04)
        if save_to_path is not None:
            plt.savefig(save_to_path, bbox_inches='tight')

class X_shaped(object):
    name = "x_shaped"
    def __init__(self, device):
        self.device = device
    def logp(self, X):
        sigmasqinv_0 = torch.tensor([[2., -1.8], [-1.8, 2.]]).to(self.device) / 0.76
        sigmasqinv_1 = torch.tensor([[2., 1.8], [1.8, 2.]]).to(self.device) / 0.76
        return -0.5 * 2 * np.log(2 * np.pi) - 0.5 * np.log(0.76 * 4) + torch.logsumexp(torch.stack(
            (-1/2 * torch.matmul(torch.matmul(X[:,None,:],sigmasqinv_0), X[:,:,None]).squeeze(-1),
            -1/2 * torch.matmul(torch.matmul(X[:,None,:],sigmasqinv_1), X[:,:,None]).squeeze(-1)),1
            ), dim = 1)
    def score(self, X):
        sigmasqinv_0 = torch.tensor([[2., -1.8], [-1.8, 2.]]).to(self.device) / 0.76
        sigmasqinv_1 = torch.tensor([[2., 1.8], [1.8, 2.]]).to(self.device) / 0.76

        Y = F.softmax(torch.stack(
            (-1/2 * torch.matmul(torch.matmul(X[:,None,:],sigmasqinv_0), X[:,:,None]).squeeze(-1),
            -1/2 * torch.matmul(torch.matmul(X[:,None,:],sigmasqinv_1), X[:,:,None]).squeeze(-1)),1
            ), dim = 1)
    
        return -Y[:,0] * torch.matmul(sigmasqinv_0, X[:,:,None]).squeeze(-1) - Y[:,1] * torch.matmul(sigmasqinv_1, X[:,:,None]).squeeze(-1)
    def contour_plot(self, bbox, ax, fnet, ngrid=100, samples=None, save_to_path=None, quiver = True, t = None):
        xx, yy = np.mgrid[bbox[0]:bbox[1]:100j, bbox[2]:bbox[3]:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        f = -np.log(-np.reshape(self.logp(torch.Tensor(positions.T).to(self.device)).cpu().numpy(), xx.shape))
        if samples is None:
            samples = self.sample(10000).cpu().numpy()
        
        cxx, cyy = np.mgrid[bbox[0]:bbox[1]:30j, bbox[2]:bbox[3]:30j]
        
        ax.axis(bbox)
        ax.set_aspect(abs(bbox[1]-bbox[0])/abs(bbox[3]-bbox[2]))
        cfset = ax.contourf(xx, yy, f, cmap='Blues', alpha=0.8, levels = 11)
        ax.plot(samples[:, 0], samples[:,1], '.', markersize= 2, color='#ff7f0e')
        if quiver:
            cpositions = np.vstack([cxx.ravel(), cyy.ravel()])
            scores = np.reshape(fnet(torch.Tensor(cpositions.T).to(self.device)).detach().cpu().numpy(), cpositions.T.shape)
            ax.quiver(cxx, cyy, scores[:, 0], scores[:, 1], width=0.002)

        plt.xticks(fontsize = 15)
        plt.yticks(fontsize = 15)
        if t:
            ax.set_title("t = {}".format(t), fontsize = 30, y=1.04)
        else:
            ax.set_title("X-shaped", fontsize = 20, y=1.04)
        if save_to_path is not None:
            plt.savefig(save_to_path, bbox_inches='tight')


class Multimodal(object):
    name = "multimodal"
    def __init__(self, device):
        self.device = device
    def logp(self, X):
        
        means = torch.tensor([[2.0,0.0],[-2.0,0.0]]).to(self.device)
        return -0.5 * 2 * np.log(2 * np.pi) - np.log(2.0) + torch.logsumexp(
            -torch.sum((X.unsqueeze(1) - means.unsqueeze(0))**2, dim=-1)/2./1**2
            , dim = 1)
    def score(self, X):
        Y = F.softmax(torch.stack(
            (-1/2 * ((X[:, 0] + 2)**2 + X[:, 1]**2),
            -1/2 * ((X[:, 0] - 2)**2 + X[:, 1]**2)),1
        ),dim=1)
        return - torch.stack((Y[:,0] * (X[:, 0] + 2) + Y[:,1] * (X[:, 0] - 2), X[:, 1]),1)
    def contour_plot(self, bbox, ax, fnet, ngrid=100, samples=None, save_to_path=None, quiver=True, t = 0):
        xx, yy = np.mgrid[bbox[0]:bbox[1]:100j, bbox[2]:bbox[3]:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        f = -np.log(-np.reshape(self.logp(torch.Tensor(positions.T).to(self.device)).cpu().numpy(), xx.shape))
        if samples is None:
            samples = self.sample(10000).cpu().numpy()
        
        cxx, cyy = np.mgrid[bbox[0]:bbox[1]:30j, bbox[2]:bbox[3]:30j]
        ax.axis(bbox)
        ax.set_aspect(abs(bbox[1]-bbox[0])/abs(bbox[3]-bbox[2]))
        cfset = ax.contourf(xx, yy, f, cmap='Blues', alpha=0.8, levels = 11)
        ax.plot(samples[:, 0], samples[:,1], '.', markersize= 2, color='#ff7f0e')
        if quiver:
            cpositions = np.vstack([cxx.ravel(), cyy.ravel()])
            scores = np.reshape(fnet(torch.Tensor(cpositions.T).to(self.device)).detach().cpu().numpy(), cpositions.T.shape)
            ax.quiver(cxx, cyy, scores[:, 0], scores[:, 1], width=0.002)
        plt.xticks(fontsize = 15)
        plt.yticks(fontsize = 15)
        if t:
            ax.set_title("t = {}".format(t), fontsize = 30, y=1.04)
        else:
            ax.set_title("Multimodal", fontsize = 20, y=1.04)
        if save_to_path is not None:
            plt.savefig(save_to_path, bbox_inches='tight')
        

class Mnist(object):
    name = "mnist"
    def __init__(self, device):
        self.device = device
    def logp(self, Z, batchdataset, batchlabel):
        """
        output: the \log \E p(Y|X,z)
        Z: the target inference parameters, shape = [T, K*(x_dim + 1)]
        batchdataset: the batch bataset with shape = [batchsize, 785]
        batchlabel: onehot for the the label corresponding to the batchdataset, shape = [n,10]
        scale_sto: num_datasets/batchsize
        """
        K = 10       # the num of classes
        dim_D = 784
        T = Z.shape[0]
        logpx_z = torch.zeros([T])
        for t in range(T):
            logpx_z[t] = (F.log_softmax(torch.matmul(batchdataset, Z[t].reshape(K, dim_D+1).t()), dim = 1) * batchlabel).sum()
        return (torch.logsumexp(logpx_z, 0) - np.log(T))/batchdataset.shape[0] #, logpx_z.sum()/(T * batchdataset.shape[0])
    
    def logpz_neg(self, Z, batchdataset, batchlabel,scale_sto = 3):
        K = 10
        dim_D = 784
        return 1/2 * (Z * Z).sum(1) - (F.log_softmax(torch.matmul(batchdataset, torch.transpose(Z.reshape(-1, K, dim_D+1), 1, 2)), dim = 2) * batchlabel).sum((1,2)) * scale_sto

    def score(self, Z, batchdataset, batchlabel, scale_sto = 30):
        """
        Z: the target inference parameters, shape = [batch, K*(x_dim + 1)]
        batchdataset: the batch bataset with shape = [batchsize, 785]
        batchlabel: the label corresponding to the batchdataset, shape = [n,]
        scale_sto: num_datasets/batchsize
        """
        batch = Z.shape[0]
        K = 10       # the num of classes
        dim_D = 784
        return -Z +  scale_sto * torch.matmul(
            (batchlabel.t() - F.softmax(
                torch.matmul(Z.reshape(-1, K, dim_D+1), batchdataset.t())
                , dim = 1)),
            batchdataset
            ).reshape(batch, -1)



class Hapt(object):
    name = "hapt"
    def __init__(self, device):
        self.device = device
    def logp(self, Z, batchdataset, batchlabel):
        """
        output: the \log \E p(Y|X,z)
        Z: the target inference parameters, shape = [T, K*(x_dim + 1)]
        batchdataset: the batch bataset with shape = [batchsize, 562]
        batchlabel: onehot for the the label corresponding to the batchdataset, shape = [n, 12]
        scale_sto: num_datasets/batchsize
        """
        
        K = 12       # the num of classes
        dim_D = 561
        T = Z.shape[0]
        logpx_z = torch.zeros([T])
        for t in range(T):
            logpx_z[t] = (F.log_softmax(torch.matmul(batchdataset, Z[t].reshape(K, dim_D+1).t()), dim = 1) * batchlabel).sum()
        return (torch.logsumexp(logpx_z, 0) - np.log(T))/batchdataset.shape[0] #, logpx_z.sum()/(T * batchdataset.shape[0])
    
    def logpz_neg(self, Z, batchdataset, batchlabel,scale_sto = 3):
        K = 10
        dim_D = 784
        T = Z.shape[0]
        return 1/2 * (Z * Z).sum(1) - (F.log_softmax(torch.matmul(batchdataset, torch.transpose(Z.reshape(-1, K, dim_D+1), 1, 2)), dim = 2) * batchlabel).sum((1,2)) * scale_sto

    def score(self, Z, batchdataset, batchlabel, scale_sto = 9):
        """
        Z: the target inference parameters, shape = [batch, K*(x_dim + 1)]
        batchdataset: the batch bataset with shape = [batchsize, 785]
        batchlabel: the label corresponding to the batchdataset, shape = [n,]
        scale_sto: num_datasets/batchsize
        """
        batch = Z.shape[0]
        K = 12       # the num of classes
        dim_D = 561
        return -Z +  scale_sto * torch.matmul(
            (batchlabel.t() - F.softmax(
                torch.matmul(Z.reshape(-1, K, dim_D+1), batchdataset.t())
                , dim = 1)),
            batchdataset
            ).reshape(batch, -1)


class LRwaveform(object):
    name = "LRwaveform"
    def __init__(self, device, alpha = 0.01):
        self.device = device
        self.alpha = alpha

    def logp(self, Z, batchdataset, batchlabel, scale_sto = 1):
        """
        output: the \E_{Y|X}\log \E p(Y|X,z), as the test log ll
        Z: the target inference parameters, shape = [T, (x_dim + 1)]
        batchdataset: the batch bataset with shape = [batchsize, (x_dim + 1)]
        batchlabel: onehot for the the label corresponding to the batchdataset, shape = [n,1]
        scale_sto: num_datasets/batchsize
        """
        B = Z.shape[0]
        W = Z
        inner_prod = torch.mm(batchdataset, W.t())
        logpy_xz = (batchlabel.reshape(-1,1) * inner_prod + F.logsigmoid(-inner_prod))
        return (torch.logsumexp(logpy_xz, dim=1).mean(0) - np.log(B))

    def score(self, Z, batchdataset, batchlabel, scale_sto):
        """
        INPUT:
        Z: the target inference parameters, shape = [batch, (x_dim + 1)]
        batchdataset: the batch bataset with shape = [batchsize, (x_dim + 1)]
        batchlabel: the label corresponding to the batchdataset, shape = [n,1]
        scale_sto: num_datasets/batchsize
        OUTPUT:
        -Z + \nabla_Z\log p(Y|X,Z), where p(y_i = 1|x_i,w) = sigmoid(Z\cdot x_i)
        """
        # batchlabel[batchlabel == -1] = 0
        W = Z
        YX = torch.mm(batchlabel.reshape(-1,1).t(), batchdataset)
        inner_prod = torch.mm(batchdataset, W.t())
        score_W = -W * self.alpha + (YX - torch.sum(torch.sigmoid(inner_prod).unsqueeze(2) * batchdataset.unsqueeze(1), dim=0)) * scale_sto
        return score_W


class Bnn(object):
    name = "Bnn"
    def __init__(self, device, d, n_hidden = 50, loglambda = 0, loggamma = 0):
        self.device = device
        self.n_hidden = n_hidden
        self.d = d
        self.dim_vars = (self.d + 1) * self.n_hidden + (self.n_hidden + 1) + 2
        self.dim_wb = self.dim_vars - 2
        self.loggamma = loggamma
        self.loglambda = loglambda
    def logp(self, Z, batchdataset, batchlabel, scale_sto = 1, max_param = 50.0):
        """
        return the log posterior distribution \log P(W|Y,X).
        """
        log_gamma = self.loggamma * torch.ones(Z.size(0)).to(self.device)
        log_lambda = self.loglambda * torch.ones(Z.size(0)).to(self.device)
        gamma_ = torch.exp(log_gamma).clamp(max=max_param)
        lambda_ = torch.exp(log_lambda).clamp(max=max_param)
        W1 = Z[:, :(self.d) * self.n_hidden].reshape(-1, self.d, self.n_hidden) # [B, d, hidden]
        b1 = Z[:, (self.d) * self.n_hidden:(self.d+1) * self.n_hidden].reshape(-1, self.n_hidden) # [B, hidden]
        W2 = Z[:, (self.d+1) * self.n_hidden:(self.d+1) * self.n_hidden+self.n_hidden][:,:,None] # [B, hidden, 1]
        b2 = Z[:,-1].reshape(-1,1) # [B, 1]
        dnn_predict = (torch.matmul(torch.max(torch.matmul(batchdataset, W1) + b1[:,None,:], torch.tensor([0.0]).to(self.device)), W2) + b2[:,None,:])   # [B, n, 1]
        log_lik_data = -0.5 * batchdataset.shape[0] * (np.log(2*np.pi) - log_gamma) - (gamma_/2) * torch.sum(((dnn_predict-batchlabel).squeeze(2))**2, 1)
        log_prior_w = -0.5 * self.dim_wb * (np.log(2*np.pi) - log_lambda) - (lambda_/2)*((W1**2).sum((1,2)) + (W2**2).sum((1,2)) + (b1**2).sum(1) + (b2**2).sum(1))
        return (log_lik_data * scale_sto + log_prior_w)
        
    def score(self, Z, batchdataset, batchlabel, scale_sto = 1, max_param = 50.0):
        """
        return the score function of posterior distribution \nabla \log P(W|Y,X).
        """
        batch_Z = Z.shape[0]
        num_data = batchdataset.shape[0]
        log_gamma = self.loggamma * torch.ones((batch_Z,1)).to(self.device) # [B, 1]
        log_lambda = self.loglambda * torch.ones((batch_Z,1)).to(self.device)
        gamma_ = torch.exp(log_gamma).clamp(max=max_param)
        lambda_ = torch.exp(log_lambda).clamp(max=max_param)
        W1 = Z[:, :(self.d) * self.n_hidden].reshape(-1, self.d, self.n_hidden) # [B, d, hidden]
        b1 = Z[:, (self.d) * self.n_hidden:(self.d+1) * self.n_hidden].reshape(-1, self.n_hidden) # [B, hidden]
        W2 = Z[:, (self.d+1) * self.n_hidden:(self.d+1) * self.n_hidden+self.n_hidden][:,:,None] # [B, hidden, 1]
        b2 = Z[:,-1].reshape(-1,1) # [B, 1]

        dnn_onelinear = torch.matmul(batchdataset, W1) + b1[:,None,:]
        dnn_relu_onelinear = torch.max(dnn_onelinear, torch.tensor([0.0]).to(self.device))
        dnn_grad_relu = (torch.sign(dnn_onelinear) + 1)/2 # shape = [B, n, hidden]
        dnn_predict = (torch.matmul(dnn_relu_onelinear, W2) + b2[:,None,:]) # shape = [B,n,1]
        nabla_predict_b1 = dnn_grad_relu * W2.transpose(1,2) # [B, n, hidden]
        nabla_predict_W1 = nabla_predict_b1[:,:,None,:] * batchdataset[None,:,:,None] # [B,n,d, hidden] 
        nabla_predict_W2 = dnn_relu_onelinear # [B,n, hidden]
        nabla_predict_b2 = torch.ones_like(dnn_predict).to(self.device) # [B,n,1]

        nabla_predict_wb = torch.cat((nabla_predict_W1.reshape(batch_Z, num_data, -1), nabla_predict_b1, nabla_predict_W2, nabla_predict_b2),dim=2)
        nabla_wb = scale_sto * gamma_ * ((batchlabel - dnn_predict) * nabla_predict_wb).sum(1) - lambda_ * Z
        return nabla_wb      # shape = [B, self.dim_vars]
    def rmse_llk(self, Z, batchdataset, batchlabel, mean_y_train, std_y_train, max_param = 50.0):
        """
        return the test RMSE and test log-likelihood of posterior distribution \nabla \log P(W|Y,X).
        """
        log_gamma = self.loggamma * torch.ones((Z.size(0),1)).to(self.device) # [B, 1]
        gamma_ = torch.exp(log_gamma).clamp(max=max_param)
        W1 = Z[:, :(self.d) * self.n_hidden].reshape(-1, self.d, self.n_hidden) # [B, d, hidden]
        b1 = Z[:, (self.d) * self.n_hidden:(self.d+1) * self.n_hidden].reshape(-1, self.n_hidden) # [B, hidden]
        W2 = Z[:, (self.d+1) * self.n_hidden:(self.d+1) * self.n_hidden+self.n_hidden][:,:,None] # [B, hidden, 1]
        b2 = Z[:,-1].reshape(-1,1) # [B, 1]
        dnn_predict = (torch.matmul(torch.max(torch.matmul(batchdataset, W1) + b1[:,None,:], torch.tensor([0.0]).to(self.device)), W2) + b2[:,None,:])
        dnn_predict_true = dnn_predict * std_y_train + mean_y_train # [B, n, 1]
        predict_mean = dnn_predict_true.mean(0)
        test_rmse = (((predict_mean - batchlabel)**2).mean())**(0.5)
        logpy_xz = -0.5 * (np.log(2*np.pi) - log_gamma[:,None,:]) - 0.5 * gamma_[:, None, :] * (dnn_predict_true - batchlabel[None, :, :])**2
        test_llk = (torch.logsumexp(logpy_xz.squeeze(2), dim=0).mean() - np.log(Z.shape[0]))
        return test_rmse.item(), test_llk.item()
    def predict_y(self, Z, batchdataset, mean_y_train, std_y_train, max_param = 50.0):
        """
        return the predicted response variable \hat{y} given the independent variables.
        """
        W1 = Z[:, :(self.d) * self.n_hidden].reshape(-1, self.d, self.n_hidden) # [B, d, hidden]
        b1 = Z[:, (self.d) * self.n_hidden:(self.d+1) * self.n_hidden].reshape(-1, self.n_hidden) # [B, hidden]
        W2 = Z[:, (self.d+1) * self.n_hidden:(self.d+1) * self.n_hidden+self.n_hidden][:,:,None] # [B, hidden, 1]
        b2 = Z[:,-1].reshape(-1,1) # [B, 1]
        dnn_predict = (torch.matmul(torch.max(torch.matmul(batchdataset, W1) + b1[:,None,:], torch.tensor([0.0]).to(self.device)), W2) + b2[:,None,:])
        dnn_predict_true = dnn_predict * std_y_train + mean_y_train
        return dnn_predict_true
    def model_selection(self, Z, batchdataset, batchlabel, mean_y_train, std_y_train, max_param = 50.0):
        """
        Adjust the heuristic loggamma if needed.
        """
        W1 = Z[:, :(self.d) * self.n_hidden].reshape(-1, self.d, self.n_hidden) # [B, d, hidden]
        b1 = Z[:, (self.d) * self.n_hidden:(self.d+1) * self.n_hidden].reshape(-1, self.n_hidden) # [B, hidden]
        W2 = Z[:, (self.d+1) * self.n_hidden:(self.d+1) * self.n_hidden+self.n_hidden][:,:,None] # [B, hidden, 1]
        b2 = Z[:,-1].reshape(-1,1) # [B, 1]
        dnn_predict = (torch.matmul(torch.max(torch.matmul(batchdataset, W1) + b1[:,None,:], torch.tensor([0.0]).to(self.device)), W2) + b2[:,None,:])
        dnn_predict_true = dnn_predict * std_y_train + mean_y_train # [B, n, 1]
        log_gamma_heu = -torch.log(((dnn_predict_true - batchlabel[None, :, :])**2).mean(1))
        self.loggamma = log_gamma_heu


target_distribution = {
    "banana":Banana_shape,
    "multimodal":Multimodal,
    "x_shaped":X_shaped,
    "mnist":Mnist,
    "hapt":Hapt,
    "LRwaveform": LRwaveform,
    "Bnn_boston": Bnn
}