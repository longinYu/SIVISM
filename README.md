## Semi-Implicit Variational Inference via Score Matching (SIVI-SM)
This repository provides the codes for the paper [Semi-Implicit Variational Inference via Score Matching](https://openreview.net/forum?id=sd90a2ytrt) by Longlin Yu and Cheng Zhang.

### Requirements
Please use the following commands to install the requirements
```
pip install -r requirements.txt
```
### Dataset
The datasets used in our experiments are placed at `./datasets`. In our experiments, we use the MNIST and UCI datasets. You can download the UCI datasets from `https://archive.ics.uci.edu/`.


### Training and evaluation
#### 2D synthetic tasks
For the toy examples of 2D synthetic tasks, we show the sampling results of variational distribution in the training dynamics. 
```
python sivism_2d.py --config multimodal.yml
```


#### Bayesian Logistic Regression
We provide the result of approximated posterior distributions of Bayesian logistic regression on the *waveform* dataset. We compare the posterior estimates of SIVI-SM with the ground truth formed from a long MCMC run.
```
python sivism_lr.py --config LRwaveform.yml --baseline_sample SGLD_LR/parallel_SGLD_LRwaveform.pt
```

#### Bayesian Multinomial Logistic Regression
For the Bayesian multinomial logistic regression problem, we provide the demo results of SIVI-SM on MNIST dataset. 
```
python sivism_mlr.py --config mnist.yml
```


#### Bayesian Neural Networks
Lastly, we provide the demo result of SIVI-SM on *boston* dataset.

```
python sivism_bnn.py --config boston.yml
```


### References
If you find this code useful for your research, please consider citing
```
@inproceedings{
yu2023semiimplicit,
title={Semi-Implicit Variational Inference via Score Matching},
author={Longlin Yu and Cheng Zhang},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023}
}
```


