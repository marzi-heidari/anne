import os
import numpy as np
import torch
from math import inf
from scipy import stats
import torch.nn.functional as F
import torch.nn as nn

def get_instance_noisy_label(n, dataset, labels, num_classes, feature_size, norm_std, seed): 
    # n -> noise_rate 
    # dataset -> mnist, cifar10 # not train_loader
    # labels -> labels (targets)
    # label_num -> class number
    # feature_size -> the size of input images (e.g. 28*28)
    # norm_std -> default 0.1
    # seed -> random_seed 
    print("building dataset...")
    label_num = num_classes
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed(int(seed))

    P = []
    flip_distribution = stats.truncnorm((0 - n) / norm_std, (1 - n) / norm_std, loc=n, scale=norm_std)
    flip_rate = flip_distribution.rvs(labels.shape[0])

    if isinstance(labels, list):
        labels = torch.FloatTensor(labels)
    labels = labels.cuda()

    W = np.random.randn(label_num, feature_size, label_num)


    W = torch.FloatTensor(W).cuda()
    for i, (x, y) in enumerate(dataset):
        # 1*m *  m*10 = 1*10
        x = x.cuda()
        A = x.contiguous().view(1, -1).mm(W[y]).squeeze(0)
        A[y] = -inf
        A = flip_rate[i] * F.softmax(A, dim=0)
        A[y] += 1 - flip_rate[i]
        P.append(A)
    P = torch.stack(P, 0).cpu().numpy()
    l = [i for i in range(label_num)]
    new_label = [np.random.choice(l, p=P[i]) for i in range(labels.shape[0])]
    record = [[0 for _ in range(label_num)] for i in range(label_num)]

    for a, b in zip(labels, new_label):
        a, b = int(a), int(b)
        record[a][b] += 1


    pidx = np.random.choice(range(P.shape[0]), 1000)
    cnt = 0
    for i in range(1000):
        if labels[pidx[i]] == 0:
            a = P[pidx[i], :]
            cnt += 1
        if cnt >= 10:
            break
    return np.array(new_label)

def norm(T):
    row_abs = torch.abs(T)
    row_sum = torch.sum(row_abs, 1).unsqueeze(1)
    T_norm = row_abs / row_sum
    return T_norm



def fit(X, num_classes, percentage, filter_outlier=False):
    # number of classes
    c = num_classes
    T = np.empty((c, c)) # +1 -> index 
    eta_corr = X
    ind = []
    for i in np.arange(c):
        if not filter_outlier:
            idx_best = np.argmax(eta_corr[:, i])
        else:
            eta_thresh = np.percentile(eta_corr[:, i], percentage,interpolation='higher')
            robust_eta = eta_corr[:, i]
            robust_eta[robust_eta >= eta_thresh] = 0.0
            idx_best = np.argmax(robust_eta)
            ind.append(idx_best)
        for j in np.arange(c):
            T[i, j] = eta_corr[idx_best, j]
            
    return T, ind

def data_split(data, targets, split_percentage, seed=1):
   
    num_samples = int(targets.shape[0])
    np.random.seed(int(seed))
    train_set_index = np.random.choice(num_samples, int(num_samples*split_percentage), replace=False)
    index = np.arange(data.shape[0])
    val_set_index = np.delete(index, train_set_index)
    train_set, val_set = data[train_set_index, :], data[val_set_index, :]
    train_labels, val_labels = targets[train_set_index], targets[val_set_index]

    return train_set, val_set, train_labels, val_labels


def transform_target(label):
    label = np.array(label)
    target = torch.from_numpy(label).long()
    return target    

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal(m.weight, mode='fan_out')
            
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=1e-1)
            
    return net

def otsu(p):
    """ Calculates a threshold to separate into two groups based on OTSU """
    hist = torch.histc(p, 1000, 0, 1)

    def Q(hist, s, mu):
        n1, n2 = hist[:s].sum(), hist[s:].sum()
        if n1 == 0 or n2 == 0:
            return -10000000, None, None, None, None
        mu1 = (hist[0:s] * (torch.arange(0, s).to(hist.device) / 1000)).sum() / n1
        mu2 = (hist[s:1000] * (torch.arange(s, 1000).to(hist.device) / 1000)).sum() / n2
        sigma1 = ((hist[0:s] * ((torch.arange(0, s).to(hist.device) / 1000) - mu1).pow(2)).sum() / n1).clamp(
            0.00001).sqrt()
        sigma2 = ((hist[s:1000] * ((torch.arange(s, 1000).to(hist.device) / 1000) - mu2).pow(2)).sum() / n2).clamp(
            0.00001).sqrt()
        q = (n1 * (mu1 - mu).pow(2) + n2 * (mu2 - mu).pow(2)) / (n1 * sigma1 + n2 * sigma2)
        return q, mu1, mu2, sigma1, sigma2

    q = [0, 0]
    for s in range(2, 998):
        q.append(Q(hist, s, p.mean())[0])
    s = torch.argmax(torch.tensor(q))
    q, mu1, mu2, sigma1, sigma2 = Q(hist, s, p.mean())
    mu2, sigma2, mu1, sigma1, s = mu2.detach().cpu().item(), sigma2.detach().cpu().item(), \
                                  mu1.detach().cpu().item(), sigma1.detach().cpu().item(), s / 1000
    return mu2, sigma2, mu1, sigma1, s

def split_dataset(p):
    # Convert logits to probabilities
    up_mean, up_sigma, lo_mean, lo_sigma, thresh = otsu(p)
    print(f'otsu split performed:\n\tup_mean: {up_mean:0.5f}\tup_sigma: {up_sigma:0.5f}\n\tlo_mean: {lo_mean:0.5f}\tlo_sigma: {lo_sigma:0.5f}\n\tthresh:  {thresh:0.5f}\n')
    # Search if we are in overfitting region
    group_4 = p < lo_mean
    group_3 = ~group_4 * (p < thresh)
    group_1 = p >= up_mean
    group_2 = ~group_1 * (p >= thresh)

    return group_1, group_2, group_3, group_4