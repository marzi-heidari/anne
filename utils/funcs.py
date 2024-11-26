from __future__ import print_function

import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import ImageFilter
from statistics import mean,stdev 


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def D(p, z, version='simplified'):  # negative cosine similarity
    if version == 'original':
        z = z.detach()  # stop gradient
        p = F.normalize(p, dim=1)  # l2-normalize
        z = F.normalize(z, dim=1)  # l2-normalize
        return -(p * z).sum(dim=1).mean()

    elif version == 'simplified':  # same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception



def aknn_predict_v2(id, feature, feature_bank, feature_labels, classes,  otsu_split=None,ceil=200, step=0.01, kmin1=40, kmin2=80):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    
    pred_score = torch.zeros((feature.size(0), classes), device=feature.device)
    pred_labels = torch.zeros((feature.size(0),), dtype=torch.long, device=feature.device)
    
    
    for id in range(sim_matrix.size(0)):
        if id in otsu_split['clean_ids']:   
            temp_radius = 0.99 
            while True:
                
                mask = sim_matrix[id]>temp_radius
                sim_indices = torch.nonzero(mask)
                
                if len(sim_indices)<5:
                    temp_radius -=step
                else:
                    break
            
        elif id in otsu_split['maybe_clean_ids']:
            temp_radius = 0.99 
            while True:
                mask = sim_matrix[id]>temp_radius
                sim_indices = torch.nonzero(mask)
                if len(sim_indices)<20:
                    
                    temp_radius -=step
                else:
                    break
        elif id in otsu_split['maybe_noisy_ids'] :
            temp_radius = 0.99
            while True:
                
                mask = sim_matrix[id]>temp_radius
                sim_indices = torch.nonzero(mask)
                
                if len(sim_indices)<kmin1:
                    temp_radius -=step
                else:
                    break
        elif id in otsu_split['noisy_ids']:
            temp_radius = 0.99 
            while True:
                
                mask = sim_matrix[id]>temp_radius
                sim_indices = torch.nonzero(mask)
                
                if len(sim_indices)<kmin2:
                    temp_radius -=step
                else:
                    break
        else:
            raise Exception("Invalid id")
                
        sim_label_topk = False 

        
        knn_k = len(sim_indices)
        
        if knn_k > ceil:
            knn_k = ceil
        elif knn_k <5:
            knn_k = 5
        
        sim_weight, sim_indices = sim_matrix[id].topk(k=knn_k, dim=-1)
        sim_label_topk = True
            
        # [B, K]
        if sim_label_topk == True:
            sim_labels = torch.gather(feature_labels.expand(1, -1), dim=-1, index=sim_indices.unsqueeze(0))
        else:
            sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices[:,1].view(1,-1))
        

        sim_weight = torch.ones_like(sim_weight)
        sim_weight = sim_weight / sim_weight.sum(dim=-1, keepdim=True)

        # counts for each class
        one_hot_label = torch.zeros(1 * knn_k, classes, device=sim_labels.device)
        # [B*K, C]
        one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
        # weighted score ---> [B, C]
        
        part_score = torch.sum(one_hot_label.view(1, -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)
        part_pred = part_score.argmax(dim=-1)

        pred_score[id] = part_score
        pred_labels[id] = part_pred

    return pred_score, pred_labels, knn_k



def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_config(args, path):
    dict = vars(args)
    if not os.path.isdir(path):
        os.mkdir(os.path.abspath(path))
    with open(path + '/params.csv', 'w') as f:
        for key in dict.keys():
            f.write("%s\t%s\n" % (key, dict[key]))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class KCropsTransform:
    """Take K random crops of one image as the query and key."""

    def __init__(self, base_transform, K=2):
        self.base_transform = base_transform
        self.K = K

    def __call__(self, x):
        res = [self.base_transform(x) for i in range(self.K)]
        return res


class MixTransform:
    def __init__(self, strong_transform, weak_transform, K=2):
        self.strong_transform = strong_transform
        self.weak_transform = weak_transform
        self.K = K

    def __call__(self, x):
        res = [self.weak_transform(x) for i in range(self.K)] + [self.strong_transform(x) for i in range(self.K)]
        return res


class DMixTransform:
    def __init__(self, transforms, nums):
        self.transforms = transforms
        self.nums = nums

    def __call__(self, x):
        res = []
        for i, trans in enumerate(self.transforms):
            res += [trans(x) for _ in range(self.nums[i])]
        return res


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
    
def otsu(p):
    """ Calculates a threshold to separate into two groups based on OTSU """
    hist = torch.histc(p, 1000, 0, 1)
    import pdb; pdb.set_trace()
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

#def weighted_knn_ball(epoch, cur_feature, feature, label, num_classes, knn_k=100, chunks=10, norm='global', radius=0.99, rule="type1",conf=None, knnweight=False, radaptive=None, otsu_split=None,teto=200):
def weighted_aknn( cur_feature, feature, label, num_classes, norm='global', otsu_split=None,ceil=200):
    # distributed fast KNN and sample selection with three different modes
    num = len(cur_feature)
    num_class = torch.tensor([torch.sum(label == i).item() for i in range(num_classes)]).to(
        feature.device) + 1e-10
    pi = num_class / num_class.sum()
    
    # it will be a sample at a time because the neighborhood change be differt for each sample
    split = torch.tensor(np.linspace(0, num, num+1, dtype=int), dtype=torch.long).to(feature.device)
    score = torch.tensor([]).to(feature.device)
    pred = torch.tensor([], dtype=torch.long).to(feature.device)
    feature = torch.nn.functional.normalize(feature, dim=1)
    knn_min = 100000
    knn_max = -1
    knn_hist = []
    with torch.no_grad():
        
        for i in range(num):
            torch.cuda.empty_cache()
            part_feature = cur_feature[split[i]: split[i + 1]]

            part_score, part_pred, k_value = akk_predict(i, part_feature, feature.T, label, num_classes, otsu_split=otsu_split,ceil=ceil)

            score = torch.cat([score, part_score], dim=0)
            pred = torch.cat([pred, part_pred], dim=0)
            knn_min = min(k_value, knn_min)
            knn_max = max(k_value, knn_max)
            knn_hist.append(k_value)
            
        knn_mean = mean(knn_hist)
        knn_std = stdev(knn_hist)
        # balanced vote
        if norm == 'global':
            # global normalization
            score = score / pi
        else:  # no normalization
            pass
        score = score/score.sum(1, keepdim=True)

    print(f'knn_min: {knn_min} knn_max: {knn_max}')

    return score, knn_min, knn_max, knn_mean, knn_std 



def akk_predict(id, feature, feature_bank, feature_labels, classes, otsu_split=None,ceil=200, floor=5, kmin1=40, kmin2=80, step=0.01):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    
    #not warmup
    if (otsu_split is not None): 
        
        if id in otsu_split['clean_ids']:   
            temp_radius = 0.99 
            while True:
                mask = sim_matrix>temp_radius
                sim_indices = torch.nonzero(mask)
                if len(sim_indices)<5:
                    temp_radius -=step
                else:
                    break
            
        elif id in otsu_split['maybe_clean_ids']:
            temp_radius = 0.99 
            while True:
                mask = sim_matrix>temp_radius
                sim_indices = torch.nonzero(mask)
                if len(sim_indices)<20:
                    temp_radius -=step
                else:
                    break
        elif id in otsu_split['maybe_noisy_ids'] :
            temp_radius = 0.99
            while True:
                mask = sim_matrix>temp_radius
                sim_indices = torch.nonzero(mask)
                if len(sim_indices)<kmin1:
                    temp_radius -=step
                else:
                    break
        elif id in otsu_split['noisy_ids']:
            temp_radius = 0.99 
            while True:
                mask = sim_matrix>temp_radius
                sim_indices = torch.nonzero(mask)
                if len(sim_indices)<kmin2:
                    temp_radius -=step
                else:
                    break
        else:
            raise Exception("Invalid id")
        
        knn_k = len(sim_indices)
    else:
        #warmup
        knn_k = ceil
        
    
    if knn_k > ceil:
        knn_k = ceil
    elif knn_k <floor:
        knn_k = floor
    
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    sim_label_topk = True
        
    # [B, K]
    if sim_label_topk == True:
        sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    else:
        sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices[:,1].view(1,-1))
    
    sim_weight = torch.ones_like(sim_weight)
    sim_weight = sim_weight / sim_weight.sum(dim=-1, keepdim=True)
    
    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)
    pred_labels = pred_scores.argmax(dim=-1)

    return pred_scores, pred_labels, knn_k
