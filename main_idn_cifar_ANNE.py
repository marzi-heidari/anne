import argparse

import torch.optim.lr_scheduler
import torchvision.transforms as transforms
from timm.data.mixup import one_hot
from timm.optim.adamp import projection
from torch.optim import SGD
from torch.utils.data import Subset
from tqdm import tqdm

from datasets.dataloader_cifar_idn import cifar_dataset
from models.preresnet import PreResNet18
from sklearn.mixture import GaussianMixture
from utils import *

parser = argparse.ArgumentParser('Train with synthetic cifar noisy dataset')
parser.add_argument('--dataset_path', default='/home/student/data/CIFAR10', help='dataset path')
parser.add_argument('--dataset', default='cifar100', help='dataset name')

# dataset settings
parser.add_argument('--noise_mode', default='instance', type=str, help='artifical noise mode (default: symmetric)')
parser.add_argument('--noise_ratio', default=0.5, type=float, help='artifical noise ratio (default: 0.5)')
parser.add_argument('--open_ratio', default=0.0, type=float, help='artifical noise ratio (default: 0.0)')

# model settings
parser.add_argument('--theta_s', default=1.0, type=float, help='threshold for selecting samples (default: 1)')
parser.add_argument('--gamma_r', default=0.9, type=float, help='threshold for relabelling samples (default: 0.9)')
parser.add_argument('--gamma_e', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--lambda_fc', default=1.0, type=float, help='weight of feature consistency loss (default: 1.0)')

# train settings
parser.add_argument('--model', default='PreResNet18', help=f'model architecture (default: PreResNet18)')
parser.add_argument('--epochs', default=300, type=int, metavar='N', help='number of total epochs to run (default: 300)')
parser.add_argument('--batch_size', default=128, type=int, help='mini-batch size (default: 128)')
parser.add_argument('--lr', default=0.02, type=float, help='initial learning rate (default: 0.02)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum of SGD solver (default: 0.9)')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay (default: 5e-4)')
parser.add_argument('--seed', default=3047, type=int, help='seed for initializing training. (default: 3047)')
parser.add_argument('--gpuid', default='0', type=str, help='Selected GPU (default: "0")')
parser.add_argument('--exp-name', type=str, default='')
parser.add_argument('--warmup', default=0, type=int, metavar='wm', help='number of total warmup')
parser.add_argument('--ceil', default=200, type=int, metavar='ceil', help='ceil knn')
parser.add_argument('--distill_mode', type=str, default='fine-gmm', choices=['kmeans', 'fine-kmeans', 'fine-gmm'],
                    help='mode for distillation kmeans or eigen.')

parser.add_argument('--kmin1', default=40, type=int, metavar='N1', help='kmin1')
parser.add_argument('--kmin2', default=80, type=int, metavar='N2', help='kmin2')


def get_knn_indices(embeddings, k=5):
    """
    embeddings: (N, d) tensor
    k: number of nearest neighbors
    Returns a (N, k) tensor of neighbor indices for each sample.
    """
    # Compute pairwise distances: shape [N, N]
    # cdist can be large for big N, but is simple for demonstration.
    distances = torch.cdist(embeddings, embeddings, p=2)

    # We don't want each sample to be its own neighbor, so set diagonal to +inf
    n = distances.shape[0]
    distances.view(-1)[::n + 1] = float('inf')

    # Sort distances and take indices of top-k
    # sorted_dists, sorted_indices: both shape [N, N]
    _, sorted_indices = torch.sort(distances, dim=1)
    return sorted_indices[:, :k]  # [N, k]


def similarity(u, v, sim_type='cosine'):
    """
    u: (d, ) or (N, d)
    v: (d, ) or (N, d)
    sim_type: 'cosine', 'dot', or 'l2'
        - 'cosine': Cosine similarity
        - 'dot':    Dot-product
        - 'l2':     Negative L2 norm (the smaller the distance, the larger the similarity)
    Returns a 1D tensor of shape (N,) if inputs have a batch dimension.
    """

    # If inputs are 1D, expand to 2D for batch processing.
    if u.dim() == 1:
        u = u.unsqueeze(0)
    if v.dim() == 1:
        v = v.unsqueeze(0)

    if sim_type == 'cosine':
        # Cosine similarity across dim=1
        sim_val = F.cosine_similarity(u, v, dim=1)
    elif sim_type == 'dot':
        # Dot-product similarity across dim=-1
        sim_val = (u * v).sum(dim=-1)
    elif sim_type == 'l2':
        # Negative L2 distance
        # For a "similarity", we negate the Euclidean norm
        dist = torch.norm(u - v, p=2, dim=1)
        sim_val = -dist
    else:
        raise ValueError(f"Unknown sim_type='{sim_type}'. Choose from ['cosine','dot','l2']")

    return sim_val


def compute_attention_weights(emb_i, emb_neighbors, tau=1.0, sim_type='l2'):
    """
    emb_i: (d,) embedding of a single sample
    emb_neighbors: (k, d) embedding of k neighbors
    tau: temperature
    Returns alpha: (k, ) attention weights that sum to 1.
    """
    # Similarities shape: (k,)
    sims = similarity(emb_i, emb_neighbors, sim_type=sim_type)

    # Exponential + softmax
    exp_sims = torch.exp(sims / tau)
    alpha = exp_sims / (exp_sims.sum() + 1e-10)
    return alpha


def aggregate_labels(alpha, neighbor_labels):
    """
    alpha: (k,) attention weights
    neighbor_labels: (k, num_classes) soft/hard labels
    Returns aggregated label: (num_classes,)
    """
    # Weighted sum
    return torch.matmul(alpha.unsqueeze(0), neighbor_labels).squeeze(0)


def update_label_soft(old_label, agg_label, beta=0.5):
    return beta * agg_label + (1 - beta) * old_label


def probability_noisy(old_label, agg_label):
    """
    old_label: (C,) original label distribution
    agg_label: (C,) aggregated label distribution
    """
    # Indices
    old_idx = torch.argmax(old_label, dim=-1)
    agg_idx = torch.argmax(agg_label, dim=-1)

    # 1) Deterministic check: if they differ
    diff_flag = 1.0 if old_idx != agg_idx else 0.0

    val_agg = agg_label[agg_idx]  # top class in aggregated
    val_old = agg_label[old_idx]  # aggregated distribution at old class
    denom = val_old + val_agg
    if denom < 1e-10:
        action_prob = 0.0
    else:
        action_prob = float(val_agg / denom) * diff_flag

    return action_prob


import torch
import torch.nn.functional as F


def compute_exp_neg_kl_reward(reward_model, projection, x, y, mask_noisy=None):
    """
    R_tilde = exp( - E_noisy[ KL( old_label || f_reward(...) ) ] )

    old_labels:   (N, C)
    agg_labels:   (N, C) - or potentially replaced by "reward_model(...)" usage
    reward_model: an nn.Module that can transform x_for_reward for computing reward
    x_for_reward: (N, input_dim) input to the reward model
    mask_noisy:   (N,) boolean mask
    """

    # Get reward model outputs (for example, we might build a distribution from it)
    # Here we illustrate simply returning embeddings => we could form some distribution

    with torch.no_grad():
        emb_reward = projection(reward_model(x))  # shape (N, d)
        neighbor_ids = get_knn_indices(emb_reward)  # (N, k)

        emb_neighbors = emb_reward[neighbor_ids]  # (k, d)
        labels_neighbors = y[neighbor_ids]  # (k, C)

        alpha = compute_attention_weights(emb_reward, emb_neighbors)
        agg_label = aggregate_labels(alpha, labels_neighbors)
        kl = F.kl_div(
            torch.log_softmax(agg_label, dim=-1),  # log Q
            y,  # P
            reduction='batchmean'
        ).sum(dim=-1)  # sum over classes => shape (N,)

        # Exponential negative
        reward = torch.exp(-kl)
    return reward


###############################################################################
# Complete iteration example
###############################################################################
def policy_iteration(
        data_x,  # (N, input_dim)
        data_y,  # (N, C) original label distribution
        policy_model,
        projection,  # used to compute embeddings for neighbor search & alpha
        tau=0.5,
        sim_type='cosine'
):
    """
    data_x, data_y: your dataset and label distributions.
    policy_model:   used for neighbor embeddings & attention.
    reward_model:   used only to compute final reward.
    knn_idx:        (N, k) neighbor indices for each sample.
    sim_type:       similarity type ('cosine', 'dot', 'l2').
    """
    # 1) Embeddings from policy model
    with torch.no_grad():
        emb_policy = projection(policy_model(data_x))  # (N, d)
        N, C = data_y.shape

        neighbor_ids = get_knn_indices(emb_policy)  # (N, k)
        emb_i = emb_policy  # (d,)
        emb_neighbors = emb_policy[neighbor_ids]  # (k, d)
        labels_neighbors = data_y[neighbor_ids]  # (k, C)

        alpha = compute_attention_weights(emb_i, emb_neighbors, tau=tau, sim_type=sim_type)
        agg_label = aggregate_labels(alpha, labels_neighbors)

        p_noisy_vec = probability_noisy(data_y, agg_label)

        random_vals = torch.rand(N, device=data_y.device)
        is_noisy_mask = (random_vals < p_noisy_vec)  # boolean mask

        # Copy old labels
        updated_labels = data_y.clone()

        # For all samples deemed noisy, apply the soft update:
        #    new_label = argmax agg_label
        updated_labels[is_noisy_mask] = F.one_hot(torch.argmax(agg_label, dim=-1), num_classes=C).device(data_y.device)
    return updated_labels, p_noisy_vec


def train(modified_label, all_trainloader, encoder, classifier, proj_head, optimizer,
          epoch, args):
    encoder.train()
    classifier.train()
    proj_head.train()

    xlosses = AverageMeter('xloss')
    ulosses = AverageMeter('uloss')

    all_bar = tqdm(all_trainloader)

    for batch_idx, ([inputs_u1, inputs_u2], _original_labels, index) in enumerate(all_bar):
        labels_x = modified_label[index]
        inputs_u1, inputs_u2 = inputs_u1.cuda(f'cuda:{args.gpuid}'), inputs_u2.cuda(f'cuda:{args.gpuid}')
        # updated_labels, p_noisy_list = policy_iteration(
        #     inputs_u1,  # (N, input_dim)
        #     labels_x,  # (N, C) original label distribution
        #     encoder,  # used to compute embeddings for neighbor search & alpha
        #     proj_head,
        #     tau=0.5,
        #     sim_type='l2')
        # reward = compute_exp_neg_kl_reward(encoder, proj_head, inputs_u1, modified_label[index])
        # batch_size = inputs_u1.size(0)


        all_inputs_x = torch.cat([inputs_u1, inputs_u2], dim=0)
        all_targets_x = torch.cat([_original_labels, _original_labels], dim=0).cuda(f'cuda:{args.gpuid}')
        feats_u1 = encoder(all_inputs_x)

        f, h = proj_head, classifier

        z1 = f(feats_u1)
        p1 = h(z1)
        ce_loss = F.cross_entropy(p1, all_targets_x)

        loss = ce_loss
        xlosses.update(ce_loss.item())
        ulosses.update(ce_loss.item())
        all_bar.set_description(
            f'Train epoch {epoch + 1} LR:{optimizer.param_groups[0]["lr"]} Labeled loss: {xlosses.avg:.4f} Unlabeled loss: {ulosses.avg:.4f}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # modified_label[index]=updated_labels
    return modified_label


def test(testloader, encoder, classifier, epoch):
    encoder.eval()
    classifier.eval()
    accuracy = AverageMeter('accuracy')
    data_bar = tqdm(testloader)
    with torch.no_grad():
        for i, (data, label) in enumerate(data_bar):
            data, label = data.cuda(f'cuda:{args.gpuid}'), label.cuda(f'cuda:{args.gpuid}')
            feat = encoder(data)

            res = classifier(feat)
            pred = torch.argmax(res, dim=1)
            acc = torch.sum(pred == label) / float(data.size(0))
            accuracy.update(acc.item(), data.size(0))
            data_bar.set_description(f'Test epoch {epoch + 1}: Accuracy#{accuracy.avg:.4f}')

    return accuracy.avg


def get_singular_vector(features, labels):
    '''
    To get top1 sigular vector in class-wise manner by using SVD of hidden feature vectors
    features: hidden feature vectors of data (numpy)
    labels: correspoding label list
    '''

    singular_vector_dict = {}
    with tqdm(total=len(np.unique(labels))) as pbar:
        for index in np.unique(labels):
            _, _, v = np.linalg.svd(features[labels == index])
            singular_vector_dict[index] = v[0]
            pbar.update(1)

    return singular_vector_dict


def get_score(singular_vector_dict, features, labels, normalization=True):
    '''
    Calculate the score providing the degree of showing whether the data is clean or not.
    '''
    if normalization:
        scores = [np.abs(np.inner(singular_vector_dict[labels[indx]], feat / np.linalg.norm(feat))) for indx, feat in
                  enumerate(tqdm(features))]
    else:
        scores = [np.abs(np.inner(singular_vector_dict[labels[indx]], feat)) for indx, feat in
                  enumerate(tqdm(features))]

    return np.array(scores)


def fit_mixture(scores, labels, gamma_e=0.5):
    '''
    Assume the distribution of scores: bimodal gaussian mixture model
    
    return clean labels
    that belongs to the clean cluster by fitting the score distribution to GMM
    '''

    clean_labels = []
    indexes = np.array(range(len(scores)))
    probs = {}
    for cls in np.unique(labels):
        cls_index = indexes[labels == cls]
        feats = scores[labels == cls]
        feats_ = np.ravel(feats).astype(np.float).reshape(-1, 1)
        if feats_.shape[0] >= 5:
            gmm = GaussianMixture(n_components=2, covariance_type='full', tol=1e-6, max_iter=10)

            gmm.fit(feats_)
            prob = gmm.predict_proba(feats_)
            prob = prob[:, gmm.means_.argmax()]
            for i in range(len(cls_index)):
                probs[cls_index[i]] = prob[i]

            clean_labels += [cls_index[clean_idx] for clean_idx in range(len(cls_index)) if prob[clean_idx] > gamma_e]
        else:
            clean_labels += [cls_index[clean_idx] for clean_idx in range(len(cls_index))]

    return np.array(clean_labels, dtype=np.int64), probs


def fine(current_features, current_labels, fit='kmeans', prev_features=None, prev_labels=None, gamma_e=0.7):
    '''
    prev_features, prev_labels: data from the previous round
    current_features, current_labels: current round's data
    
    return clean labels
    
    if you insert the prev_features and prev_labels to None,
    the algorthm divides the data based on the current labels and current features
    
    '''
    if (prev_features != None) and (prev_labels != None):
        singular_vector_dict = get_singular_vector(prev_features, prev_labels)
    else:
        singular_vector_dict = get_singular_vector(current_features, current_labels)

    scores = get_score(singular_vector_dict, features=current_features, labels=current_labels)

    if 'kmeans' in fit:
        clean_labels = cleansing(scores, current_labels)
        probs = None
    elif 'gmm' in fit:
        # fit a two-component GMM to the loss
        clean_labels, probs = fit_mixture(scores, current_labels, gamma_e)
    else:
        raise NotImplemented

    return clean_labels, probs, scores


def cleansing(scores, labels):
    '''
    Assume the distribution of scores: bimodal spherical distribution.
    
    return clean labels 
    that belongs to the clean cluster made by the KMeans algorithm
    '''

    indexes = np.array(range(len(scores)))
    clean_labels = []
    for cls in np.unique(labels):
        cls_index = indexes[labels == cls]
        kmeans = cluster.KMeans(n_clusters=2, random_state=0).fit(scores[cls_index].reshape(-1, 1))
        if np.mean(scores[cls_index][kmeans.labels_ == 0]) < np.mean(
                scores[cls_index][kmeans.labels_ == 1]): kmeans.labels_ = 1 - kmeans.labels_

        clean_labels += cls_index[kmeans.labels_ == 0].tolist()

    return np.array(clean_labels, dtype=np.int64)


def extract_cleanidx(features, labels, mode='fine-kmeans', gamma_e=0.6):
    scores = None

    # get teacher_idx
    if 'fine' in mode:
        teacher_idx, probs, scores = fine(current_features=features, current_labels=labels, fit=mode, gamma_e=gamma_e)

    teacher_idx = torch.tensor(teacher_idx)
    return teacher_idx, probs, scores


def main():
    args = parser.parse_args()
    seed_everything(args.seed)

    args.run_path = args.exp_name

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid

    # generate noisy_t dataset with our transformation
    if not os.path.isdir(f'{args.dataset}'):
        os.mkdir(f'{args.dataset}')
    if not os.path.isdir(f'{args.dataset}/{args.run_path}'):
        os.mkdir(f'{args.dataset}/{args.run_path}')

    ############################# Dataset initialization ##############################################
    if args.dataset == 'cifar10':
        args.num_classes = 10
        args.image_size = 32
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    elif args.dataset in ['cifar100', 'red']:
        args.num_classes = 100
        args.image_size = 32
        normalize = transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))
    else:
        raise ValueError(f'args.dataset should be cifar10 or cifar100, rather than {args.dataset}!')

    # data loading
    weak_transform = transforms.Compose([
        transforms.RandomCrop(args.image_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])
    none_transform = transforms.Compose([transforms.ToTensor(), normalize])
    strong_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                           transforms.RandomHorizontalFlip(),
                                           CIFAR10Policy(),
                                           transforms.ToTensor(),
                                           normalize])

    transform_test = transforms.Compose([

        transforms.Resize((32, 32), interpolation=2),
        transforms.ToTensor(),
        transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
    ])

    noise_file = '%s/%.1f_%s.pt' % (args.dataset_path, args.noise_ratio, args.noise_mode)




    test_data = cifar_dataset(dataset=args.dataset, noise_mode=args.noise_mode, r=args.noise_ratio,
                              root_dir=args.dataset_path, transform=transform_test, mode='test')

    all_data = cifar_dataset(dataset=args.dataset, noise_mode=args.noise_mode, r=args.noise_ratio,
                             root_dir=args.dataset_path,
                             transform=MixTransform(strong_transform=strong_transform, weak_transform=weak_transform,
                                                    K=1), mode="all", noise_file=noise_file)



    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                              pin_memory=True)

    all_loader = torch.utils.data.DataLoader(all_data, batch_size=args.batch_size, num_workers=4, shuffle=True,
                                             pin_memory=True, drop_last=True)

    ################################ Model initialization ###########################################
    encoder = PreResNet18(args.num_classes)
    classifier = torch.nn.Linear(128, args.num_classes)
    proj_head = torch.nn.Sequential(torch.nn.Linear(encoder.fc.in_features, 256),
                                    torch.nn.BatchNorm1d(256),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(256, 128))

    encoder.fc = torch.nn.Identity()

    encoder.cuda(f'cuda:{args.gpuid}')
    classifier.cuda(f'cuda:{args.gpuid}')
    proj_head.cuda(f'cuda:{args.gpuid}')

    #################################### Training initialization #######################################
    optimizer = SGD(
        [{'params': encoder.parameters()}, {'params': classifier.parameters()}, {'params': proj_head.parameters()},
         ],
        lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr / 50.0)

    acc_logs = open(f'{args.dataset}/{args.run_path}/acc.txt', 'w')
    stat_logs = open(f'{args.dataset}/{args.run_path}/stat.txt', 'w')
    save_config(args, f'{args.dataset}/{args.run_path}')
    print('Train args: \n', args)
    best_acc = 0
    all_acc = []

    ################################ Training loop ###########################################

    modified_label = all_loader.dataset.noise_label
    for i in range(args.epochs):

        # balanced_sampler

        modified_label = train(modified_label, all_loader, encoder, classifier, proj_head, optimizer, i,
                                 args)

        cur_acc = test(test_loader, encoder, classifier, i)
        all_acc.append(cur_acc)
        scheduler.step()
        if cur_acc > best_acc:
            best_acc = cur_acc
            save_checkpoint({
                'cur_epoch': i,
                'classifier': classifier.state_dict(),
                'encoder': encoder.state_dict(),
                'proj_head': proj_head.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, filename=f'{args.dataset}/{args.run_path}/best_acc.pth.tar')
        acc_logs.write(f'Epoch [{i + 1}/{args.epochs}]: Best accuracy@{best_acc}! Current accuracy@{cur_acc} \n')
        acc_logs.flush()
        print(f'Epoch [{i + 1}/{args.epochs}]: Best accuracy@{best_acc}! Current accuracy@{cur_acc} \n')

    save_checkpoint({
        'cur_epoch': args.epochs,
        'classifier': classifier.state_dict(),
        'encoder': encoder.state_dict(),
        'proj_head': proj_head.state_dict(),

        'optimizer': optimizer.state_dict(),
    }, filename=f'{args.dataset}/{args.run_path}/last.pth.tar')


if __name__ == '__main__':
    main()
