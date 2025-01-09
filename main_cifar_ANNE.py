import argparse

import torch.optim.lr_scheduler
import torchvision.transforms as transforms
from torch.optim import SGD
from torch.utils.data import Subset
from tqdm import tqdm

from datasets.dataloader_cifar import cifar_dataset
from models.preresnet import PreResNet18
from sklearn.mixture import GaussianMixture
from utils import *

parser = argparse.ArgumentParser('Train with synthetic cifar noisy dataset')
parser.add_argument('--dataset_path', default='~/CIFAR/CIFAR10', help='dataset path')
parser.add_argument('--dataset', default='cifar10', help='dataset name')
parser.add_argument('--open_path', default='~/CIFAR100', help='open noise dataset path')
parser.add_argument('--open_name', default='cifar100', help='open noise dataset name')

# dataset settings
parser.add_argument('--noise_mode', default='sym', type=str, help='artifical noise mode (default: symmetric)')
parser.add_argument('--noise_ratio', default=0.5, type=float, help='artifical noise ratio (default: 0.5)')
parser.add_argument('--open_ratio', default=0.0, type=float, help='artifical noise ratio (default: 0.0)')

# model settings
parser.add_argument('--theta_s', default=1.0, type=float, help='threshold for selecting samples (default: 1)')
parser.add_argument('--gamma_r', default=0.9, type=float, help='threshold for relabelling samples (default: 0.9)')
parser.add_argument('--gamma_e', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--lambda_fc', default=1.0, type=float, help='weight of feature consistency loss (default: 1.0)')

# train settings
parser.add_argument('--model', default='PreResNet18', help='model architecture (default: PreResNet18)')
parser.add_argument('--epochs', default=300, type=int, metavar='N', help='number of total epochs to run (default: 300)')
parser.add_argument('--batch_size', default=128, type=int, help='mini-batch size (default: 128)')
parser.add_argument('--lr', default=0.02, type=float, help='initial learning rate (default: 0.02)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum of SGD solver (default: 0.9)')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay (default: 5e-4)')
parser.add_argument('--seed', default=3047, type=int, help='seed for initializing training. (default: 3047)')
parser.add_argument('--gpuid', default='0', type=str, help='Selected GPU (default: "0")')
parser.add_argument('--exp-name', type=str, default='')
parser.add_argument('--warmup', default=0, type=int, metavar='wm', help='number of total warmup')
parser.add_argument('--ceil', default=200, type=int, metavar='ceil', help= 'knn max valuee')
parser.add_argument('--distill_mode', type=str, default='fine-gmm', choices=['kmeans','fine-kmeans','fine-gmm'], help='mode for distillation kmeans or eigen.')

parser.add_argument('--kmin1', default=40, type=int, metavar='N1', help='kmin1')
parser.add_argument('--kmin2', default=80, type=int, metavar='N2', help='kmin2')

def train(labeled_trainloader, modified_label, all_trainloader, encoder, classifier, proj_head, pred_head, optimizer, epoch, args):
    encoder.train()
    classifier.train()
    proj_head.train()
    pred_head.train()
    xlosses = AverageMeter('xloss')
    ulosses = AverageMeter('uloss')
    labeled_train_iter = iter(labeled_trainloader)
    all_bar = tqdm(all_trainloader)
    for batch_idx, ([inputs_u1, inputs_u2], _, _, _) in enumerate(all_bar):
        try:
            
            [inputs_x1, inputs_x2], labels_x, _, index = next(labeled_train_iter)
        except:
            labeled_train_iter = iter(labeled_trainloader)
            
            [inputs_x1, inputs_x2], labels_x, _, index = next(labeled_train_iter)

        # cross-entropy training with mixup
        batch_size = inputs_x1.size(0)
        inputs_x1, inputs_x2 = inputs_x1.cuda(), inputs_x2.cuda()
        labels_x = modified_label[index]
        targets_x = torch.zeros(batch_size, args.num_classes, device=inputs_x1.device).scatter_(1, labels_x.view(-1, 1), 1)
        l = np.random.beta(4, 4)
        l = max(l, 1 - l)
        all_inputs_x = torch.cat([inputs_x1, inputs_x2], dim=0)
        all_targets_x = torch.cat([targets_x, targets_x], dim=0)
        idx = torch.randperm(all_inputs_x.size()[0])
        input_a, input_b = all_inputs_x, all_inputs_x[idx]
        target_a, target_b = all_targets_x, all_targets_x[idx]
        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        logits = classifier(encoder(mixed_input))
        Lce = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * mixed_target, dim=1))

        # optional feature-consistency
        inputs_u1, inputs_u2 = inputs_u1.cuda(), inputs_u2.cuda()

        feats_u1 = encoder(inputs_u1)
        feats_u2 = encoder(inputs_u2)
        f, h = proj_head, pred_head

        z1, z2 = f(feats_u1), f(feats_u2)
        p1, p2 = h(z1), h(z2)
        Lfc = negative_cosine_sim(p2, z1)
        loss = Lce + args.lambda_fc * Lfc
        xlosses.update(Lce.item())
        ulosses.update(Lfc.item())
        all_bar.set_description(
            f'Train epoch {epoch+1} LR:{optimizer.param_groups[0]["lr"]} Labeled loss: {xlosses.avg:.4f} Unlabeled loss: {ulosses.avg:.4f}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test(testloader, encoder, classifier, epoch):
    encoder.eval()
    classifier.eval()
    accuracy = AverageMeter('accuracy')
    data_bar = tqdm(testloader)
    with torch.no_grad():
        for i, (data, label, _) in enumerate(data_bar):
            data, label = data.cuda(), label.cuda()
            feat = encoder(data)
            res = classifier(feat)
            pred = torch.argmax(res, dim=1)
            acc = torch.sum(pred == label) / float(data.size(0))
            accuracy.update(acc.item(), data.size(0))
            data_bar.set_description(f'Test epoch {epoch+1}: Accuracy#{accuracy.avg:.4f}')
    
    return accuracy.avg


def evaluate(dataloader, encoder, classifier, args, noisy_label, clean_label, i, stat_logs):
    encoder.eval()
    classifier.eval()
    feature_bank = []
    prediction = []

    ################################### feature extraction ###################################
    with torch.no_grad():
        # generate feature bank
        for (data, target, _, index) in tqdm(dataloader, desc='Feature extracting'):
            data = data.cuda()
            feature = encoder(data)
            feature_bank.append(feature)
            res = classifier(feature)
            prediction.append(res)
        feature_bank = F.normalize(torch.cat(feature_bank, dim=0), dim=1)

        ################################### sample relabelling ###################################
        prediction_cls = torch.softmax(torch.cat(prediction, dim=0), dim=1)
        his_score, his_label = prediction_cls.max(1)
        print(f'Prediction track: mean: {his_score.mean()} max: {his_score.max()} min: {his_score.min()}')
        conf_id = torch.where(his_score > args.gamma_r)[0]
        modified_label = torch.clone(noisy_label).detach()
        modified_label[conf_id] = his_label[conf_id]

        otsu_split = None
        if i>=args.warmup:
            
            group_1_clean, group_2_maybe_clean, group_3_maybe_noisy, group_4_noisy = split_dataset(his_score)
            otsu_split = {'clean_ids':torch.nonzero(group_1_clean),
                        'maybe_clean_ids': torch.nonzero(group_2_maybe_clean),
                        'maybe_noisy_ids': torch.nonzero(group_3_maybe_noisy),
                        'noisy_ids': torch.nonzero(group_4_noisy)
            }
        
        ################################### sample selection ###################################
        
        prediction_knn, _, _, _, _ = weighted_aknn(feature_bank, feature_bank, modified_label, args.num_classes,   otsu_split=otsu_split, ceil=args.ceil )  # temperature in weighted KNN
        vote_y = torch.gather(prediction_knn, 1, modified_label.view(-1, 1)).squeeze()
        vote_max = prediction_knn.max(dim=1)[0]
        right_score = vote_y / vote_max
        clean_id = torch.where(right_score >= args.theta_s)[0]
        noisy_id = torch.where(right_score < args.theta_s)[0]

        #fine
        if i>=args.warmup:
            temp_modified_label = modified_label.clone().cpu().detach().numpy()
            teacher_idx_1, _, _ = extract_cleanidx(feature_bank.cpu(), temp_modified_label, mode=args.distill_mode, gamma_e=args.gamma_e)

            temp_id = [idx for idx in clean_id.cpu().numpy().tolist() if (idx in otsu_split['maybe_noisy_ids']) or (idx in otsu_split['noisy_ids'])  ]

            for idx in teacher_idx_1:
                if (idx.item() in otsu_split['clean_ids']) or (idx.item() in otsu_split['maybe_clean_ids']):
                    temp_id.append(idx.item())
            clean_id = torch.tensor(temp_id).cuda()

            

        ################################### SSR monitor ###################################
        TP = torch.sum(modified_label[clean_id] == clean_label[clean_id])
        FP = torch.sum(modified_label[clean_id] != clean_label[clean_id])
        TN = torch.sum(modified_label[noisy_id] != clean_label[noisy_id])
        FN = torch.sum(modified_label[noisy_id] == clean_label[noisy_id])
        print(f'Epoch [{i+1}/{args.epochs}] selection: theta_s:{args.theta_s} TP: {TP} FP:{FP} TN:{TN} FN:{FN}')

        correct = torch.sum(modified_label[conf_id] == clean_label[conf_id])
        orginal = torch.sum(noisy_label[conf_id] == clean_label[conf_id])
        all = len(conf_id)
        
        print(f'Epoch [{i+1}/{args.epochs}] relabelling:  correct: {correct} original: {orginal} total: {all}')

        stat_logs.write(f'Epoch [{i}/{args.epochs}] selection: theta_s:{args.theta_s} TP: {TP} FP:{FP} TN:{TN} FN:{FN}\n')
        stat_logs.write(f'Epoch [{i}/{args.epochs}] relabelling:  correct: {correct} original: {orginal} total: {all}\n')
        stat_logs.flush()
                
    return clean_id, noisy_id, modified_label

def get_singular_vector(features, labels):
    '''
    To get top1 sigular vector in class-wise manner by using SVD of hidden feature vectors
    features: hidden feature vectors of data (numpy)
    labels: correspoding label list
    '''
    
    singular_vector_dict = {}
    with tqdm(total=len(np.unique(labels))) as pbar:
        for index in np.unique(labels):
            
            
            _, _, v = np.linalg.svd(features[labels==index])
            singular_vector_dict[index] = v[0]
            pbar.update(1)

    return singular_vector_dict  

def get_score(singular_vector_dict, features, labels, normalization=True):
    '''
    Calculate the score providing the degree of showing whether the data is clean or not.
    '''
    if normalization:
        scores = [np.abs(np.inner(singular_vector_dict[labels[indx]], feat/np.linalg.norm(feat))) for indx, feat in enumerate(tqdm(features))]
    else:
        scores = [np.abs(np.inner(singular_vector_dict[labels[indx]], feat)) for indx, feat in enumerate(tqdm(features))]    
    
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
        cls_index = indexes[labels==cls]
        feats = scores[labels==cls]
        feats_ = np.ravel(feats).astype(np.float).reshape(-1, 1)
        if feats_.shape[0]>=5:
            gmm = GaussianMixture(n_components=2, covariance_type='full', tol=1e-6, max_iter=10)
            
            gmm.fit(feats_)
            prob = gmm.predict_proba(feats_)
            prob = prob[:,gmm.means_.argmax()]
            for i in range(len(cls_index)):
                probs[cls_index[i]] = prob[i]
    
            clean_labels += [cls_index[clean_idx] for clean_idx in range(len(cls_index)) if prob[clean_idx] > gamma_e] 
        else:
            clean_labels += [cls_index[clean_idx] for clean_idx in range(len(cls_index))]
    
    return np.array(clean_labels, dtype=np.int64), probs
    
    
def fine(current_features, current_labels, fit = 'kmeans', prev_features=None, prev_labels=None, gamma_e=0.7):
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
    
    scores = get_score(singular_vector_dict, features = current_features, labels = current_labels)
    
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
        cls_index = indexes[labels==cls]
        kmeans = cluster.KMeans(n_clusters=2, random_state=0).fit(scores[cls_index].reshape(-1, 1))
        if np.mean(scores[cls_index][kmeans.labels_==0]) < np.mean(scores[cls_index][kmeans.labels_==1]): kmeans.labels_ = 1 - kmeans.labels_
            
        clean_labels += cls_index[kmeans.labels_ == 0].tolist()
        
    return np.array(clean_labels, dtype=np.int64)
    

def extract_cleanidx(features, labels, mode='fine-kmeans', gamma_e=0.6):
    
    scores=None
        
    # get teacher_idx
    if 'fine' in mode:

        teacher_idx, probs, scores = fine(current_features=features, current_labels=labels, fit = mode, gamma_e=gamma_e)
    
    teacher_idx = torch.tensor(teacher_idx)
    return teacher_idx, probs, scores




def main():
    args = parser.parse_args()
    seed_everything(args.seed)
    
    args.run_path = args.exp_name

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
    
    # generate noisy dataset with our transformation
    if not os.path.isdir(f'{args.dataset}'):
        os.mkdir(f'{args.dataset}')
    if not os.path.isdir(f'{args.dataset}/{args.run_path}'):
        os.mkdir(f'{args.dataset}/{args.run_path}')

    ############################# Dataset initialization ##############################################
    if args.dataset == 'cifar10':
        args.num_classes = 10
        args.image_size = 32
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    elif args.dataset == 'cifar100':
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

    # generate train dataset with only filtered clean subset
    train_data = cifar_dataset(dataset=args.dataset, root_dir=args.dataset_path,
                               open_name=args.open_name, open_dir=args.open_path,
                               transform=KCropsTransform(strong_transform, 2), open_ratio=args.open_ratio,
                               dataset_mode='train', noise_ratio=args.noise_ratio, noise_mode=args.noise_mode,
                               noise_file=f'{args.dataset}_{args.noise_ratio}_{args.open_ratio}_{args.noise_mode}_noise.json')
                               
    eval_data = cifar_dataset(dataset=args.dataset, root_dir=args.dataset_path, transform=weak_transform,
                              
                               open_name=args.open_name, open_dir=args.open_path,
                              dataset_mode='train', noise_ratio=args.noise_ratio, noise_mode=args.noise_mode,
                              open_ratio=args.open_ratio,
                              noise_file=f'{args.dataset}_{args.noise_ratio}_{args.open_ratio}_{args.noise_mode}_noise.json')
                            
    test_data = cifar_dataset(dataset=args.dataset, root_dir=args.dataset_path, transform=none_transform,
                              
                              open_name=args.open_name, open_dir=args.open_path, 
                              dataset_mode='test')
    all_data = cifar_dataset(dataset=args.dataset, root_dir=args.dataset_path,
                                   
                                   open_name=args.open_name, open_dir=args.open_path,
                                   transform=MixTransform(strong_transform=strong_transform, weak_transform=weak_transform, K=1),
                                   open_ratio=args.open_ratio,
                                   dataset_mode='train', noise_ratio=args.noise_ratio, noise_mode=args.noise_mode,
                                   noise_file=f'{args.dataset}_{args.noise_ratio}_{args.open_ratio}_{args.noise_mode}_noise.json')
                                

    # extract noisy labels and clean labels for performance monitoring
    
    noisy_label = torch.tensor(eval_data.cifar_label).cuda()
    clean_label = torch.tensor(eval_data.clean_label).cuda()

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    eval_loader = torch.utils.data.DataLoader(eval_data, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    all_loader = torch.utils.data.DataLoader(all_data, batch_size=args.batch_size, num_workers=4, shuffle=True, pin_memory=True, drop_last=True)

    ################################ Model initialization ###########################################
    encoder = PreResNet18(args.num_classes)
    classifier = torch.nn.Linear(encoder.fc.in_features, args.num_classes)
    proj_head = torch.nn.Sequential(torch.nn.Linear(encoder.fc.in_features, 256),
                                    torch.nn.BatchNorm1d(256),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(256, 128))
    pred_head = torch.nn.Sequential(torch.nn.Linear(128, 256),
                                    torch.nn.BatchNorm1d(256),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(256, 128))
    encoder.fc = torch.nn.Identity()

    encoder.cuda()
    classifier.cuda()
    proj_head.cuda()
    pred_head.cuda()

    #################################### Training initialization #######################################
    optimizer = SGD([{'params': encoder.parameters()}, {'params': classifier.parameters()}, {'params': proj_head.parameters()}, {'params': pred_head.parameters()}],
                    lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr/50.0)

    acc_logs = open(f'{args.dataset}/{args.run_path}/acc.txt', 'w')
    stat_logs = open(f'{args.dataset}/{args.run_path}/stat.txt', 'w')
    save_config(args, f'{args.dataset}/{args.run_path}')
    print('Train args: \n', args)
    best_acc = 0
    all_acc = []

    ################################ Training loop ###########################################
    for i in range(0,args.epochs):
        clean_id, noisy_id, modified_label = evaluate(eval_loader, encoder, classifier, args, noisy_label, clean_label, i, stat_logs)

        # balanced_sampler
        clean_subset = Subset(train_data, clean_id.cpu())
        sampler = ClassBalancedSampler(labels=modified_label[clean_id], num_classes=args.num_classes)
        labeled_loader = torch.utils.data.DataLoader(clean_subset, batch_size=args.batch_size, sampler=sampler, num_workers=4, drop_last=True)
     
        train(labeled_loader, modified_label, all_loader, encoder, classifier, proj_head, pred_head, optimizer, i, args)

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
                'pred_head': pred_head.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, filename=f'{args.dataset}/{args.run_path}/best_acc.pth.tar')
        acc_logs.write(f'Epoch [{i+1}/{args.epochs}]: Best accuracy@{best_acc}! Current accuracy@{cur_acc} \n')
        acc_logs.flush()
        print(f'Epoch [{i+1}/{args.epochs}]: Best accuracy@{best_acc}! Current accuracy@{cur_acc} \n')

    save_checkpoint({
        'cur_epoch': args.epochs,
        'classifier': classifier.state_dict(),
        'encoder': encoder.state_dict(),
        'proj_head': proj_head.state_dict(),
        'pred_head': pred_head.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, filename=f'{args.dataset}/{args.run_path}/last.pth.tar')
    

if __name__ == '__main__':
    main()