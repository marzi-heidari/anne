import os
import numpy as np
from torch.utils.data import Dataset
import json
import random
from torchvision.datasets.cifar import *


def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict


class cifar_dataset(Dataset):
    def __init__(self, dataset, root_dir, open_name,  open_dir, transform,  noise_mode='sym',
                 dataset_mode='train', noise_ratio=0.5, open_ratio=0.5, noise_file=None):

        self.r = noise_ratio  # total noise ratio
        self.on = open_ratio  # proportion of open noise
        self.transform = transform
        self.mode = dataset_mode
        self.transition = {0: 0, 2: 0, 4: 7, 7: 7, 1: 1, 9: 1, 3: 5, 5: 3, 6: 6, 8: 8}  # class transition for asymmetric noise
        self.open_noise = None
        self.closed_noise = None

        if self.mode == 'test':
            if dataset == 'cifar10':
                cifar_dic = unpickle('%s/cifar-10-batches-py/test_batch' % root_dir)
                self.cifar_data = cifar_dic['data']
                self.cifar_data = self.cifar_data.reshape((10000, 3, 32, 32))
                self.cifar_data = self.cifar_data.transpose((0, 2, 3, 1))
                self.cifar_label = cifar_dic['labels']
            elif dataset == 'cifar100':
                cifar_dic = unpickle('%s/cifar-100-python/test' % root_dir)
                self.cifar_data = cifar_dic['data'].reshape((10000, 3, 32, 32)).transpose((0, 2, 3, 1))
                self.cifar_label = cifar_dic['fine_labels']

        elif self.mode == 'train':
            if dataset == 'cifar10':
                cifar_data = []
                cifar_label = []
                for n in range(1, 6):
                    dpath = '%s/cifar-10-batches-py/data_batch_%d' % (root_dir, n)
                    data_dic = unpickle(dpath)
                    cifar_data.append(data_dic['data'])
                    cifar_label = cifar_label + data_dic['labels']
                self.cifar_data = np.concatenate(cifar_data).reshape((50000, 3, 32, 32)).transpose((0, 2, 3, 1))

            elif dataset == 'cifar100':
                cifar_dic = unpickle('%s/cifar-100-python/train' % root_dir)
                cifar_label = cifar_dic['fine_labels']
                self.cifar_data = cifar_dic['data'].reshape((50000, 3, 32, 32)).transpose((0, 2, 3, 1))
            self.clean_label = cifar_label

            if open_ratio > 0:
                if open_name == 'cifar100':
                    open_data = unpickle('%s/cifar-100-python/train' % open_dir)['data'].reshape((50000, 3, 32, 32)).transpose(
                        (0, 2, 3, 1))
                else:
                    print("Open noise dataset not supported")

            if os.path.exists(noise_file):
                noise = json.load(open(noise_file, "r"))
                noise_labels = noise['noise_labels']
                self.closed_noise = noise['closed_noise']
                if open_ratio > 0:
                    self.open_noise = noise['open_noise']
                    self.open_id = np.array(self.open_noise)[:, 0] if len(self.open_noise) !=0 else None
                
                    for cleanIdx, noisyIdx in noise['open_noise']:
                        if open_name == 'imagenet32':
                            self.cifar_data[cleanIdx] = np.asarray(
                                Image.open('{}/{}.png'.format(open_dir, str(noisyIdx + 1).zfill(7)))).reshape((32, 32, 3))
                            # set the groundtruth outliers label to be -1
                            self.clean_label[cleanIdx] = 10000
                        else:
                            self.cifar_data[cleanIdx] = open_data[noisyIdx]
                            self.clean_label[cleanIdx] = 10000
                self.cifar_label = noise_labels
            
            else:
                # inject noise
                noise_labels = []  # all labels (some noisy, some clean)
                idx = list(range(50000))  # indices of cifar dataset
                random.shuffle(idx)
                num_total_noise = int(self.r * 50000)  # total amount of noise
                num_open_noise = int(self.on * num_total_noise)  # total amount of noisy/openset images
                print('Statistics of synthetic noisy CIFAR dataset: ', 'num of clean samples: ', 50000 - num_total_noise,
                      ' num of closed-set noise: ', num_total_noise - num_open_noise, ' num of open-set noise: ', num_open_noise)
                # print(num_open_noise, num_total_noise)
                if open_name == 'imagenet32':  # indices of openset source images
                    target_noise_idx = list(range(1281149))
                else:
                    target_noise_idx = list(range(50000))
                random.shuffle(target_noise_idx)
                self.open_noise = list(
                    zip(idx[:num_open_noise], target_noise_idx[:num_open_noise]))  # clean sample -> openset sample mapping
                self.closed_noise = idx[num_open_noise:num_total_noise]  # closed set noise indices
                # populate noise_labels
                for i in range(50000):
                    if i in self.closed_noise:
                        if noise_mode == 'sym':
                            if dataset == 'cifar10':
                                noiselabel = random.randint(0, 9)
                            elif dataset == 'cifar100':
                                noiselabel = random.randint(0, 99)
                        elif noise_mode == 'asym':
                            noiselabel = self.transition[cifar_label[i]]
                        noise_labels.append(noiselabel)
                    else:
                        noise_labels.append(cifar_label[i])
                # populate openset noise images
                if open_ratio>0:
                    for cleanIdx, noisyIdx in self.open_noise:
                        if open_name == 'imagenet32':
                            self.cifar_data[cleanIdx] = np.asarray(
                                Image.open('{}/{}.png'.format(open_dir, str(noisyIdx + 1).zfill(7)))).reshape((32, 32, 3))
                            self.clean_label[cleanIdx] = 10000

                        else:
                            self.cifar_data[cleanIdx] = open_data[noisyIdx]
                            self.clean_label[cleanIdx] = 10000

                # write noise to a file, to re-use
                noise = {'noise_labels': noise_labels, 'open_noise': self.open_noise, 'closed_noise': self.closed_noise}
                print("save noise to %s ..." % noise_file)
                json.dump(noise, open(noise_file, "w"))
                self.cifar_label = noise_labels
                self.open_id = np.array(self.open_noise)[:, 0] if len(self.open_noise) !=0 else None

        else:
            raise ValueError(f'Dataset mode should be train or test rather than {self.dataset_mode}!')

    def update_labels(self, new_label):
        self.cifar_label = new_label.cpu()

    def __getitem__(self, index):
        # print(index)
        if self.mode == 'train':
            img = self.cifar_data[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            target = self.cifar_label[index]
            clean_target = self.clean_label[index]
            return img, target, clean_target, index
        else:
            img = self.cifar_data[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            target = self.cifar_label[index]
            return img, target, index

    def __len__(self):
        return len(self.cifar_data)

    def get_noise(self):
        return (self.open_noise, self.closed_noise)

    def __repr__(self):
        return f'dataset_mode: {self.mode}, dataset number: {len(self)} \n'

