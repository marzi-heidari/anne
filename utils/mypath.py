class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'webvision':
            return 'samples/webvision/'
        elif dataset == 'clothing':
            return 'samples/clothing1M/'
        elif dataset == 'miniimagenet_preset':
            return 'samples/mini-imagenet/'
        elif dataset == 'cifar100':
            return '../../../datasets/cifar100/'
        elif dataset == 'imagenet32':
            return '../../../datasets/'
        else:
            raise NotImplementedError('Dataset {} not available.'.format(dataset))
        
