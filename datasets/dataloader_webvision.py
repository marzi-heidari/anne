import os

from PIL import Image
from torch.utils.data import Dataset


class imagenet_dataset(Dataset):
    def __init__(self, transform, num_class=50, root_dir='./'):
        # self.root = root_dir + '/val/'
        self.root = root_dir + '/imagenet_val/'
        self.transform = transform
        self.val_data = []
        classes = os.listdir(self.root)
        classes.sort()
        for c in range(num_class):
            imgs = os.listdir(self.root + classes[c])
            for img in imgs:
                self.val_data.append([c, os.path.join(self.root, classes[c], img)])

    def __getitem__(self, index):
        data = self.val_data[index]
        target = data[0]
        image = Image.open(data[1]).convert('RGB')
        img = self.transform(image)
        return img, target, index

    def __len__(self):
        return len(self.val_data)
    
class imagenet_dataset_v2(Dataset):
    def __init__(self, transform, num_class=50, root_dir='./'):
        # self.root = root_dir + '/val/'
        # self.root = root_dir + '/imagenet_val/'
        self.root = root_dir
        self.transform = transform
        # self.val_data = []

        self.val_imgs = []    #filtered images containing the first 100 classes
        self.val_labels = {}
        # classes = os.listdir(self.root)
        # classes.sort()
        # for c in range(num_class):
        #     imgs = os.listdir(self.root + classes[c])
        #     for img in imgs:
        #         self.val_data.append([c, os.path.join(self.root, classes[c], img)])

        all_50k_images = [ x for x in os.listdir(self.root) if os.path.isfile(self.root+x)]
        all_50k_images.sort()

        classes1000 = []
        with open(self.root+'data/ILSVRC2012_validation_to_webvision_labels_ground_truth.txt') as f:
            lines=f.readlines()
            for li in lines:
                classes1000.append(int(li))

        assert(len(classes1000)==len(all_50k_images))

        for id, img_name in enumerate(all_50k_images):
            if int(classes1000[id])<50:  #does not have class 0
                self.val_imgs.append(img_name)
                #self.val_labels.append(classes1000[id])
                self.val_labels[img_name] = int(classes1000[id])

    def __getitem__(self, index):
        # data = self.val_data[index]
        img_path = self.val_imgs[index]
        # target = data[0]
        target = self.val_labels[img_path]  
        # image = Image.open(data[1]).convert('RGB')
        image = Image.open(self.root+img_path).convert('RGB') 
        img = self.transform(image)
        return img, target, index

    def __len__(self):
        #return len(self.val_data)
        return len(self.val_imgs)


class miniwebvision_dataset(Dataset):
    def __init__(self, root_dir, transform, dataset_mode, num_class=50):
        self.root = root_dir
        self.transform = transform
        self.mode = dataset_mode

        if self.mode == 'test':
            with open(self.root + '/info/val_filelist.txt') as f:
                lines = f.readlines()
            self.val_imgs = []
            self.val_labels = {}
            for line in lines:
                img, target = line.split()
                target = int(target)
                if target < num_class:
                    self.val_imgs.append(img)
                    self.val_labels[img] = target
        elif self.mode == 'train':
            with open(self.root + '/info/train_filelist_google.txt') as f:
                lines = f.readlines()
            self.train_imgs = []
            # self.train_labels = {}
            self.train_labels = []
            for line in lines:
                img, target = line.split()
                target = int(target)
                if target < num_class:
                    self.train_imgs.append(img)
                    # self.train_labels[img] = target
                    self.train_labels.append(target)
        else:
            raise ValueError(f'dataset_mode should be train or test, rather than {self.mode}!')

    def update_labels(self, new_label_dict):
        if self.mode == 'train':
            self.train_labels = new_label_dict.cpu()
        else:
            raise ValueError(f'Dataset mode should be train rather than {self.mode}!')

    def __getitem__(self, index):
        if self.mode == 'train':
            img_path = self.train_imgs[index]
            target = self.train_labels[index]
            image = Image.open(self.root + '/' + img_path).convert('RGB')
            img = self.transform(image)
            return img, target, index
        elif self.mode == 'test':
            img_path = self.val_imgs[index]
            target = self.val_labels[img_path]
            image = Image.open(self.root + '/val_images_256/' + img_path).convert('RGB')
            img = self.transform(image)
            return img, target, index

    def __len__(self):
        if self.mode != 'test':
            return len(self.train_imgs)
        else:
            return len(self.val_imgs)
