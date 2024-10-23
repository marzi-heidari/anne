from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import torch
import os
import PIL
import torchvision
import tensorflow.compat.v1 as tf
# from preprocess_efnet import _decode_and_random_crop



class imagenet_dataset(Dataset):
    def __init__(self, root_dir, transform, num_class):
        # self.root = root_dir+'imagenet/val/'
        # self.root = root_dir+'images/'
        targets = []
        self.root = root_dir
        self.transform = transform
        self.val_data = []

        # with open(self.root+'ILSVRC2012_validation_ground_truth.txt') as f:
        #     lines=f.readlines()
        #     for li in lines:
        #         # print(li.rstrip())
        #         targets.append(int(li.rstrip()))
        # images = os.listdir(self.root+'images/')
        # for counter, img in enumerate(images):
        #     if targets[counter] < num_class:
        #         self.val_data.append([targets[counter],os.path.join(self.root,'images/',img)])      
        # import pdb; pdb.set_trace()
        # for c in range(num_class):
        #     imgs = os.listdir(self.root+str(c))
        #     for img in imgs:
        #         self.val_data.append([c,os.path.join(self.root,str(c),img)])      
        classes = []
        with open(self.root+'labels.txt') as f:
            lines=f.readlines()
            for li in lines:
                # print(li.split()[0].rstrip())d
                classes.append(li.split()[0].rstrip())   
        for c in classes[:num_class]:
            imgs = os.listdir(self.root+'images/'+str(c))
            for img in imgs:
                #self.val_data.append([c,os.path.join(self.root+'images/',str(c),img)])           
                self.val_data.append([classes.index(c),os.path.join(self.root+'images/',str(c),img)])           
                
    def __getitem__(self, index):
        data = self.val_data[index]
        target = data[0]
        image = Image.open(data[1]).convert('RGB')   
        img = self.transform(image) 
        return img, target
    
    def __len__(self):
        return len(self.val_data)

class webvision_dataset(Dataset): 
    def __init__(self, root_dir, transform, mode, num_class, pred=[], probability=[], log=''): 
        self.root = root_dir
        self.transform = transform
        self.mode = mode  
     
        if self.mode=='test':
            with open(self.root+'info/val_filelist.txt') as f:
                lines=f.readlines()
            self.val_imgs = []
            self.val_labels = {}
            for line in lines:
                img, target = line.split()
                target = int(target)
                if target<num_class:
                    self.val_imgs.append(img)
                    self.val_labels[img]=target                             
        else:    
            with open(self.root+'info/train_filelist_google.txt') as f:
                lines=f.readlines()    
            train_imgs = []
            self.train_labels = {}
            for line in lines:
                img, target = line.split()
                target = int(target)
                if target<num_class:
                    train_imgs.append(img)
                    self.train_labels[img]=target            
            if self.mode == 'all':
                self.train_imgs = train_imgs
            else:                   
                if self.mode == "labeled":
                    pred_idx = pred.nonzero()[0]
                    self.train_imgs = [train_imgs[i] for i in pred_idx]                
                    self.probability = [probability[i] for i in pred_idx]            
                    print("%s data has a size of %d"%(self.mode,len(self.train_imgs)))            
                    log.write('Numer of labeled samples:%d \n'%(pred.sum()))
                    log.flush()                          
                elif self.mode == "unlabeled":
                    pred_idx = (1-pred).nonzero()[0]                                               
                    self.train_imgs = [train_imgs[i] for i in pred_idx]                           
                    print("%s data has a size of %d"%(self.mode,len(self.train_imgs)))            
                    
    def __getitem__(self, index):
        if self.mode=='labeled':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path] 
            prob = self.probability[index]
            image = Image.open(self.root+img_path).convert('RGB')    
            img1 = self.transform(image) 
            img2 = self.transform(image) 
            return img1, img2, target, prob              
        elif self.mode=='unlabeled':
            img_path = self.train_imgs[index]
            image = Image.open(self.root+img_path).convert('RGB')    
            img1 = self.transform(image) 
            img2 = self.transform(image) 
            return img1, img2  
        elif self.mode=='all':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]     
            image = Image.open(self.root+img_path).convert('RGB')   
            img = self.transform(image)
            return img, target, index        
        elif self.mode=='test':
            img_path = self.val_imgs[index]
            target = self.val_labels[img_path]     
            image = Image.open(self.root+'val_images_256/'+img_path).convert('RGB')   
            img = self.transform(image) 
            return img, target
           
    def __len__(self):
        if self.mode!='test':
            return len(self.train_imgs)
        else:
            return len(self.val_imgs)  

class red_dataset(Dataset): 
    def __init__(self, root_dir, transform, mode, num_class, r,  color='blue', pred=[], probability=[], log=''): 
        self.root = root_dir
        self.transform = transform
        self.mode = mode  
        # self.dataset_name = dataset_name

        # if self.dataset_name =='cars':
        #     self.root = self.root + 'stanford_cars/'
        # elif self.dataset_name =='mini-imagenet':
        self.root = self.root + 'mini-imagenet/'
     
        if self.mode=='test':
            with open(self.root+'split/clean_validation') as f:            
                lines=f.readlines()
            self.val_imgs = []
            self.val_labels = {}
            for line in lines:
                img, target = line.split()

                target = int(target)

                img_path = 'validation_original/'+str(target) + '/'+img
                self.val_imgs.append(img_path)
                #self.val_labels[img]=target                             
                self.val_labels[img_path]=target                             
        else:    
            #with open(self.root+'info/train_filelist_google.txt') as f:
            noise_file = '{}_noise_nl_{}'.format(color,r)
            #with open(self.root+'info/train_filelist_google.txt') as f:
            with open(self.root+'split/'+noise_file) as f:
                lines=f.readlines()   

            train_imgs = []
            # self.train_labels = {}
            self.train_labels = []
            for line in lines:
                img, target = line.split()
                
                target = int(target)
                
                
                train_path = 'all_images/'
                # train_path = 'stanford_cars/training/'+noise_file + '/'+ str(target) + '/'

                train_imgs.append(train_path + img)
                #self.train_labels[img]=target            
                # self.train_labels[train_path + img]=target   
                self.train_labels.append(target)         
            if self.mode == 'all':
                self.train_imgs = train_imgs
            elif self.mode == 'warmup_reduced':
                self.train_imgs = train_imgs[5000:]
            elif self.mode == 'val_warmup':
                self.train_imgs = train_imgs[0:5000]
            else:                   
                if self.mode == "labeled":
                    pred_idx = pred.nonzero()[0]
                    self.train_imgs = [train_imgs[i] for i in pred_idx]                
                    self.probability = [probability[i] for i in pred_idx]            
                    print("%s data has a size of %d"%(self.mode,len(self.train_imgs)))            
                    # log.write('Numer of labeled samples:%d \n'%(pred.sum()))
                    # log.flush()                          
                elif self.mode == "unlabeled":
                    pred_idx = (1-pred).nonzero()[0]                                               
                    self.train_imgs = [train_imgs[i] for i in pred_idx]                           
                    print("%s data has a size of %d"%(self.mode,len(self.train_imgs)))  

    # def preprocess_for_train(image_bytes, use_bfloat16=False, image_size=299):
    #     """Preprocesses the given image for evaluation.

    #     Args:
    #     image_bytes: `Tensor` representing an image binary of arbitrary size.
    #     use_bfloat16: `bool` for whether to use bfloat16.
    #     image_size: image size.

    #     Returns:
    #     A preprocessed image `Tensor`.
    #     """
    #     image = _decode_and_random_crop(image_bytes, image_size)
    #     image = _flip(image)
    #     # image = tf.reshape(image, [image_size, image_size, 3])
    #     # image = tf.image.convert_image_dtype(
    #     # image, dtype=tf.bfloat16 if use_bfloat16 else tf.float32)
    #     return image 

                                    
                    
    def __getitem__(self, index):
        if self.mode=='labeled':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path] 
            prob = self.probability[index]
            image = Image.open(self.root+img_path).convert('RGB')    
            img1 = self.transform(image) 
            img2 = self.transform(image) 
            return img1, img2, target, prob              
        elif self.mode=='unlabeled':
            img_path = self.train_imgs[index]
            image = Image.open(self.root+img_path).convert('RGB')    
            img1 = self.transform(image) 
            img2 = self.transform(image) 
            return img1, img2  
        #elif self.mode=='all':
        elif self.mode in ['all', 'warmup_reduced', 'val_warmup']:
            img_path = self.train_imgs[index]
            # target = self.train_labels[img_path] 
            target = self.train_labels[index]    
            image = Image.open(self.root+img_path).convert('RGB')   
            # image = Image.open(self.root+img_path).convert('L').convert('RGB')
            # import pdb; pdb.set_trace()
            
            # image = _decode_and_random_crop(image, 299)
            img = self.transform(image)

            return img, target, index        
            # return img, target, img_path        
        elif self.mode=='test':
            img_path = self.val_imgs[index]
            target = self.val_labels[img_path]     
            #image = Image.open(self.root+'val_images_256/'+img_path).convert('RGB')   
            image = Image.open(self.root+img_path).convert('RGB')   
            # image = Image.open(self.root+img_path).convert('L').convert('RGB')   
            img = self.transform(image) 
            return img, target
           
    def __len__(self):
        if self.mode!='test':
            return len(self.train_imgs)
        else:
            return len(self.val_imgs)    


class red_dataloader():  
    #def __init__(self, r, color, batch_size, num_class, num_workers, root_dir, log, root_val):
    def __init__(self, r, color, batch_size, num_class, num_workers, root_dir, img_size=32):

        self.r = r
        self.color = color
        self.batch_size = batch_size
        self.num_class = num_class
        self.num_workers = num_workers
        self.root_dir = root_dir
        # self.root_val = root_val
        # self.log = log
        # self.dataset_name = dataset_name

        self.transform_train = transforms.Compose([
                transforms.Resize((32, 32), interpolation=2),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
            ]) 
         

        self.transform_test = transforms.Compose([

                transforms.Resize((32, 32), interpolation=2),
                transforms.ToTensor(),
                transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
            ])          

    def run(self,mode,pred=[],prob=[], args=None):
        if mode=='warmup':
            all_dataset = red_dataset(root_dir=self.root_dir, transform=self.transform_train, mode="all", num_class=self.num_class, r=self.r, color=self.color)                
            # all_dataset = torchvision.datasets.ImageFolder(root=self.root_dir+"train", transform = self.train_tfms)
            trainloader = DataLoader(
                dataset=all_dataset, 
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True)                 
            return trainloader

        elif mode=='warmup_reduced':
            all_dataset = red_dataset(root_dir=self.root_dir, transform=self.transform_train, mode="warmup_reduced", num_class=self.num_class, r=self.r, color=self.color)                
            trainloader = DataLoader(
                dataset=all_dataset, 
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True)                 
            return trainloader

        elif mode=='val_warmup':
            all_dataset = red_dataset(root_dir=self.root_dir, transform=self.transform_train, mode="val_warmup", num_class=self.num_class, r=self.r, color=self.color)                
            trainloader = DataLoader(
                dataset=all_dataset, 
                batch_size=self.batch_size*2,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True)                 
            return trainloader
                                     
        elif mode=='train':
            labeled_dataset = red_dataset(root_dir=self.root_dir, transform=self.transform_train, mode="labeled",num_class=self.num_class, r=self.r, color=self.color, pred=pred,probability=prob)              
            
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                drop_last=True ,
                pin_memory=True)        
            
            unlabeled_dataset = red_dataset(root_dir=self.root_dir, transform=self.transform_train, mode="unlabeled",num_class=self.num_class, r=self.r, color=self.color, pred=pred)                    
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                drop_last=True ,
                pin_memory=True)     
            return labeled_trainloader, unlabeled_trainloader

        elif mode=='contrastive':
            size1 = size= 32
            mean = [0.5071, 0.4867, 0.4408]
            std = [0.2675, 0.2565, 0.2761]
            transform_constrastive = torchvision.transforms.Compose([
                # torchvision.transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
                torchvision.transforms.Resize(size1, interpolation=PIL.Image.BICUBIC),
                torchvision.transforms.RandomCrop(size, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomApply([
                    torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                torchvision.transforms.RandomGrayscale(p=0.2),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean, std),
            ])
            labeled_dataset = red_dataset(root_dir=self.root_dir, transform=transform_constrastive, mode="labeled",num_class=self.num_class, r=self.r, color=self.color, pred=pred,probability=prob)              
            
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=args.batch_size_cl,
                shuffle=True,
                num_workers=self.num_workers,
                drop_last=True ,
                pin_memory=True)        
            
                
            return labeled_trainloader
        
        elif mode=='test':
            test_dataset = red_dataset(root_dir=self.root_dir, transform=self.transform_test, mode='test', num_class=self.num_class, r=0)      
            # test_dataset = torchvision.datasets.ImageFolder(root=self.root_dir+"test", transform = self.transform_test)
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                #num_workers=self.num_workers,
                num_workers=2,
                drop_last=True ,
                pin_memory=True)               
            return test_loader
        
        elif mode=='eval_train':
            eval_dataset = red_dataset(root_dir=self.root_dir, transform=self.transform_test, mode='all', num_class=self.num_class, r=self.r, color=self.color)      
            # eval_dataset = torchvision.datasets.ImageFolder(root=self.root_dir+"train", transform = self.transform_train)
            eval_loader = DataLoader(
                dataset=eval_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True)               
            return eval_loader     
        
        # elif mode=='imagenet':
        #     imagenet_val = imagenet_dataset(root_dir=self.root_val, transform=self.transform_imagenet, num_class=self.num_class)      
        #     imagenet_loader = DataLoader(
        #         dataset=imagenet_val, 
        #         batch_size=self.batch_size*20,
        #         shuffle=False,
        #         num_workers=self.num_workers,
        #         pin_memory=True)               
        #     return imagenet_loader     





