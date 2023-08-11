from torch.utils import data
import os
import cv2
import numpy as np
import torch
import math
import random
import torchvision.transforms as transforms
import os
from PIL import Image
from torchvision import transforms as T
import torch
import torch.nn.functional as F
import torch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch.optim as optim
import os
from matplotlib import pyplot as plt
import glob
import random
from tqdm import tqdm


transform_x = T.Compose([T.Resize(112, Image.ANTIALIAS),
                         T.ToTensor(),
                         T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])

def denormalize(img):
    std = np.array([0.229, 0.224, 0.225])
    mean = np.array([0.485, 0.456, 0.406])
    x = (((img.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x


class MVTecTrainDataset(Dataset):

    def __init__(self, class_name, root_dir=r'D:\IMSN-YHM\dataset', resize_shape=None):
        self.root_dir = os.path.join(root_dir, class_name, 'train')
        self.images = sorted(glob.glob(self.root_dir+"/*/*.png"))
        self.resize_shape=resize_shape

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        img_path = self.images[item]
        x = Image.open(img_path).convert('RGB').resize((256,256))
        img = transform_x(x)
        return img



class MVTecTestDataset(Dataset):

    def __init__(self, class_name, root_dir=r'D:\IMSN-YHM\dataset', resize_shape=None):
        self.root_dir = root_dir + "/" + class_name + "/test"
        self.images = sorted(glob.glob(self.root_dir+"/*/*.png"))
        self.resize_shape=resize_shape

    def __len__(self):
        return len(self.images)

    def transform_image(self, image_path, mask_path):
        x = Image.open(image_path).convert('RGB')
        image = transform_x(x)
        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            image0 = cv2.imread(image_path)
            mask = np.zeros((image0.shape[0], image0.shape[1]))
        if self.resize_shape != None:
            mask = cv2.resize(mask, dsize=(self.resize_shape[1], self.resize_shape[0]))

        # mask = mask / 255.0
        mask = np.array(mask)
        mask = cv2.resize(mask, dsize=(112, 112))
        # mask = np.transpose(mask, (2, 0, 1))
        return image, mask

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images[idx]
        dir_path, file_name = os.path.split(img_path)
        base_dir = os.path.basename(dir_path)
        if base_dir == 'good':
            image, mask = self.transform_image(img_path, None)
            has_anomaly = np.array([0], dtype=np.float32)
        else:
            mask_path = os.path.join(dir_path, '../../ground_truth/')
            mask_path = os.path.join(mask_path, base_dir)
            mask_file_name = file_name.split(".")[0]+"_mask.png"
            #mask_file_name = file_name.split(".")[0].split("_")[0] +'_gt_' +file_name.split(".")[0].split("_")[-1]+ ".png"
            mask_path = os.path.join(mask_path, mask_file_name)

            image, mask = self.transform_image(img_path, mask_path)
            has_anomaly = np.array([1], dtype=np.float32)
        sample = {'image': image, 'has_anomaly': has_anomaly,'mask': mask, 'idx': idx}

        return sample


class MVTecAAD(torch.utils.data.Dataset):
    def __init__(self, array=['carpet', 'tile', 'wood','leather'], length=100):
        self.length = length
        self.dataset = {}
        self.img_size = 112

        # load data
        for i,class_name in enumerate(array):
            ClassDataset = MVTecTrainDataset(class_name=class_name)
            self.dataset[i] = DataLoader(ClassDataset, batch_size=1, shuffle=True, drop_last=True, num_workers=0)



    def make_normal_img(self):
        rows = []
        row = []
        for i in range(4):
            data = next(iter(self.dataset[i]))
            # random scale
            row.append(data)
            if (i+1) % 2 == 0:
                row = torch.cat(row, dim=2)
                rows.append(row)
                row = []
        img = torch.cat(rows, dim=3)
        img = img.squeeze()
        sample = {'image': img}
        return sample

    def __getitem__(self, index):
        sample = self.make_normal_img()

        return sample

    def __len__(self):
        return self.length


class MVTecAADTest(torch.utils.data.Dataset):
    def __init__(self, array=['carpet', 'tile', 'wood','leather'], type='global', length=100):
        self.length = length
        self.array = array
        self.dataset = {}
        self.type = type
        # load data
        for i, class_name in enumerate(array):
            ClassDataset = MVTecTestDataset(class_name=class_name)
            self.dataset[i] = DataLoader(ClassDataset, batch_size=1, shuffle=True, drop_last=True, num_workers=0)

    def __getitem__(self, index):
        if self.type == 'normal':
            sample = self.make_normal(index)
        if self.type == 'local':
            sample = self.make_local(index)
        if self.type == 'global':
            sample = self.make_global(index)

        return sample


    def make_normal(self, index):
        rows = []
        row = []
        for i in range(4):
            sample = next(iter(self.dataset[i]))
            while sample['has_anomaly'] != 0:
                sample = next(iter(self.dataset[i]))
            data = sample['image']
            row.append(data)
            if (i + 1) % 2 == 0:
                row = torch.cat(row, dim=2)
                rows.append(row)
                row = []
        img = torch.cat(rows, dim=3)
        img = img.squeeze()
        sample = {'image': img}
        return sample


    def make_local(self, index):
        rows = []
        row = []

        mask_rows = []
        mask_row = []
        for i in range(4):
            sample = next(iter(self.dataset[i]))
            if sample['has_anomaly'] != 1:
                sample = next(iter(self.dataset[i]))
            data = sample['image']
            mask = sample['mask']

            row.append(data)
            mask_row.append(mask)

            if (i + 1) % 2 == 0:
                row = torch.cat(row, dim=2)
                rows.append(row)
                row = []


                mask_row = torch.cat(mask_row, dim=1)
                mask_rows.append(mask_row)
                mask_row = []

        img = torch.cat(rows, dim=3)
        img = img.squeeze()

        mask = torch.cat(mask_rows, dim=2)
        mask = mask.squeeze().unsqueeze(-1)
        mask = np.ascontiguousarray(mask)
        mask = np.array(mask, dtype=np.uint8)
        sample = {'image': img, 'mask': mask}
        return sample



    def make_global(self, index):
        original_array = self.array
        new_array = np.random.choice(original_array, len(original_array), replace=True)
        while np.array_equal(original_array, new_array):
            new_array = np.random.choice(original_array, len(original_array), replace=True)
        new_dataset = {}

        for i, class_name in enumerate(new_array):
            ClassDataset = MVTecTestDataset(class_name=class_name)
            new_dataset[i] = DataLoader(ClassDataset, batch_size=1, shuffle=True, drop_last=True, num_workers=0)

        rows = []
        row = []
        mask_rows = []
        mask_row = []

        for i in range(4):
            sample = next(iter(new_dataset[i]))
            while sample['has_anomaly'] != 0:
                sample = next(iter(new_dataset[i]))
            data = sample['image']

            if original_array[i] != new_array[i]:
                mask = torch.tensor(np.array(255*np.ones((1, 112, 112))))
            else:
                # mask = sample['mask']
                mask = torch.tensor(np.array(np.zeros((1, 112, 112))))

            row.append(data)
            mask_row.append(mask)

            if (i + 1) % 2 == 0:
                row = torch.cat(row, dim=2)
                rows.append(row)
                row = []

                mask_row = torch.cat(mask_row, dim=1)
                mask_rows.append(mask_row)
                mask_row = []

        img = torch.cat(rows, dim=3)
        img = img.squeeze()

        mask = torch.cat(mask_rows, dim=2)
        mask = mask.squeeze()
        sample = {'image': img, 'mask': mask}
        return sample


    def __len__(self):
        return self.length


if __name__ == "__main__":
    # array1 = ['carpet', 'tile', 'wood', 'leather']
    # array2 = ['grid', 'bottle', 'cable', 'capsule']
    # array3 = ['hazelnut', 'metal_nut', 'pill', 'screw']
    array1 = ['toothbrush', 'transistor', 'zipper', 'grid']

    save_path = r'D:\IMSN-YHM\dataset\mvtecaad'

    name = 'ttzg'

    ################ TRAIN #################
    # train_path = os.path.join(save_path, name, 'train', 'good')
    # if not os.path.exists(train_path):
    #     os.makedirs(train_path)
    #
    # dataset = MVTecAAD(array=array1)
    # train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=True, num_workers=0)
    # for index, item in enumerate(tqdm(train_dataloader, ncols=80)):
    #     input_frame = item['image']
    #     input_frame = denormalize(input_frame.clone().squeeze(0).cpu().detach().numpy())
    #     input_frame = input_frame[..., ::-1]
    #     image_path = os.path.join(train_path, str(index)+'.png')
    #     cv2.imwrite(image_path,input_frame)
    ################ TEST GOOD#################

    # test_normal_path = os.path.join(save_path, name, 'test', 'good')
    # if not os.path.exists(test_normal_path):
    #     os.makedirs(test_normal_path)
    # test_normal_dataset = MVTecAADTest(array=array1, type='normal')
    # test_normal_dataloader = DataLoader(test_normal_dataset, batch_size=1, shuffle=True, drop_last=True, num_workers=0)
    # for index, item in enumerate(tqdm(test_normal_dataloader, ncols=80)):
    #     input_frame = item['image']
    #     input_frame = denormalize(input_frame.clone().squeeze(0).cpu().detach().numpy())
    #     input_frame = input_frame[..., ::-1]
    #     image_path = os.path.join(test_normal_path, str(index) + '.png')
    #     cv2.imwrite(image_path, input_frame)

    ################ TEST LOCAL#################

    # test_local_path = os.path.join(save_path, name, 'test', 'local')
    # gt_local_path = os.path.join(save_path, name, 'ground_truth', 'local')
    # if not os.path.exists(test_local_path):
    #     os.makedirs(test_local_path)
    # if not os.path.exists(gt_local_path):
    #     os.makedirs(gt_local_path)
    #
    # test_normal_dataset = MVTecAADTest(array=array1, type='local')
    # test_normal_dataloader = DataLoader(test_normal_dataset, batch_size=1, shuffle=True, drop_last=True,
    #                                         num_workers=0)
    # for index, item in enumerate(tqdm(test_normal_dataloader, ncols=80)):
    #     input_frame = item['image']
    #     gt_image = item['mask'].clone().squeeze(0).cpu().detach().numpy()
    #     input_frame = denormalize(input_frame.clone().squeeze(0).cpu().detach().numpy())
    #     input_frame = input_frame[..., ::-1]
    #     image_path = os.path.join(test_local_path, str(index) + '.png')
    #     cv2.imwrite(image_path, input_frame)
    #
    #     gt_path = os.path.join(gt_local_path, str(index) + '_mask.png')
    #     cv2.imwrite(gt_path, gt_image)

    ################ TEST GOLOBAL#################

    test_global_path = os.path.join(save_path, name, 'test', 'global')
    gt_global_path = os.path.join(save_path, name, 'ground_truth', 'global')
    if not os.path.exists(test_global_path):
        os.makedirs(test_global_path)
    if not os.path.exists(gt_global_path):
        os.makedirs(gt_global_path)

    test_normal_dataset = MVTecAADTest(array=array1, type='global')
    test_normal_dataloader = DataLoader(test_normal_dataset, batch_size=1, shuffle=True, drop_last=True,
                                            num_workers=0)
    for index, item in enumerate(tqdm(test_normal_dataloader, ncols=80)):
        input_frame = item['image']
        gt_image = item['mask'].clone().squeeze(0).cpu().detach().numpy()
        input_frame = denormalize(input_frame.clone().squeeze(0).cpu().detach().numpy())
        input_frame = input_frame[..., ::-1]
        image_path = os.path.join(test_global_path, str(index) + '.png')
        cv2.imwrite(image_path, input_frame)

        gt_path = os.path.join(gt_global_path, str(index) + '_mask.png')
        cv2.imwrite(gt_path, gt_image)




