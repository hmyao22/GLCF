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



transform_x = T.Compose([T.Resize(224, Image.ANTIALIAS),
                         T.ToTensor(),
                         T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])

def load_image(path):
    x = Image.open(path).convert('RGB').resize((256,256))
    x = transform_x(x)
    x = x.unsqueeze(0)
    return x


def denormalize(img):
    std = np.array([0.229, 0.224, 0.225])
    mean = np.array([0.485, 0.456, 0.406])
    x = (((img.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x


def Getfiles(file_dir, file_type):
    file_list = []
    for image in os.listdir(file_dir):
            if image.endswith(('.%s' % file_type)):  # 判断文件类型
                file_list.append(os.path.join(file_dir, image))

    return file_list


def GetFiles(file_dir, file_type, IsCurrent=False):
    file_list = []
    for parent, dirnames, filenames in os.walk(file_dir):
        for filename in filenames:
            for type in file_type:
                if filename.endswith(('.%s'%type)):
                    file_list.append(os.path.join(parent, filename))
                
        if IsCurrent == True:
            break
    return file_list


class TrainFeatureData(data.Dataset):
    def __init__(self, feature):
        self.features = feature


    def __getitem__(self, item):
        feature = self.features[item]

        return feature

    def __len__(self):
        return len(self.features)

class ValidateData(data.Dataset):
    def __init__(self, opt):
        root = opt.validate_raw_data_root
        print(root)
        imgs = GetFiles(root, ["jpg","png","jpeg"])
        self.imgs = [img for img in imgs]


    def __getitem__(self, item):
        img_path = self.imgs[item]
        x = Image.open(img_path).convert('RGB').resize((256,256))
        img = transform_x(x)


        return img

    def __len__(self):
        return len(self.imgs)


class TrainData(data.Dataset):
    def __init__(self, opt):
        root = opt.train_raw_data_root
        print(root)
        imgs = GetFiles(root, ["jpg","png","jpeg"])
        self.imgs = [img for img in imgs]


    def __getitem__(self, item):
        img_path = self.imgs[item]
        x = Image.open(img_path).convert('RGB').resize((256,256))
        img = transform_x(x)


        return img

    def __len__(self):
        return len(self.imgs)


class TestData(data.Dataset):
    def __init__(self, opt):
        root = opt.test_raw_data_root
        imgs = GetFiles(root,["jpg","png","jpeg"])
        self.imgs = [img for img in imgs]

    def __getitem__(self, item):
        img_path = self.imgs[item]
        x = Image.open(img_path).convert('RGB').resize((256,256))
        img = transform_x(x)
        return img

    def __len__(self):
        return len(self.imgs)


class UniTrainData(data.Dataset):
    def __init__(self, opt):
        self.images =[]
        root = opt.data_root
        total_list = [
            'grid',
            'carpet',
            'leather',
            'tile',
            'wood',
            'bottle',
            'cable',
            'capsule',
            'hazelnut',
            'metal_nut',
            'toothbrush',
            'transistor',
            'zipper',
            'screw',
            'pill',
        ]
        for v in total_list:
            path_list = os.path.join(root, v, 'train')
            imgs = GetFiles(path_list, ["jpg","png","jpeg"])
            imgs = [img for img in imgs]
            self.images.extend(imgs)

    def __getitem__(self, item):
        img_path = self.images[item]
        x = Image.open(img_path).convert('RGB').resize((256,256))
        img = transform_x(x)

        return img

    def __len__(self):
        return len(self.images)


class TestData(data.Dataset):
    def __init__(self, opt):
        root = opt.test_raw_data_root
        imgs = GetFiles(root,["jpg","png","jpeg"])
        self.imgs = [img for img in imgs]

    def __getitem__(self, item):
        img_path = self.imgs[item]
        x = Image.open(img_path).convert('RGB').resize((256,256))
        img = transform_x(x)
        return img

    def __len__(self):
        return len(self.imgs)




def image_rotate_transform(image):
    dst1 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    dst2 = cv2.rotate(image, cv2.ROTATE_180)
    dst3 = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return [dst1,dst2,dst3]


def image_noise_transform(image):
    img1 =AddPepperNoise(0.7)(image)
    img2 =AddPepperNoise(0.7)(image)
    img3 =AddPepperNoise(0.7)(image)

    return [img1,img2,img3]


def image_cut_transform(image):
    imgs =_create_disjoint_masks(image)
    [img1,img2] =imgs

    return [img1,img2]


def image_Erase_transform(image):
    img1 =RandomErasing()(image)
    img2 =RandomErasing()(image)
    img3 =RandomErasing()(image)

    return [img1,img2,img3]


def image_blur_transform(image):
    img1 = cv2.blur(image, (1, 1))
    img2 = cv2.blur(image, (1, 1))
    img3 = cv2.blur(image, (1, 1))

    return [img1,img2,img3]

    
    
class AddPepperNoise(object):
    def __init__(self, snr):
        assert isinstance(snr, float) 
        self.snr = snr

    def __call__(self, img):
        img_ = np.array(img).copy()
        h, w, c = img_.shape
        signal_pct = self.snr
        noise_pct = (1 - self.snr)

        mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[signal_pct, noise_pct/2., noise_pct/2.])
        mask = np.repeat(mask, c, axis=2)
        img_[mask == 1] = 255 
        img_[mask == 2] = 0 
        return img_
    
def _create_disjoint_masks(img, cutout_size=32, num_disjoint_masks=2):
    img_h, img_w, img_C = img.shape
    grid_h = math.ceil(img_h / cutout_size)
    grid_w = math.ceil(img_w / cutout_size)
    num_grids = grid_h * grid_w
    disjoint_cut = []
    for grid_ids in np.array_split(np.random.permutation(num_grids), num_disjoint_masks):
        flatten_mask = np.zeros(num_grids)
        flatten_mask[grid_ids] = 1
        mask = flatten_mask.reshape((grid_h, grid_w))
        mask = mask.repeat(cutout_size, axis=0).repeat(cutout_size, axis=1)
        mask = np.tile(mask[:, :, np.newaxis], (1, 1, 3))
        cutout = np.array(img * mask,dtype=np.float32)
        disjoint_cut.append(cutout)
    return disjoint_cut

class RandomErasing:
    def __init__(self, sl=0.1, sh=0.5, r1=0.3):
        self.s = (sl, sh)
        self.r = (r1, 1 / r1)

    def __call__(self, img):
        image = img.copy()
        assert len(image.shape) == 3, 'image should be a 3 dimension numpy array'
        while True:
            Se = random.uniform(*self.s) * img.shape[0] * img.shape[1]
            re = random.uniform(*self.r)

            He = int(round(math.sqrt(Se * re)))
            We = int(round(math.sqrt(Se / re)))

            xe = random.randint(0, image.shape[1])
            ye = random.randint(0, image.shape[0])

            if xe + We <= image.shape[1] and ye + He <= image.shape[0]:
                image[ye: ye + He, xe: xe + We, :] = np.random.randint(low=0, high=255, size=(He, We, image.shape[2]))
                return image
            
            
def feature_create_disjoint_masks(cutout_size=8, num_disjoint_masks=2):
    img_h, img_w = 64, 64
    grid_h = math.ceil(img_h / cutout_size)
    grid_w = math.ceil(img_w / cutout_size)
    num_grids = grid_h * grid_w
    for grid_ids in np.array_split(np.random.permutation(num_grids), num_disjoint_masks):
        flatten_mask = np.zeros(num_grids)
        flatten_mask[grid_ids] = 1
        mask = flatten_mask.reshape((grid_h, grid_w))
        mask = mask.repeat(cutout_size, axis=0).repeat(cutout_size, axis=1)
        mask = np.tile(mask[np.newaxis, :, :], (1, 1, 1))
        mask = torch.tensor(mask).type(torch.FloatTensor)
    return mask


def normal(data):
    _range = np.max(data) - np.min(data)
    return data + np.min(data)


