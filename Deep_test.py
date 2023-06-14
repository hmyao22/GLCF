import torch.optim as opti
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn import init
import os
from datasets import TrainData, TestData
from config import DefaultConfig
import torch
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from gpu_mem_track import MemTracker
from datasets import normal

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
from datasets import TrainData, TestData
from torch.utils.data import DataLoader
from datasets import transform_x, denormalize
from torchvision import transforms as T
from PIL import Image
import time
import models.Model as Model

from fvcore.nn import FlopCountAnalysis,parameter_count_table

#opt.parse_model_root({'load_model_path': './check_points/VIT_SIZE/ckpt_base/'})
#opt.parse_model_root({'load_model_path': './check_points/DFT/without_DFT/base/'})

def load_image(path):
    transform_x = T.Compose([T.Resize(224, Image.ANTIALIAS),
                             T.ToTensor(),
                             T.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])])
    x = Image.open(path).convert('RGB').resize((256,256))
    x = transform_x(x)
    x = x.unsqueeze(0)
    return x


MVTEC_list = [
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
    'pill',
    'screw',
    'zipper',
    'transistor',
]



gpu_tracker = MemTracker()


image_path =r'D:\IMSN-YHM\dataset\cable\test\good\008.png'
#### structural_anomalies logical_anomalies

def model_test(image_path, show_feature_map=True):
    opt = DefaultConfig()
    class_name = image_path.split('\\')[-4]
    defect_name = image_path.split('\\')[-2]
    # label_path = os.path.join(r'D:\IMSN-YHM\dataset\mvtec_loco_anomaly_detection', class_name, 'ground_truth',
    #                           defect_name, image_path.split('\\')[-1].split('.')[0], '000.png')

    label_path = os.path.join(r'D:\IMSN-YHM\dataset', class_name, 'ground_truth',defect_name, image_path.split('\\')[-1].split('.')[0]+ '_mask.png')

    print(label_path)
    label_image = cv2.imread(label_path, 0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    original_tensor = load_image(image_path).to(device)

    model_name = class_name+'_SDCC2.pth'
    fuse_weight = torch.load(opt.load_model_path + 'fuse_weight/' + class_name + 'fuse_weight.pth')
    model = Model.SDCC_Model()
    if opt.use_gpu:
        model = model.to(device)
        if class_name in ['breakfast_box', 'splicing_connectors', 'screw_bag', 'juice_bottle', 'pushpins']:
            model.load_state_dict(torch.load(opt.load_model_path + 'loco/v2/' + model_name, map_location=device))
        if class_name in MVTEC_list:
            model.load_state_dict(torch.load(opt.load_model_path + 'mvtec/' + model_name, map_location=device))

    #### logical/struc ####
    [dis_map1, dis_map2, dis_map3], [rec_map1, rec_map2, rec_map3] = model.a_map(original_tensor)

    # dis_map1 = gaussian_filter(dis_map1, sigma=4)
    # dis_map2 = gaussian_filter(dis_map2, sigma=4)
    # dis_map3 = gaussian_filter(dis_map3, sigma=4)
    #
    # rec_map1 = gaussian_filter(rec_map1, sigma=4)
    # rec_map2 = gaussian_filter(rec_map2, sigma=4)
    # rec_map3 = gaussian_filter(rec_map3, sigma=4)



    ############## scale fuse ###############

    dis_map1 = (dis_map1 - fuse_weight['scale1'][1]) / fuse_weight['scale1'][0]
    dis_map2 = (dis_map2 - fuse_weight['scale2'][1]) / fuse_weight['scale2'][0]
    dis_map3 = (dis_map3 - fuse_weight['scale3'][1]) / fuse_weight['scale3'][0]

    rec_map1 = (rec_map1 - fuse_weight['scale1'][3]) / fuse_weight['scale1'][2]
    rec_map2 = (rec_map2 - fuse_weight['scale2'][3]) / fuse_weight['scale2'][2]
    rec_map3 = (rec_map3 - fuse_weight['scale3'][3]) / fuse_weight['scale3'][2]

    dis_amap = dis_map1 + dis_map2*3 + dis_map3*3
    rec_map = rec_map1*1 + rec_map2*2 + rec_map3*3

    #### logical/struc ###
    fuse_map = dis_amap*5 + rec_map*1

    input_frame = denormalize(original_tensor.clone().squeeze(0).cpu().detach().numpy())
    cv2_input = np.array(input_frame, dtype=np.uint8)


    if show_feature_map == True:
        deep_feature, local_feature, global_feature, global_feature_reg = model(original_tensor)
        plt.figure()
        plt.subplot(4, 3, 1)
        plt.imshow(deep_feature[0][0][:, :, 0].cpu().detach().numpy())
        plt.subplot(4, 3, 2)
        plt.imshow(deep_feature[1][0][:, :, 0].cpu().detach().numpy())
        plt.subplot(4, 3, 3)
        plt.imshow(deep_feature[2][0][:, :, 0].cpu().detach().numpy())

        plt.subplot(4, 3, 4)
        plt.imshow(local_feature[0][0][:, :, 0].cpu().detach().numpy())
        plt.subplot(4, 3, 5)
        plt.imshow(local_feature[1][0][:, :, 0].cpu().detach().numpy())
        plt.subplot(4, 3, 6)
        plt.imshow(local_feature[2][0][:, :, 0].cpu().detach().numpy())

        plt.subplot(4, 3, 7)
        plt.imshow(global_feature[0][0][:, :, 0].cpu().detach().numpy())
        plt.subplot(4, 3, 8)
        plt.imshow(global_feature[1][0][:, :, 0].cpu().detach().numpy())
        plt.subplot(4, 3, 9)
        plt.imshow(global_feature[2][0][:, :, 0].cpu().detach().numpy())

        plt.subplot(4, 3, 10)
        plt.imshow(global_feature_reg[0][0][:, :, 0].cpu().detach().numpy())
        plt.subplot(4, 3, 11)
        plt.imshow(global_feature_reg[1][0][:, :, 0].cpu().detach().numpy())
        plt.subplot(4, 3, 12)
        plt.imshow(global_feature_reg[2][0][:, :, 0].cpu().detach().numpy())

        plt.show()

    plt.figure(figsize=(8, 4))
    plt.subplot(3, 4, 1)
    plt.axis('off')
    plt.imshow(dis_map1, cmap='jet')
    plt.colorbar()
    plt.title(dis_map1.mean())
    plt.subplot(3, 4, 2)
    plt.axis('off')
    plt.imshow(dis_map2, cmap='jet')
    plt.colorbar()
    plt.title(dis_map2.mean())
    plt.subplot(3, 4, 3)
    plt.axis('off')
    plt.imshow(dis_map3, cmap='jet')
    plt.colorbar()
    plt.title(dis_map3.mean())
    plt.subplot(3, 4, 4)
    plt.axis('off')
    plt.imshow(dis_amap, cmap='jet')
    plt.colorbar()
    plt.title(dis_amap.mean())

    plt.subplot(3, 4, 5)
    plt.axis('off')
    plt.imshow(rec_map1, cmap='jet')
    plt.colorbar()
    plt.title(rec_map1.mean())
    plt.subplot(3, 4, 6)
    plt.axis('off')
    plt.imshow(rec_map2, cmap='jet')
    plt.colorbar()
    plt.title(rec_map2.mean())
    plt.subplot(3, 4, 7)
    plt.axis('off')
    plt.imshow(rec_map3, cmap='jet')
    plt.colorbar()
    plt.title(rec_map3.mean())
    plt.subplot(3, 4, 8)
    plt.axis('off')
    plt.imshow(rec_map, cmap='jet')
    plt.colorbar()
    plt.title(rec_map.mean())

    plt.subplot(3, 4, 9)
    plt.axis('off')
    plt.imshow(cv2_input)
    plt.subplot(3, 4, 10)
    plt.axis('off')
    plt.imshow(fuse_map, cmap='jet')
    plt.colorbar()
    plt.title(fuse_map.std())
    plt.subplot(3, 4, 11)
    plt.axis('off')
    plt.imshow(label_image, cmap='jet')
    plt.title('label')

    plt.show()


if __name__ =="__main__":
    model_test(image_path)