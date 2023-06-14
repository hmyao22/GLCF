# import torch.optim as opti
# import torch.nn as nn
# from torch.autograd import Variable
# from torch.utils.data import DataLoader
# from torch.nn import init
# import os
# from datasets import TrainData, TestData, UniTrainData,ValidateData
# from config import DefaultConfig
# import torch
# import cv2
# import numpy as np
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# import models.Model as Model
# from scipy.ndimage import gaussian_filter
# from datasets import denormalize
# from models.misc import NativeScalerWithGradNormCount as NativeScaler
# import measure
# from models.utils import adjust_learning_rate
# from prefetch_generator import BackgroundGenerator
#
# def fuse_weight_cal(opt):
#     device = opt.device
#     print(device)
#
#     model = Model.SDCC_Model()
#     model_name = opt.class_name+'_SDCC2.pth'
#
#     if opt.use_gpu:
#         model.to(device)
#     model.load_state_dict(torch.load(opt.load_model_path + model_name,  map_location=device))
#     model.eval()
#     print("load weights!")
#
#     validateDataset = TrainData(opt=opt)
#     validate_dataloader = DataLoader(validateDataset, batch_size=1, shuffle=True, drop_last=True, num_workers=0, pin_memory=True)
#     local_std_ave_scale1 = 0
#     local_mean_ave_scale1 = 0
#
#     local_std_ave_scale2 = 0
#     local_mean_ave_scale2 = 0
#
#     local_std_ave_scale3 = 0
#     local_mean_ave_scale3 = 0
#
#
#     global_std_ave_scale1 = 0
#     global_mean_ave_scale1 = 0
#
#     global_std_ave_scale2 = 0
#     global_mean_ave_scale2 = 0
#
#     global_std_ave_scale3 = 0
#     global_mean_ave_scale3 = 0
#
#
#     total_local_map1 = []
#     total_local_map2 = []
#     total_local_map3 = []
#
#     total_global_map1 = []
#     total_global_map2 = []
#     total_global_map3 = []
#
#     for index, item in enumerate(tqdm(validate_dataloader, ncols=80)):
#         input_frame = item
#
#         if opt.use_gpu:
#             input_frame = input_frame.to(device, non_blocking=True)
#
#         [local_map1, local_map2, local_map3], [global_map1, global_map2, global_map3]= model.a_map(input_frame)
#
#         local_map1 = gaussian_filter(local_map1, sigma=4)
#         local_map2 = gaussian_filter(local_map2, sigma=4)
#         local_map3 = gaussian_filter(local_map3, sigma=4)
#
#         global_map1 = gaussian_filter(global_map1, sigma=4)
#         global_map2 = gaussian_filter(global_map2, sigma=4)
#         global_map3 = gaussian_filter(global_map3, sigma=4)
#         ############# local ############
#         total_local_map1.append(local_map1)
#         total_local_map2.append(local_map2)
#         total_local_map3.append(local_map3)
#
#         ############# global ############
#         total_global_map1.append(global_map1)
#         total_global_map2.append(global_map2)
#         total_global_map3.append(global_map3)
#
#     total_local_map1 = np.array(total_local_map1)
#     total_local_map2 = np.array(total_local_map2)
#     total_local_map3 = np.array(total_local_map3)
#
#     total_global_map1 = np.array(total_global_map1)
#     total_global_map2 = np.array(total_global_map2)
#     total_global_map3 = np.array(total_global_map3)
#
#
#     local_std_ave_scale1 = np.mean(total_local_map1, axis=0)
#     local_mean_ave_scale1 = np.std(total_local_map1, axis=0)
#
#     local_std_ave_scale2 = np.mean(total_local_map2, axis=0)
#     local_mean_ave_scale2 = np.std(total_local_map2, axis=0)
#
#     local_std_ave_scale3 = np.mean(total_local_map3, axis=0)
#     local_mean_ave_scale3 = np.std(total_local_map3, axis=0)
#
#
#     global_std_ave_scale1 = np.mean(total_global_map1, axis=0)
#     global_mean_ave_scale1 = np.std(total_global_map1, axis=0)
#
#     global_std_ave_scale2 = np.mean(total_global_map2, axis=0)
#     global_mean_ave_scale2 = np.std(total_global_map2, axis=0)
#
#     global_std_ave_scale3 = np.std(total_global_map3, axis=0)
#     global_mean_ave_scale3 = np.std(total_global_map3, axis=0)
#
#
#     plt.figure()
#     plt.subplot(4,3,1)
#     plt.imshow(local_std_ave_scale1)
#     plt.subplot(4,3,2)
#     plt.imshow(local_std_ave_scale2)
#     plt.subplot(4,3,3)
#     plt.imshow(local_std_ave_scale3)
#     plt.subplot(4,3,4)
#     plt.imshow(global_std_ave_scale1)
#     plt.subplot(4,3,5)
#     plt.imshow(global_std_ave_scale2)
#     plt.subplot(4,3,6)
#     plt.imshow(global_std_ave_scale3)
#
#
#     plt.subplot(4,3,7)
#     plt.imshow(local_mean_ave_scale1)
#     plt.subplot(4,3,8)
#     plt.imshow(local_mean_ave_scale2)
#     plt.subplot(4,3,9)
#     plt.imshow(local_mean_ave_scale3)
#     plt.subplot(4,3,10)
#     plt.imshow(global_mean_ave_scale1)
#     plt.subplot(4,3,11)
#     plt.imshow(global_mean_ave_scale2)
#     plt.subplot(4,3,12)
#     plt.imshow(global_mean_ave_scale3)
#     plt.show()
#
#
#
#
#     fuse_weight = {
#     'scale1':[local_std_ave_scale1, local_mean_ave_scale1, global_std_ave_scale1, global_mean_ave_scale1],
#     'scale2':[local_std_ave_scale2, local_mean_ave_scale2, global_std_ave_scale2, global_mean_ave_scale2],
#     'scale3':[local_std_ave_scale3, local_mean_ave_scale3, global_std_ave_scale3, global_mean_ave_scale3]
#     }
#
#     torch.save(fuse_weight, opt.load_model_path + opt.class_name + 'fuse_weight.pth')
#
# opt = DefaultConfig()
# opt.parse({'class_name': 'breakfast_box'})
# fuse_weight_cal(opt)


import torch.optim as opti
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn import init
import os
from datasets import TrainData, TestData, UniTrainData,ValidateData
from config import DefaultConfig
import torch
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import models.Model as Model
from scipy.ndimage import gaussian_filter
from datasets import denormalize
from models.misc import NativeScalerWithGradNormCount as NativeScaler
import measure
from models.utils import adjust_learning_rate
from prefetch_generator import BackgroundGenerator

def fuse_weight_cal(opt):
    device = opt.device
    print(device)

    model = Model.SDCC_Model()
    model_name =  opt.class_name +'_SDCC4.pth'

    if opt.use_gpu:
        model.to(device)
        model.load_state_dict(torch.load(opt.load_model_path + model_name, map_location=device))
    print("load weights!")

    validateDataset = TrainData(opt=opt)
    validate_dataloader = DataLoader(validateDataset, batch_size=1, shuffle=True, drop_last=True, num_workers=0, pin_memory=True)
    dis_std_ave_scale1 = 0
    dis_mean_ave_scale1 = 0

    dis_std_ave_scale2 = 0
    dis_mean_ave_scale2 = 0

    dis_std_ave_scale3 = 0
    dis_mean_ave_scale3 = 0


    rec_std_ave_scale1 = 0
    rec_mean_ave_scale1 = 0

    rec_std_ave_scale2 = 0
    rec_mean_ave_scale2 = 0

    rec_std_ave_scale3 = 0
    rec_mean_ave_scale3 = 0

    model.eval()
    for index, item in enumerate(tqdm(validate_dataloader, ncols=80)):
        input_frame = item

        if opt.use_gpu:
            input_frame = input_frame.to(device, non_blocking=True)

        [dis_map1, dis_map2, dis_map3], [rec_map1, rec_map2, rec_map3] = model.a_map(input_frame)

        dis_map1 = gaussian_filter(dis_map1, sigma=4)
        dis_map2 = gaussian_filter(dis_map2, sigma=4)
        dis_map3 = gaussian_filter(dis_map3, sigma=4)

        rec_map1 = gaussian_filter(rec_map1, sigma=4)
        rec_map2 = gaussian_filter(rec_map2, sigma=4)
        rec_map3 = gaussian_filter(rec_map3, sigma=4)

        ############# dis ############
        dis_std_ave_scale1 = dis_std_ave_scale1 + np.std(dis_map1)
        dis_mean_ave_scale1 = dis_mean_ave_scale1 + np.mean(dis_map1)

        dis_std_ave_scale2 = dis_std_ave_scale2 + np.std(dis_map2)
        dis_mean_ave_scale2 = dis_mean_ave_scale2 + np.mean(dis_map2)

        dis_std_ave_scale3 = dis_std_ave_scale3 + np.std(dis_map3)
        dis_mean_ave_scale3 = dis_mean_ave_scale3 + np.mean(dis_map3)

        ############# rec ############
        rec_std_ave_scale1 = rec_std_ave_scale1 + np.std(rec_map1)
        rec_mean_ave_scale1 = rec_mean_ave_scale1 + np.mean(rec_map1)

        rec_std_ave_scale2 = rec_std_ave_scale2 + np.std(rec_map2)
        rec_mean_ave_scale2 = rec_mean_ave_scale2 + np.mean(rec_map2)

        rec_std_ave_scale3 = rec_std_ave_scale3 + np.std(rec_map3)
        rec_mean_ave_scale3 = rec_mean_ave_scale3 + np.mean(rec_map3)


    dis_std_ave_scale1 = dis_std_ave_scale1 / validateDataset.__len__()
    dis_mean_ave_scale1 = dis_mean_ave_scale1 / validateDataset.__len__()

    dis_std_ave_scale2 = dis_std_ave_scale2 / validateDataset.__len__()
    dis_mean_ave_scale2 = dis_mean_ave_scale2 / validateDataset.__len__()

    dis_std_ave_scale3 = dis_std_ave_scale3 / validateDataset.__len__()
    dis_mean_ave_scale3 = dis_mean_ave_scale3 / validateDataset.__len__()


    rec_std_ave_scale1 = rec_std_ave_scale1 / validateDataset.__len__()
    rec_mean_ave_scale1 = rec_mean_ave_scale1 / validateDataset.__len__()

    rec_std_ave_scale2 = rec_std_ave_scale2 / validateDataset.__len__()
    rec_mean_ave_scale2 = rec_mean_ave_scale2 / validateDataset.__len__()

    rec_std_ave_scale3 = rec_std_ave_scale3 / validateDataset.__len__()
    rec_mean_ave_scale3 = rec_mean_ave_scale3 / validateDataset.__len__()



    fuse_weight = {
    'scale1':[dis_std_ave_scale1, dis_mean_ave_scale1, rec_std_ave_scale1, rec_mean_ave_scale1],
    'scale2':[dis_std_ave_scale2, dis_mean_ave_scale2, rec_std_ave_scale2, rec_mean_ave_scale2],
    'scale3':[dis_std_ave_scale3, dis_mean_ave_scale3, rec_std_ave_scale3, rec_mean_ave_scale3]
    }

    print(fuse_weight)
    torch.save(fuse_weight, opt.load_model_path + opt.class_name + 'fuse_weight.pth')



opt = DefaultConfig()
LOCO = ['breakfast_box', 'splicing_connectors', 'screw_bag', 'juice_bottle', 'pushpins']
opt.parse({'class_name': 'pushpins'})
fuse_weight_cal(opt)







