import torch.optim as opti
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn import init
import os
from datasets import TrainData, TestData, UniTrainData
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


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_normal(m.weight)


def train(opt, show_feature_map=True):
    loss_scaler = NativeScaler()
    device = opt.device
    print(device)

    model = Model.SDCC_Model()
    model_name = opt.class_name + '_' + 'SDCC4.pth'

    if opt.use_gpu:
        model.to(device)
    if os.path.exists(opt.load_model_path + model_name):
        model.load_state_dict(torch.load(opt.load_model_path + model_name))
        print("load weights!")


    optimizer = opti.AdamW(model.parameters(), lr=opt.lr,betas=(0.9, 0.95))

    trainDataset = TrainData(opt=opt)
    train_dataloader = DataLoader(trainDataset, batch_size=opt.train_batch_size, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)


    testDataset = TestData(opt=opt)
    test_dataloader = DataLoaderX(testDataset, batch_size=1, shuffle=True, drop_last=True, num_workers=0)


    max_auc =0.0
    for epoch in range(opt.max_epoch):
        adjust_learning_rate(optimizer, epoch)
        running_loss = 0.0

        for index, item in enumerate(tqdm(train_dataloader, ncols=80)):
            input_frame = item

            if opt.use_gpu:
                input_frame = input_frame.to(device, non_blocking=True)

            loss =model.loss(input_frame)

            loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=(index + 1) % 1 == 0)
            running_loss += loss.item()


            if index == len(train_dataloader)-1:
                print(f"[{epoch}]  F_loss: {(running_loss / (1 * len(trainDataset))):.3f}")


        if epoch % 1 == 0:
            if epoch == 0:
                model_dict = model.state_dict()
                torch.save(model_dict, opt.load_model_path + model_name)
            model.eval()
            item = next(iter(test_dataloader))
            input_frame = item

            if opt.use_gpu:
                input_frame = input_frame.to(device, non_blocking=True)

            [local_map1, local_map2, local_map3], [global_map1, global_map2, global_map3]= model.a_map(input_frame)

            local_map1 = gaussian_filter(local_map1, sigma=4)
            local_map2 = gaussian_filter(local_map2, sigma=4)
            local_map3 = gaussian_filter(local_map3, sigma=4)


            global_map1 = gaussian_filter(global_map1, sigma=4)
            global_map2 = gaussian_filter(global_map2, sigma=4)
            global_map3 = gaussian_filter(global_map3, sigma=4)

            input_frame1 = denormalize(input_frame.clone().squeeze(0).cpu().detach().numpy())
            cv2_input = np.array(input_frame1, dtype=np.uint8)

            plt.figure()
            plt.subplot(2, 4, 1)
            plt.axis('off')
            plt.imshow(cv2_input)
            plt.subplot(2, 4, 2)
            plt.axis('off')
            plt.imshow(local_map1, cmap='jet')
            plt.subplot(2, 4, 3)
            plt.axis('off')
            plt.imshow(local_map2, cmap='jet')
            plt.subplot(2, 4, 4)
            plt.axis('off')
            plt.imshow(local_map3, cmap='jet')

            plt.subplot(2, 4, 5)
            plt.axis('off')
            plt.imshow(cv2_input)
            plt.subplot(2, 4, 6)
            plt.axis('off')
            plt.imshow(global_map1, cmap='jet')
            plt.subplot(2, 4, 7)
            plt.axis('off')
            plt.imshow(global_map2, cmap='jet')
            plt.subplot(2, 4, 8)
            plt.axis('off')
            plt.imshow(global_map3, cmap='jet')

            plt.savefig("temp.png")
            plt.close()

            if show_feature_map == True:
                deep_feature, local_feature, global_feature, global_feature_reg  = model(input_frame)
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
                
                

                plt.savefig('feature/' + str(epoch) + opt.is_STVT + '.png')
                plt.close()


            # if epoch % 10 == 0:
            #     with torch.cuda.device(device):
            #         obj_list=[opt.class_name]
            #         auc = measure.test(opt, obj_list, mvtec_path=opt.data_root, checkpoint_path=opt.load_model_path, training_model=model)
            #         if auc > max_auc:
            #             print('weight updata!')
            #             max_auc = auc
            #             model_dict = model.state_dict()
            #             torch.save(model_dict, opt.load_model_path + model_name)
            model_dict = model.state_dict()
            torch.save(model_dict, opt.load_model_path + model_name)
            model = model.train(True)



LOCO = ['breakfast_box', 'splicing_connectors', 'screw_bag', 'juice_bottle', 'pushpins']
total_list_0 = ['pushpins']

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
    'pill',
    'screw',
    'zipper',
    'transistor',
]


cifar = [
        'airplane',
        'automobile',
        'bird',
        'cat',
        'deer',
        'dog',
        'frog',
        'horse',
        'ship',
        'truck',
                 ]

MNIST = [
        '0',
        '1',
        '2',
        '3',
        '4',
        '5',
        '6',
        '7',
        '8',
        '9',
        ]


video = ['ped2', 'avenue']
medical = ['OCT', 'brains']
opt = DefaultConfig()


if __name__ == '__main__':
    for obj in total_list_0:
        opt.parse({'class_name': obj})
        print('training_dataset:' + str(opt.class_name))
        train(opt)


# opt.parse({'is_STVT': is_STVT[1]})
# opt.parse({'backbone_name': backbone_name[0]})
# for obj in total_list:
#     opt.parse({'class_name': obj})
#     print('training_dataset:' + str(opt.class_name))
#     train(opt)
#
