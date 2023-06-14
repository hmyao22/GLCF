import os
import numpy as np
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import cv2
import glob
from config import DefaultConfig
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from datasets import feature_create_disjoint_masks
from scipy.ndimage import gaussian_filter
from datasets import TrainData
from datasets import transform_x
from PIL import Image
import matplotlib.pyplot as plt
from datasets import denormalize
import pandas as pd
from skimage import measure
from numpy import ndarray
from statistics import mean
from sklearn.metrics import auc
import models.Model as Model
from get_spro import Get_SPRO_Fun
from sklearn.metrics import roc_curve, auc

def compute_pro(masks, amaps, num_th: int = 200) -> None:

    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        category (str): Category of product
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """

    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=np.bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        df = df.append({"pro": mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)

    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc


class MVTecLOCOTestDataset(Dataset):

    def __init__(self, root_dir, resize_shape=None):
        self.root_dir = root_dir
        self.images = sorted(glob.glob(root_dir + "/*/*.png"))
        print(self.images)
        self.resize_shape = resize_shape

    def __len__(self):
        return len(self.images)

    def transform_image(self, image_path, mask_path):
        x = Image.open(image_path).convert('RGB').resize((256, 256))
        image = transform_x(x)
        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            image0 = cv2.imread(image_path)
            mask = np.zeros((image0.shape[0], image0.shape[1]))
        if self.resize_shape != None:
            mask = cv2.resize(mask, dsize=(self.resize_shape[1], self.resize_shape[0]))

        mask = mask / 255.0
        mask = np.array(mask).reshape((mask.shape[0], mask.shape[1], 1)).astype(np.float32)
        mask = np.transpose(mask, (2, 0, 1))
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
            # mask_file_name = file_name.split(".")[0]+'/'+"000.png"
            # mask_path = os.path.join(mask_path, mask_file_name)
            # image, mask = self.transform_image(img_path, mask_path)
            mask_ave = np.zeros((1, self.resize_shape[1], self.resize_shape[0]))
            for mask_image_path in os.listdir(os.path.join(mask_path, file_name.split(".")[0])):
                mask_path_ = os.path.join(os.path.join(mask_path, file_name.split(".")[0]), mask_image_path)
                image, mask = self.transform_image(img_path, mask_path_)
                mask_ave += mask
                mask = mask_ave / len(os.listdir(os.path.join(mask_path, file_name.split(".")[0])))

            has_anomaly = np.array([1], dtype=np.float32)
        sample = {'image': image, 'has_anomaly': has_anomaly, 'mask': mask, 'idx': idx}

        return sample



class MVTecTestDataset(Dataset):

    def __init__(self, root_dir, resize_shape=None):
        self.root_dir = root_dir
        self.images = sorted(glob.glob(root_dir+"/*/*.png"))
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


        mask = mask / 255.0
        mask = np.array(mask).reshape((mask.shape[0], mask.shape[1], 1)).astype(np.float32)
        mask = np.transpose(mask, (2, 0, 1))
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
            mask_path = os.path.join(mask_path, mask_file_name)
            image, mask = self.transform_image(img_path, mask_path)
            has_anomaly = np.array([1], dtype=np.float32)
        sample = {'image': image, 'has_anomaly': has_anomaly,'mask': mask, 'idx': idx}

        return sample


def test(opt, obj_names, mvtec_path, checkpoint_path, training_model=None, is_save=False):
    obj_ap_pixel_list = []
    obj_auroc_pixel_list = []
    obj_ap_image_list = []
    obj_auroc_image_list = []
    obj_pro_pixel_list=[]


    for obj_name in obj_names:
        img_dim = 224
        device = opt.device
        print(device)
        model = Model.SDCC_Model()
        model_name = obj_name + '_' + 'SDCC2.pth'

        if training_model is None:
            if obj_name in ['breakfast_box', 'splicing_connectors', 'screw_bag', 'juice_bottle', 'pushpins']:
                model.load_state_dict(torch.load(checkpoint_path +'loco/v2/' +model_name, map_location=device))
            else:
                model.load_state_dict(torch.load(checkpoint_path + 'mvtec/' + model_name, map_location=device))
            fuse_weight = torch.load(opt.load_model_path +'fuse_weight/'+ obj_name + 'fuse_weight.pth')
            # if obj_name in ['breakfast_box', 'splicing_connectors', 'screw_bag', 'juice_bottle', 'pushpins']:
            #     model.load_state_dict(torch.load(checkpoint_path + model_name, map_location=device))
            # fuse_weight = torch.load(opt.load_model_path + obj_name + 'fuse_weight.pth')
        else:
            model = training_model

        if opt.use_gpu:
            model = model.to(device)
            model = model.eval()

        dataset = MVTecTestDataset(mvtec_path + "/" + obj_name + "/test", resize_shape=[img_dim, img_dim])
        dataset = MVTecLOCOTestDataset(mvtec_path + "/" + obj_name + "/test", resize_shape=[img_dim, img_dim])
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
        total_pixel_scores = np.zeros((img_dim * img_dim * len(dataset)))
        total_gt_pixel_scores = np.zeros((img_dim * img_dim * len(dataset)))
        mask_cnt = 0

        anomaly_score_gt = []
        anomaly_score_prediction = []

        anomaly_map_gt = []
        anomaly_map_prediction = []

        for i_batch, sample_batched in enumerate(tqdm(dataloader, ncols=80)):
            gray_batch = sample_batched["image"].to(device, non_blocking=True)

            is_normal = sample_batched["has_anomaly"].detach().numpy()[0, 0]
            anomaly_score_gt.append(is_normal)
            true_mask = sample_batched["mask"]
            true_mask_cv = true_mask.detach().numpy()[0, :, :, :].transpose((1, 2, 0))

            [dis_map1, dis_map2, dis_map3], [rec_map1, rec_map2, rec_map3] = model.a_map(gray_batch)

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

            dis_amap = dis_map1*1 + dis_map2 * 3 + dis_map3 * 6
            rec_map = rec_map1*1 + rec_map2 * 3 + rec_map3 * 6

            if obj_name in ['grid', 'carpet', 'leather', 'tile', 'wood']:
                amap = dis_amap * 1 + rec_map * 0
            else:
                amap = dis_amap *0 + rec_map * 1


            #
            # dis_amap = dis_map1 + dis_map2 * 10 + dis_map3 * 30
            # rec_map = rec_map1 + rec_map2 * 15 + rec_map3 * 30
            # #### logical/struc ###
            # amap = dis_amap * 8 + rec_map * 1

            if is_save:
                input_frame = denormalize(gray_batch.clone().squeeze(0).cpu().detach().numpy())
                cv2_input = np.array(input_frame, dtype=np.uint8)
                plt.figure()
                plt.subplot(131)
                plt.imshow(cv2_input)
                plt.subplot(132)
                plt.imshow(amap, cmap='jet')
                plt.axis('off')
                plt.title(amap.std())
                plt.subplot(133)
                plt.imshow(true_mask_cv, cmap='jet')
                plt.axis('off')
                plt.savefig(opt.measure_save_path + obj_name + str(i_batch) + '.png')
                plt.close()
            else:
                pass

            ############## image score #################s
            """std"""
            out_mask_cv = amap
            image_score = np.std(amap)

            anomaly_score_prediction.append(image_score)
            #
            true_mask_cv[true_mask_cv > 0.5] = 1
            true_mask_cv[true_mask_cv <= 0.5] = 0
            anomaly_map_gt.append(np.squeeze(true_mask_cv))
            anomaly_map_prediction.append(np.squeeze(out_mask_cv))

            flat_true_mask = true_mask_cv.flatten()
            flat_out_mask = out_mask_cv.flatten()
            total_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_out_mask
            total_gt_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_true_mask
            mask_cnt += 1


        #########################PRO ################################
        # anomaly_map_gt = np.array(anomaly_map_gt)
        # anomaly_map_prediction = np.array(anomaly_map_prediction)
        # aupro_pixel = compute_pro(anomaly_map_gt, anomaly_map_prediction)

        ##########################sPRO################################
        #
        # anomaly_map_prediction = np.array(anomaly_map_prediction)
        # Get_SPRO_Fun(anomaly_map_prediction, obj_name)

        ##########################AUC###############################

        anomaly_score_prediction = np.array(anomaly_score_prediction)
        anomaly_score_gt = np.array(anomaly_score_gt)

        auroc = roc_auc_score(anomaly_score_gt, anomaly_score_prediction)
        ap = average_precision_score(anomaly_score_gt, anomaly_score_prediction)

        ##########################################################

        total_gt_pixel_scores = total_gt_pixel_scores.astype(np.uint8)
        total_gt_pixel_scores = total_gt_pixel_scores[:img_dim * img_dim * mask_cnt]
        total_pixel_scores = total_pixel_scores[:img_dim * img_dim * mask_cnt]

        auroc_pixel = roc_auc_score(total_gt_pixel_scores, total_pixel_scores)
        ap_pixel = average_precision_score(total_gt_pixel_scores, total_pixel_scores)
        ##########################################################

        obj_ap_pixel_list.append(ap_pixel)
        obj_auroc_pixel_list.append(auroc_pixel)
        obj_auroc_image_list.append(auroc)
        obj_ap_image_list.append(ap)
        #obj_pro_pixel_list.append(aupro_pixel)
        print(obj_name)
        print("AUC Image:  " + str(auroc))
        print("AP Image:  " + str(ap))
        print("AUC Pixel:  " + str(auroc_pixel))
        print("AP Pixel:  " + str(ap_pixel))
        #print("PRO Pixel:  " + str(aupro_pixel))
        print("==============================")
    print("AUC Image mean:  " + str(np.mean(obj_auroc_image_list)))
    print("AP Image mean:  " + str(np.mean(obj_ap_image_list)))
    print("AUC Pixel mean:  " + str(np.mean(obj_auroc_pixel_list)))
    print("AP Pixel mean:  " + str(np.mean(obj_ap_pixel_list)))
    #print("PRO Pixel mean:  " + str(np.mean(obj_pro_pixel_list)))

    return np.mean(obj_auroc_pixel_list) + np.mean(obj_auroc_image_list)


if __name__ == "__main__":
    from config import DefaultConfig

    opt = DefaultConfig()
    LOCO = ['splicing_connectors', 'breakfast_box', 'screw_bag', 'juice_bottle', 'pushpins']
    test_list = [
    'screw',
    ]
    MVTEC=['splicing_connectors']

    with torch.cuda.device(0):
            test(opt, MVTEC, mvtec_path=opt.data_root, checkpoint_path=opt.load_model_path, is_save=True)
