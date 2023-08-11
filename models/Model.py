import models.swim_transformer_decoder as decoder
import models.swim_transformer_encoder as encoder
from torch.nn import functional as F
import models.cnn_encoder as cnn_encoder
import models.cnn_decoder as cnn_decoder
import models.bottleneck as bottle
from torch import nn
import torch
import time
import numpy as np
class SDCC_Model(nn.Module):
    def __init__(self):
        super(SDCC_Model, self).__init__()
        self.teacher_encoder,_ = cnn_encoder.wide_resnet50_2(pretrained=True)
        self.student_decoder = cnn_decoder.de_wide_resnet50_2(pretrained=False)
        self.student_decoder_reg_local = cnn_decoder.de_wide_resnet50_2(pretrained=False)
        self.student_decoder_reg_global = cnn_decoder.de_wide_resnet50_2(pretrained=False)
        self.bottle = bottle.DPTV()
        # path = r'D:/IMSN-YHM/cycle/weights/swin_tiny_patch4_window7_224.pth'
        # ckpt = torch.load(path)
        # msg = self.teacher_encoder.load_state_dict(ckpt['model'], strict=False)
        self.teacher_encoder.eval()

    def forward(self, imgs):
        ########## local branch ##########

        deep_feature = self.teacher_encoder(imgs)


        ########## global branch ##########
        #latent_feature = [latent_feature.permute(0, 3, 1, 2) for latent_feature in deep_feature[1:]]

        latent_feature = deep_feature[1:]

        local_latent, global_latent = self.bottle(latent_feature)
        global_feature = self.student_decoder(global_latent)
        global_feature_reg = self.student_decoder_reg_global(local_latent)
        local_feature = self.student_decoder_reg_local(local_latent)

        deep_feature = deep_feature[:3]

        return deep_feature, local_feature, global_feature, global_feature_reg

    def a_map(self, imgs):
        deep_feature, local_feature, global_feature, global_feature_reg = self(imgs)

        local_map1, local_map2, local_map3 = self.anomaly_map(deep_feature, local_feature)
        global_map1, global_map2, global_map3 = self.anomaly_map(global_feature, global_feature_reg)


        local_map = [local_map1, local_map2, local_map3]
        global_map = [global_map1, global_map2, global_map3]
        return local_map, global_map


    def anomaly_map(self, deep_feature, recon_feature, mod='cos'):
        batch_size = recon_feature[0].shape[0]
        if mod =='dis':
            dis_map1 = torch.mean((deep_feature[0] - recon_feature[0]) ** 2, dim=1)
            dis_map1 = dis_map1.reshape(batch_size, 1, 56, 56)
            dis_map1 = nn.functional.interpolate(dis_map1, size=(224, 224), mode="bilinear",
                                                 align_corners=True).squeeze(1)
            dis_map1 = dis_map1.clone().squeeze(0).cpu().detach().numpy()

            dis_map2 = torch.mean((deep_feature[1] - recon_feature[1]) ** 2, dim=1)
            dis_map2 = dis_map2.reshape(batch_size, 1, 28, 28)
            dis_map2 = nn.functional.interpolate(dis_map2, size=(224, 224), mode="bilinear",
                                                 align_corners=True).squeeze(1)
            dis_map2 = dis_map2.clone().squeeze(0).cpu().detach().numpy()

            dis_map3 = torch.mean((deep_feature[2] - recon_feature[2]) ** 2, dim=1)
            dis_map3 = dis_map3.reshape(batch_size, 1, 14, 14)
            dis_map3 = nn.functional.interpolate(dis_map3, size=(224, 224), mode="bilinear",
                                                 align_corners=True).squeeze(1)
            dis_map3 = dis_map3.clone().squeeze(0).cpu().detach().numpy()


            return dis_map1, dis_map2, dis_map3
        if mod == 'cos':
            dir_map_1 = 1 - F.cosine_similarity(deep_feature[0], recon_feature[0])
            dir_map_1 = dir_map_1.reshape(batch_size, 1, 56, 56)
            dir_map_1 = nn.functional.interpolate(dir_map_1, size=(224, 224), mode="bilinear",
                                                  align_corners=True).squeeze(1)
            dir_map_1 = dir_map_1.clone().squeeze(0).cpu().detach().numpy()

            dir_map_2 = 1 - F.cosine_similarity(deep_feature[1], recon_feature[1])
            dir_map_2 = dir_map_2.reshape(batch_size, 1, 28, 28)
            dir_map_2 = nn.functional.interpolate(dir_map_2, size=(224, 224), mode="bilinear",
                                                  align_corners=True).squeeze(1)
            dir_map_2 = dir_map_2.clone().squeeze(0).cpu().detach().numpy()

            dir_map_3 = 1 - F.cosine_similarity(deep_feature[2], recon_feature[2])
            dir_map_3 = dir_map_3.reshape(batch_size, 1, 14, 14)
            dir_map_3 = nn.functional.interpolate(dir_map_3, size=(224, 224), mode="bilinear",
                                                  align_corners=True).squeeze(1)
            dir_map_3 = dir_map_3.clone().squeeze(0).cpu().detach().numpy()



            return dir_map_1, dir_map_2, dir_map_3


    def loss(self, imgs):
        deep_feature, local_feature, global_feature, global_feature_reg = self(imgs)

        local_loss = self.loss_fucntion(deep_feature, local_feature)
        global_loss = self.loss_fucntion(global_feature, deep_feature)
        global_loss_reg = self.loss_fucntion(global_feature, global_feature_reg)
        loss = local_loss + 1*global_loss + 1*global_loss_reg

        return loss


    def loss_fucntion(self, a, b):
        cos_loss = torch.nn.CosineSimilarity()
        loss = 0
        for item in range(len(a)):
            loss += torch.mean(1 - cos_loss(a[item].view(a[item].shape[0], -1),
                                            b[item].view(b[item].shape[0], -1)))
        return loss


    def our_loss(self,a,b):
        dis_loss_1 = ((a[0] - b[0]) ** 2).mean(dim=1)
        dis_loss_2 = ((a[1] - b[1]) ** 2).mean(dim=1)
        dis_loss_3 = ((a[2] - b[2]) ** 2).mean(dim=1)
        dis_loss = dis_loss_1.mean() + dis_loss_2.mean() + dis_loss_3.mean()

        cos_loss_1 = 1 - torch.cosine_similarity(a[0], b[0],dim=1)
        cos_loss_2 = 1 - torch.cosine_similarity(a[1], b[1],dim=1)
        cos_loss_3 = 1 - torch.cosine_similarity(a[2], b[2],dim=1)
        cos_loss = cos_loss_1.mean() + cos_loss_2.mean() + cos_loss_3.mean()

        total_loss = dis_loss
        return total_loss



if __name__ == '__main__':
    import torch
    import time
    model = SDCC_Model().cuda()
    input_image = torch.rand(1, 3, 224, 224).cuda()
    for i in range(10):
        t1=time.time()
        deep_feature, local_feature,_,_ = model(input_image)
        t2 = time.time()
        print(t2-t1)

    from thop import profile
    from thop import clever_format
    with torch.no_grad():
        tensor = torch.rand(1, 3, 224, 224).cuda()
        flops, params = profile(model, inputs=(tensor,))
        flops, params = clever_format([flops, params], "%.3f")
        print(flops)
        print(params)


