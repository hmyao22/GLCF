import models.swim_transformer_decoder as decoder
import models.swim_transformer_encoder as encoder
import models.cnn_decoder as c_decoder
import models.cnn_encoder as c_encoder
import models.bottleneck as bottle
from torch import nn
import torch
import time
class SDCC_Model(nn.Module):
    def __init__(self):
        super(SDCC_Model, self).__init__()
        self.teacher_encoder = encoder.SwinTransformer()
        self.student_decoder = decoder.SwinTransformerDec()
        self.student_decoder_reg_local = decoder.SwinTransformerDec()
        self.student_decoder_reg_global = decoder.SwinTransformerDec()
        self.bottle = bottle.DPTV()
        path =  r'D:/IMSN-YHM/cycle/weights/swin_tiny_patch4_window7_224.pth'
        ckpt = torch.load(path)
        msg = self.teacher_encoder.load_state_dict(ckpt['model'], strict=False)
        self.teacher_encoder.eval()

    def forward(self, imgs):
        ########## local branch ##########
        with torch.no_grad():
            deep_feature = self.teacher_encoder(imgs)


        ########## global branch ##########
        latent_feature = [latent_feature.permute(0, 3, 1, 2) for latent_feature in deep_feature[1:]]

        # latent_feature[-1] = latent_feature[-1].permute(0, 2, 3, 1)
        # latent_feature[-1] = latent_feature[-1].view(latent_feature[-1].shape[0], 49, 768)
        # local_latent= latent_feature[-1]
        # global_latent = latent_feature[-1]


        local_latent, global_latent = self.bottle(latent_feature)

        global_feature = self.student_decoder(global_latent)
        global_feature_reg = self.student_decoder_reg_global(local_latent)
        local_feature = self.student_decoder_reg_local(local_latent)

        
        deep_feature = deep_feature[:3]
        local_feature = list(reversed(local_feature))[:3]
        global_feature = list(reversed(global_feature))[:3]
        global_feature_reg = list(reversed(global_feature_reg))[:3]
        

        return deep_feature, local_feature, global_feature, global_feature_reg

    def a_map(self, imgs):
        deep_feature, local_feature, global_feature, global_feature_reg = self(imgs)



        local_map1, local_map2, local_map3 = self.anomaly_map(deep_feature, local_feature)
        global_map1, global_map2, global_map3 = self.anomaly_map(global_feature, global_feature_reg)

        local_map = [local_map1, local_map2, local_map3]
        global_map = [global_map1, global_map2, global_map3]
        return local_map, global_map


    def anomaly_map(self, deep_feature, recon_feature, mod='dis'):
        batch_size = recon_feature[0].shape[0]
        if mod =='dis':
            dis_map1 = torch.mean((deep_feature[0] - recon_feature[0]) ** 2, dim=-1)
            dis_map1 = dis_map1.reshape(batch_size, 1, 56, 56)
            dis_map1 = nn.functional.interpolate(dis_map1, size=(224, 224), mode="bilinear",
                                                 align_corners=True).squeeze(1)
            dis_map1 = dis_map1.clone().squeeze(0).cpu().detach().numpy()

            dis_map2 = torch.mean((deep_feature[1] - recon_feature[1]) ** 2, dim=-1)
            dis_map2 = dis_map2.reshape(batch_size, 1, 28, 28)
            dis_map2 = nn.functional.interpolate(dis_map2, size=(224, 224), mode="bilinear",
                                                 align_corners=True).squeeze(1)
            dis_map2 = dis_map2.clone().squeeze(0).cpu().detach().numpy()

            dis_map3 = torch.mean((deep_feature[2] - recon_feature[2]) ** 2, dim=-1)
            dis_map3 = dis_map3.reshape(batch_size, 1, 14, 14)
            dis_map3 = nn.functional.interpolate(dis_map3, size=(224, 224), mode="bilinear",
                                                 align_corners=True).squeeze(1)
            dis_map3 = dis_map3.clone().squeeze(0).cpu().detach().numpy()


            return dis_map1, dis_map2, dis_map3
        if mod == 'cos':
            dir_map_1 = 1 - torch.nn.CosineSimilarity(-1)(deep_feature[0], recon_feature[0])
            dir_map_1 = dir_map_1.reshape(batch_size, 1, 56, 56)
            dir_map_1 = nn.functional.interpolate(dir_map_1, size=(224, 224), mode="bilinear",
                                                  align_corners=True).squeeze(1)
            dir_map_1 = dir_map_1.clone().squeeze(0).cpu().detach().numpy()

            dir_map_2 = 1 - torch.nn.CosineSimilarity(-1)(deep_feature[1], recon_feature[1])
            dir_map_2 = dir_map_2.reshape(batch_size, 1, 28, 28)
            dir_map_2 = nn.functional.interpolate(dir_map_2, size=(224, 224), mode="bilinear",
                                                  align_corners=True).squeeze(
                1)
            dir_map_2 = dir_map_2.clone().squeeze(0).cpu().detach().numpy()

            dir_map_3 = 1 - torch.nn.CosineSimilarity(-1)(deep_feature[2], recon_feature[2])
            dir_map_3 = dir_map_3.reshape(batch_size, 1, 14, 14)
            dir_map_3 = nn.functional.interpolate(dir_map_3, size=(224, 224), mode="bilinear",
                                                  align_corners=True).squeeze(
                1)
            dir_map_3 = dir_map_3.clone().squeeze(0).cpu().detach().numpy()



            return dir_map_1, dir_map_2, dir_map_3


    def loss(self, imgs):
        deep_feature, local_feature, global_feature, global_feature_reg = self(imgs)

        local_loss_1 = ((deep_feature[0] - local_feature[0]) ** 2).mean(dim=-1)
        local_loss_2 = ((deep_feature[1] - local_feature[1]) ** 2).mean(dim=-1)
        local_loss_3 = ((deep_feature[2] - local_feature[2]) ** 2).mean(dim=-1)

        global_loss_1 = ((global_feature[0] - deep_feature[0]) ** 2).mean(dim=-1)
        global_loss_2 = ((global_feature[1] - deep_feature[1]) ** 2).mean(dim=-1)
        global_loss_3 = ((global_feature[2] - deep_feature[2]) ** 2).mean(dim=-1)
        
        global_loss_reg1 = ((global_feature[0] - global_feature_reg[0]) ** 2).mean(dim=-1)
        global_loss_reg2 = ((global_feature[1] - global_feature_reg[1]) ** 2).mean(dim=-1)
        global_loss_reg3 = ((global_feature[2] - global_feature_reg[2]) ** 2).mean(dim=-1)



        loss = local_loss_1.mean() + local_loss_2.mean() + local_loss_3.mean() + \
               global_loss_1.mean() + global_loss_2.mean() + global_loss_3.mean() + \
               global_loss_reg1.mean() + global_loss_reg2.mean() + global_loss_reg3.mean()


        # dir_map_1 = 1 - torch.nn.CosineSimilarity(-1)(deep_feature[0], recon_feature[0])
        # dir_map_2 = 1 - torch.nn.CosineSimilarity(-1)(deep_feature[1], recon_feature[1])
        # dir_map_3 = 1 - torch.nn.CosineSimilarity(-1)(deep_feature[2], recon_feature[2])
        # dir_map_4 = 1 - torch.nn.CosineSimilarity(-1)(deep_feature[3], recon_feature[3])
        # loss = dir_map_1.mean() + dir_map_2.mean() + dir_map_3.mean() + dir_map_4.mean()

        return loss

if __name__ == '__main__':
    import torch
    import time
    model = SDCC_Model().cuda()
    input_image = torch.rand(1, 3, 224, 224).cuda()
    for i in range(10):
        t1=time.time()
        output = model.loss(input_image)
        t2 = time.time()
        print(t2-t1)

