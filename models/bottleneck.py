# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
from functools import partial
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Block
from models.utils import get_2d_sincos_pos_embed


class DPTV(nn.Module):
    def __init__(self, img_size=[28, 14, 7], patch_size=[4, 2, 1], in_chans=[512, 1024, 2048],
                 embed_dim=480, output_dim=2048, depth=6, num_heads=12,
                 decoder_embed_dim=480, decoder_depth=6, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.in_chans = in_chans
        self.decoder_embed_dim = decoder_embed_dim

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed1 = PatchEmbed(img_size[0], patch_size[0], in_chans[0], embed_dim)
        self.patch_embed2 = PatchEmbed(img_size[1], patch_size[1], in_chans[1], embed_dim)
        self.patch_embed3 = PatchEmbed(img_size[2], patch_size[2], in_chans[2], embed_dim)

        num_patches = self.patch_embed1.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.auxiliary_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.latent_auxiliary_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding

        self.global_decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])
        
        self.local_decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, output_dim, bias=True)

        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed1.num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(self.patch_embed1.num_patches ** .5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)

        w = self.patch_embed1.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        w = self.patch_embed2.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        w = self.patch_embed3.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, x):
        input_x = x
        ################################ multi-scale embedding ##################################
        x1 = self.patch_embed1(x[0])
        x2 = self.patch_embed2(x[1])
        x3 = self.patch_embed3(x[2])
        x = x1+x2+x3
        x = x + self.pos_embed[:, 1:, :]
        N, L, D = x.shape  # batch, length, dim

        ############################################ v1 #########################################
        #
        # ########### sematic token cat #############
        # auxiliary_token = self.auxiliary_token.repeat(N, L, 1)
        # auxiliary_token = auxiliary_token + self.pos_embed[:, 1:, :]
        # x_ = torch.cat([x, auxiliary_token], dim=1)
        #
        # ############### encoder ################
        # for encoder_phase, blk in enumerate(self.blocks):
        #     x_ = blk(x_)+x_
        # x_ = self.norm(x_)
        # x_global_latent = x_[:, L:, :]
        # x_local_latent = x_[:, :L, :]
        #
        # ########## encoder-decoder ###########
        # x_global_latent = self.decoder_embed(x_global_latent)
        # x_local_latent = self.decoder_embed(x_local_latent)
        # ########## decoder ###########
        # for gl_blk, lo_blk in zip(self.global_decoder_blocks, self.local_decoder_blocks):
        #     x_global_latent = gl_blk(x_global_latent)+x_global_latent
        #     x_local_latent = lo_blk(x_local_latent)+x_local_latent
        #
        # x_global_latent = self.norm(x_global_latent)
        # x_global_output = self.decoder_pred(x_global_latent)
        #
        # x_local_latent = self.norm(x_local_latent)
        # x_local_output = self.decoder_pred(x_local_latent)

        ############################################ v2 #########################################

        # ########### sematic token cat #############
        # auxiliary_token = self.auxiliary_token.repeat(N, L, 1)
        # auxiliary_token = auxiliary_token + self.pos_embed[:, 1:, :]
        # x_ = torch.cat([x, auxiliary_token], dim=1)
        #
        # ############### encoder ################
        # for encoder_phase, blk in enumerate(self.blocks):
        #     x_ = blk(x_)+x_
        # x_ = self.norm(x_)
        # x_global_latent = x_[:, L:, :]
        # x_local_latent = x_[:, :L, :]
        #
        # ########## encoder-decoder ###########
        # x_global_latent = self.decoder_embed(x_global_latent)
        # x_local_latent = self.decoder_embed(x_local_latent)
        #
        # # ######### latent-pos-embedding ###########
        # latent_auxiliary_token = self.latent_auxiliary_token.repeat(N, L, 1)
        # latent_auxiliary_token = latent_auxiliary_token + self.pos_embed[:, 1:, :]
        #
        # ########### decoder global #############
        # x_global_latent = torch.cat([x_global_latent, latent_auxiliary_token], dim=1)
        # for decoder_phase, blk in enumerate(self.global_decoder_blocks):
        #     x_global_latent = blk(x_global_latent)+x_global_latent
        #
        # x_global_latent = self.norm(x_global_latent)
        # x_global_output = self.decoder_pred(x_global_latent)
        # x_global_output = x_global_output[:, -L:, :]
        #
        # ########### decoder local #############
        # for decoder_phase, blk in enumerate(self.local_decoder_blocks):
        #     x_local_latent = blk(x_local_latent)+x_local_latent
        #
        # x_local_latent = self.norm(x_local_latent)
        # x_local_output = self.decoder_pred(x_local_latent)

        ############################################ v3 #########################################

        ########## sematic token cat #############
        auxiliary_token = self.auxiliary_token.repeat(N, 1, 1)
        x_ = torch.cat([x, auxiliary_token], dim=1)

        ############### encoder ################
        for encoder_phase, blk in enumerate(self.blocks):
            x_ = blk(x_)+x_
        x_ = self.norm(x_)
        x_global_latent = x_[:, L:, :]
        x_local_latent = x_[:, :L, :]

        ########## encoder-decoder ###########
        x_global_latent = self.decoder_embed(x_global_latent)
        x_local_latent = self.decoder_embed(x_local_latent)

        # ######### latent-pos-embedding ###########
        latent_auxiliary_token = self.latent_auxiliary_token.repeat(N, L, 1)
        latent_auxiliary_token = latent_auxiliary_token + self.pos_embed[:, 1:, :]

        ########### decoder global #############
        x_global_latent = torch.cat([x_global_latent, latent_auxiliary_token], dim=1)
        for decoder_phase, blk in enumerate(self.global_decoder_blocks):
            x_global_latent = blk(x_global_latent)+x_global_latent

        x_global_latent = self.decoder_norm(x_global_latent)
        x_global_output = self.decoder_pred(x_global_latent)
        x_global_output = x_global_output[:, -L:, :]

        ########### decoder local #############
        for decoder_phase, blk in enumerate(self.local_decoder_blocks):
            x_local_latent = blk(x_local_latent)+x_local_latent

        x_local_latent = self.decoder_norm(x_local_latent)
        x_local_output = self.decoder_pred(x_local_latent)

        return self.unpatchify(x_local_output).contiguous(), self.unpatchify(x_global_output).contiguous()


        ##########################################w/o sb #####################
        # return input_x[2], input_x[2]




    def forward(self, imgs):
        pred = self.forward_encoder(imgs)

        return pred

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = 1
        h = w = int(x.shape[1] ** .5)
        c = int(x.shape[-1])
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs


if __name__ == '__main__':
    import time
    import torch

    model = DPTV().cuda()
    input_tensor = [torch.rand(1, 512, 28, 28).cuda(), torch.rand(1, 1024, 14, 14).cuda(), torch.rand(1, 2048, 7, 7).cuda()]
    for i in range(10):
        t1 = time.time()
        output = model.forward(input_tensor)
        t2 = time.time()
        print(t2 - t1)
        print(output[0].shape)
