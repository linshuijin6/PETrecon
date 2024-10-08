import numpy as np
import torch
from torch import nn
from geometry.BuildGeometry_v4 import BuildGeometry_v4
from model.network_swinTrans import SwinIR
from recon_astraFBP import sino2pic as s2p
from utils.transfer_si import i2s, s2i, s2i_batch


def normlazation2one(input_tensor):
    # 假设输入的 tensor 形状为 (batchsize, channels=1, h, w)
    # input_tensor = torch.randn(4, 1, 64, 64)  # 示例的输入张量

    # 为了每个 batch 归一化，我们要按batch维度进行最小值和最大值的计算
    # 计算每个batch的最小值和最大值，保持维度为 (batchsize, 1, 1, 1)
    min_val = input_tensor.reshape(input_tensor.size(0), -1).min(dim=1)[0].reshape(-1, 1, 1, 1)
    max_val = input_tensor.reshape(input_tensor.size(0), -1).max(dim=1)[0].reshape(-1, 1, 1, 1)

    # 进行归一化，将所有数值归一化到 [0, 1] 区间
    normalized_tensor = (input_tensor - min_val) / (max_val - min_val + 1e-8)  # 1e-8 防止除以0
    assert input_tensor.shape == normalized_tensor.shape

    return normalized_tensor  # 确认输出形状 (batchsize, 1, h, w)


class PETReconNet(nn.Module):
    def __init__(self, geo, device, num_block=3, img_size=168):
        super().__init__()
        self.num_block = num_block
        self.geo = geo
        # self.PET = PET
        # self.norm1 = nn.BatchNorm2d(1)
        # self.norm2 = nn.BatchNorm2d(1)
        # self.norm3 = nn.BatchNorm2d(1)
        self.denoiseBlock1 = SwinIR(img_size, depths=[3, 3], num_heads=[4, 4]).to(device)
        self.denoiseBlock2 = SwinIR(img_size, depths=[3, 3], num_heads=[4, 4]).to(device)
        self.denoiseBlock3 = SwinIR(img_size, depths=[3, 3], num_heads=[4, 4]).to(device)

    def forward(self, image_p, sino_o, AN, mask):
        image = self.denoiseBlock1(image_p)
        image = normlazation2one(image)
        image = self.DCLayer(image, mask, sino_o, AN)
        image = normlazation2one(image)
        # image = self.denoiseBlock2(image)
        # image = normlazation2one(image)
        # image = self.DCLayer(image, mask, sino_o, AN)
        # image = self.denoiseBlock3(image)
        # image = normlazation2one(image)
        # image = self.DCLayer(image, mask, sino_o, AN)
        return image

    def DCLayer(self, x_p, mask, sino_o, AN):
        sino_re = i2s(x_p, AN, geoMatrix=self.geo, sinogram_nAngular=360)
        sino_re = sino_re.to(self.device)
        sino_re = sino_re[None, None, :, :]
        out_sino = sino_o*(1-mask) + sino_re*mask
        out_sino = s2i_batch(out_sino, device_now=self.device)
        # if out_sino.shape[0] == 1:
        #     out_sino = s2i(out_sino)
        #     # out_sino = out_sino[None, None, :, :]
        #     out_sino = out_sino.unsqueeze([0, 1])
        #     return out_sino
        # else:
        #     sino_t = []
        #     for sino in out_sino:
        #         sino_t.append(s2i(sino))
        #     out_sino = torch.stack(sino_t, 0)
        #     out_sino = out_sino.unsqueeze([0, 1])
        return out_sino


class PETDenoiseNet(nn.Module):
    def __init__(self, device, num_block=3):
        super().__init__()
        self.num_block = num_block
        self.denoiseBlock1 = SwinIR(img_size=168, embed_dim=32, depths=[3, 3], num_heads=[4, 4], window_size=4, mlp_ratio=2).to(device)
        # self.denoiseBlock2 = SwinIR(img_size=168)
        # self.denoiseBlock3 = SwinIR(img_size=168)

    def forward(self, image_p):
        image = self.denoiseBlock1(image_p)
        image = normlazation2one(image)
        # image = self.denoiseBlock2(image)
        # image = normlazation2one(image)
        # image = self.denoiseBlock3(image)
        # image = normlazation2one(image)
        return image











