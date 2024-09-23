import numpy as np
import torch
from torch import nn
from geometry.BuildGeometry_v4 import BuildGeometry_v4
from model.network_swinTrans import SwinIR
from recon_astraFBP import sino2pic as s2p


def normlazation2one(input_tensor):
    # 假设输入的 tensor 形状为 (batchsize, channels=1, h, w)
    # input_tensor = torch.randn(4, 1, 64, 64)  # 示例的输入张量

    # 为了每个 batch 归一化，我们要按batch维度进行最小值和最大值的计算
    # 计算每个batch的最小值和最大值，保持维度为 (batchsize, 1, 1, 1)
    min_val = input_tensor.view(input_tensor.size(0), -1).min(dim=1)[0].view(-1, 1, 1, 1)
    max_val = input_tensor.view(input_tensor.size(0), -1).max(dim=1)[0].view(-1, 1, 1, 1)

    # 进行归一化，将所有数值归一化到 [0, 1] 区间
    normalized_tensor = (input_tensor - min_val) / (max_val - min_val + 1e-8)  # 1e-8 防止除以0
    assert input_tensor.shape == normalized_tensor.shape

    return normalized_tensor  # 确认输出形状 (batchsize, 1, h, w)


class PETReconNet(nn.Module):
    def __init__(self, PET, device, num_block=3):
        super().__init__()
        self.num_block = num_block
        self.device = device
        self.PET = PET
        # self.norm1 = nn.BatchNorm2d(1)
        # self.norm2 = nn.BatchNorm2d(1)
        # self.norm3 = nn.BatchNorm2d(1)
        self.denoiseBlock1 = SwinIR(img_size=168)
        self.denoiseBlock2 = SwinIR(img_size=168)
        self.denoiseBlock3 = SwinIR(img_size=168)

    def forward(self, image_p, sino_o, AN, mask):
        image = self.denoiseBlock1(image_p)
        # image = self.norm1(image)
        image = normlazation2one(image)
        image = self.DCLayer(image, mask, sino_o, AN)
        image = self.denoiseBlock2(image)
        image = normlazation2one(image)
        # image = self.norm2(image)
        image = self.DCLayer(image, mask, sino_o, AN)
        image = self.denoiseBlock3(image)
        image = normlazation2one(image)
        # image = self.norm3(image)
        image = self.DCLayer(image, mask, sino_o, AN)
        return image

    def DCLayer(self, x_p, mask, sino_o, AN):
        sino_re, _, _, _ = self.PET.simulateSinogramData(x_p, AN)
        sino_re = torch.from_numpy(sino_re).to(self.device)
        sino_re = sino_re[None, None, :, :]
        out_sino = sino_o*(1-mask) + sino_re*mask
        if out_sino.shape[0] == 1:
            out_sino = s2p(out_sino)
            out_sino = out_sino[None, None, :, :]
            return out_sino.to(self.device)
        else:
            sino_t = []
            for sino in out_sino:
                sino_t.append(s2p(sino))
            out_sino = torch.from_numpy(np.array(sino_t)).to(self.device)
            out_sino = out_sino[None, None, :, :]
            return out_sino.to(self.device)











