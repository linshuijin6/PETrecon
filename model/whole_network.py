import numpy as np
import torch
from torch import nn
from geometry.BuildGeometry_v4 import BuildGeometry_v4
from model.network_swinTrans import SwinIR
from recon_astraFBP import sino2pic as s2p
from utils.transfer_si import i2s, s2i, s2i_batch
from modelSwinUnet.SUNet import SUNet_model


def otsu_threshold_batch(img_batch):
    # Step 1: 计算每个图像的直方图（在灰度值范围0-255）
    batch_size, channels, height, width = img_batch.shape
    hist = torch.stack([torch.histc(img_batch[i], bins=256, min=0, max=255) for i in range(batch_size)], dim=0)  # shape: (batch_size, 256)
    hist = hist / hist.sum(dim=1, keepdim=True)  # 归一化直方图，每个图像的灰度分布

    # Step 2: 计算累积和和累积均值
    cumsum_hist = torch.cumsum(hist, dim=1)  # 累积和，shape: (batch_size, 256)
    cumsum_mean = torch.cumsum(hist * torch.arange(256, device=img_batch.device), dim=1)  # 累积均值，shape: (batch_size, 256)
    global_mean = cumsum_mean[:, -1]  # 全局均值，shape: (batch_size,)

    # Step 3: 计算类间方差
    numerator = (global_mean.unsqueeze(1) * cumsum_hist - cumsum_mean) ** 2
    denominator = cumsum_hist * (1 - cumsum_hist)
    between_class_variance = numerator / (denominator + 1e-6)  # 避免除零

    # Step 4: 获取最大方差对应的阈值
    _, optimal_thresholds = torch.max(between_class_variance, dim=1)  # shape: (batch_size,)

    # Step 5: 根据最优阈值生成掩膜
    optimal_thresholds = optimal_thresholds.view(batch_size, 1, 1, 1).expand(-1, channels, height, width)  # 调整阈值形状
    mask_batch = (img_batch >= optimal_thresholds).float()  # 将掩膜转换为0和1的浮点型结果

    return mask_batch


def normalization2one(input_tensor):
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
    def __init__(self, radon, device, config, num_block=3):
        super().__init__()
        self.num_block = num_block
        self.radon = radon
        # self.n_theta = 2 * self.geo[0].shape[0]
        self.device = device
        # self.PET = PET
        # self.norm1 = nn.BatchNorm2d(1)
        # self.norm2 = nn.BatchNorm2d(1)
        # self.norm3 = nn.BatchNorm2d(1)
        # self.denoiseBlock1 = SwinIR(img_size, depths=[3, 3], num_heads=[4, 4]).to(device)
        self.denoiseBlock1 = SUNet_model(config).to(self.device)
        self.denoiseBlock2 = SUNet_model(config).to(self.device)
        # self.denoiseBlock3 = SUNet_model(config).to(self.device)
        # self.denoiseBlock2 = SwinIR(img_size, depths=[3, 3], num_heads=[4, 4]).to(device)
        # self.denoiseBlock3 = SwinIR(img_size, depths=[3, 3], num_heads=[4, 4]).to(device)

    def forward(self, image_p, sino_o, mask):
        image = normalization2one(image_p)
        image = self.denoiseBlock1(image)
        image = normalization2one(image)
        image = self.DCLayer(image, mask, sino_o)
        image = normalization2one(image)
        image = self.denoiseBlock2(image)
        image = normalization2one(image)
        # image = self.DCLayer(image, mask, sino_o)
        # image = normalization2one(image)
        # image = self.denoiseBlock3(image)
        # image = normalization2one(image)
        # image = self.DCLayer(image, mask, sino_o)
        return image

    def DCLayer(self, x_p, mask, sino_o):
        sino_re = self.radon.forward(x_p)
        sino_re = sino_re.to(self.device)
        # sino_re = sino_re[:, None, :, :]
        out_sino = normalization2one(sino_o) * (1 - mask) + normalization2one(sino_re) * mask

        out_pic = self.radon.filter_backprojection(out_sino)
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
        return out_pic


class PETDenoiseNet(nn.Module):
    def __init__(self, device, num_block=3):
        super().__init__()
        self.num_block = num_block
        self.denoiseBlock1 = SwinIR(img_size=168, embed_dim=32, depths=[3, 3], num_heads=[4, 4], window_size=4, mlp_ratio=2).to(device)
        # self.denoiseBlock2 = SwinIR(img_size=168)
        # self.denoiseBlock3 = SwinIR(img_size=168)

    def forward(self, image_p):
        image_p = normalization2one(image_p)
        # mask_tem = otsu_threshold_batch(255*image_p)
        image = self.denoiseBlock1(image_p)
        image = normalization2one(image)
        # image = mask_tem * image
        return image











