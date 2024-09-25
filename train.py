import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DistributedSampler, DataLoader
import matplotlib.pyplot as plt
from geometry.BuildGeometry_v4 import BuildGeometry_v4
from utils.data import DatasetPETRecon, tv_loss
from utils.data import load_data, generate_mask
from recon_astraFBP import sino2pic as s2p
from model.whole_network import PETReconNet, PETDenoiseNet
from utils.transfer_si import s2i, i2s, s2i_batch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
from model.whole_network import normlazation2one
# from model.network_swinTrans import SwinIR

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def simulate_geometry(device):
    temPath = './tmp_1'
    PET = BuildGeometry_v4('mmr', device, 0.5)  # scanner mmr, with radial crop factor of 50%
    PET.loadSystemMatrix(temPath, is3d=False)
    return PET


def main(file_path):
    # # 数据
    # setup(rank, world_size)
    train_set = DatasetPETRecon(file_path, 'train')
    # train_sampler = DistributedSampler(train_set, shuffle=True, drop_last=True, seed=seed)
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True)

    # 模型初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    world_size = torch.cuda.device_count()  # 使用所有可用的 GPU
    # PET = simulate_geometry(device)

    denoise_model_pre = nn.DataParallel(PETDenoiseNet(device=device)).to(device)
    denoise_model = nn.DataParallel(PETReconNet(device=device)).to(
        device)  # x (4,1,96,96) (batch_size_in_each_GPU, input_image_channel, H, W)
    # denoise_model_pre = DDP(denoise_model_pre, device_ids=[rank])
    # denoise_model = DDP(denoise_model, device_ids=[rank])

    # print(torch.cuda.memory_summary())

    criterion = nn.MSELoss()
    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(denoise_model.parameters(), lr=0.01)

    # 训练
    num_epochs = 100
    for epoch in range(num_epochs):
        denoise_model.train()
        denoise_model_pre.train()
        running_loss = 0.0
        for inputs, _ in train_loader:
            x1, x2, AN = inputs
            x1, x2, AN = x1.to(device), x2.to(device), AN.to(device)

            # print(torch.cuda.memory_summary())

            # sinogram去噪，noise2noise训练
            x1_denoised = denoise_model_pre(x1)
            x2_denoised = denoise_model_pre(x2)
            # 平均输出的sinogram
            aver_x = (x1_denoised + x2_denoised) / 2.

            # PET图去噪

            mask_p1, mask_p2 = generate_mask(aver_x.shape, 0.01)
            mask_p1, mask_p2 = torch.from_numpy(mask_p1).unsqueeze(1).float().to(device), torch.from_numpy(mask_p2).unsqueeze(1).float().to(device)
            sino_p1, sino_p2 = aver_x * mask_p2, aver_x * mask_p1
            pic_in_p1, pic_in_p2 = s2i_batch(sino_p1), s2i_batch(sino_p2)
            pic_recon_p1, pic_recon_p2 = denoise_model(pic_in_p1, aver_x, AN, mask_p1), denoise_model(pic_in_p2, aver_x, AN, mask_p2)

            # 计算mask角度下的loss
            sino_recon_p1, sino_recon_p2 = i2s(pic_recon_p1, AN, sinogram_nAngular=360), i2s(pic_recon_p2, AN, sinogram_nAngular=360)
            sino_recon_p1 = sino_recon_p1[None, None, :, :] if len(sino_recon_p1.shape) == 2 else sino_recon_p1[:, None, :, :]
            sino_recon_p2 = sino_recon_p2[None, None, :, :] if len(sino_recon_p2.shape) == 2 else sino_recon_p2[:, None, :, :]
            sino_recon_p1_m2, sino_recon_p2_m1 = sino_recon_p1 * mask_p2, sino_recon_p2 * mask_p1
            sino_recon_p1_m2, sino_recon_p2_m1 = normlazation2one(sino_recon_p1_m2), normlazation2one(sino_recon_p2_m1)

            lsp1, lsp2 = criterion(sino_recon_p1_m2, sino_p2), criterion(sino_recon_p2_m1, sino_p1)
            lspre = criterion(x1_denoised, x2_denoised) + tv_loss(x1_denoised) + tv_loss(x2_denoised)
            li = criterion(pic_recon_p1, pic_recon_p2)
            loss = lspre + lsp1 + lsp2 + li

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')


if __name__ == '__main__':
    temPath = r'./tmp_1'
    PET = BuildGeometry_v4('mmr', 0)  # scanner mmr, with radial crop factor of 50%
    PET.loadSystemMatrix(temPath, is3d=False)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # 绕过 GPU 0，只使用 GPU 1 和 GPU 2
    world_size = torch.cuda.device_count()
    # path = './simulation_angular/angular_180'
    os.environ['MASTER_ADDR'] = '10.181.8.117'
    os.environ['MASTER_PORT'] = '12345'
    # torch.multiprocessing.spawn(main, args=(world_size, path), nprocs=world_size, join=True)

    main('./simulation_angular/angular_360')
