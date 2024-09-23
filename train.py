import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DistributedSampler, DataLoader
import matplotlib.pyplot as plt
from geometry.BuildGeometry_v4 import BuildGeometry_v4
from utils.data import DatasetPETRecon
from utils.data import load_data, generate_mask
from recon_astraFBP import sino2pic as s2p
from model.whole_network import PETReconNet
# from model.network_swinTrans import SwinIR


def simulate_geometry(device):
    temPath = './tmp_1'
    PET = BuildGeometry_v4('mmr', device, 0.5)  # scanner mmr, with radial crop factor of 50%
    PET.loadSystemMatrix(temPath, is3d=False)
    return PET


def main(file_path):
    # # 数据
    # name_pre = 'transverse'
    # # X, sinogram; Y, pic
    # X_train, Y_train, X_test, Y_test, X_validation, Y_validation = load_data(file_path, name_pre)
    # mask_1, mask_2 = generate_mask(X_train.shape, sigma=0.1, column=True)
    # X1_train, X2_train = X_train * mask_1, X_train * mask_2
    # x1_pic, x2_pic = [], []
    # for sino_o1 in X1_train:
    #     x1_pic.append(s2p(sino_o1, 172))
    # for sino_o2 in X2_train:
    #     x2_pic.append(s2p(sino_o2, 172))
    # x1_input = torch.from_numpy(np.expand_dims(np.array(x1_pic), 1))
    # x2_input = torch.from_numpy(np.expand_dims(np.array(x2_pic), 1))
    # x1_pic, x2_pic = s2p((sino_o for sino_o in X1_train), 172), s2p((sino_o for sino_o in X1_train), 172)
    train_set = DatasetPETRecon(file_path, 'train')
    # train_sampler = DistributedSampler(train_set, shuffle=True, drop_last=True, seed=seed)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)

    # 模型初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PET = simulate_geometry(device)
    denoise_model = PETReconNet(PET, device=device).to(device)  # x (4,1,96,96) (batch_size_in_each_GPU, input_image_channel, H, W)
    criterion = nn.MSELoss()
    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(denoise_model.parameters(), lr=0.001)


    # 训练
    num_epochs = 100
    for epoch in range(num_epochs):
        denoise_model.train()
        running_loss = 0.0
        for inputs, _ in train_loader:
            x1, x2, AN, mask_p1, mask_p2, x_o = inputs
            x1, x2, AN, mask_p1, mask_p2, x_o = x1.to(device), x2.to(device), AN.to(device), mask_p1.to(device), mask_p2.to(device), x_o.to(device)
            x1_denoised = denoise_model(x1, x_o, AN, mask_p1)
            x2_denoised = denoise_model(x2, x_o, AN, mask_p2)
            sino_recon_1, _, _, _ = PET.simulateSinogramData(x1_denoised, AN)
            sino_recon_2, _, _, _ = PET.simulateSinogramData(x2_denoised, AN)
            sino_recon_1 = torch.from_numpy(np.expand_dims(sino_recon_1, (0, 1))).to(device) if len(sino_recon_1.shape) == 2 else torch.from_numpy(np.expand_dims(sino_recon_1, 1)).to(device)
            sino_recon_2 = torch.from_numpy(np.expand_dims(sino_recon_2, (0, 1))).to(device) if len(sino_recon_2.shape) == 2 else torch.from_numpy(np.expand_dims(sino_recon_2, 1)).to(device)
            sino_recon_p1, sino_recon_p2 = sino_recon_1 * mask_p2, sino_recon_2 * mask_p1
            sino_o_p1, sino_o_p2 = x_o*mask_p1, x_o*mask_p2
            lsp1, lsp2 = criterion(sino_recon_p1, sino_o_p2), criterion(sino_recon_p2, sino_o_p1)
            li = criterion(x1_denoised, x2_denoised)
            loss = lsp1 + lsp2 + li

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')


if __name__ == '__main__':
    main('./simulation_angular/angular_360')
