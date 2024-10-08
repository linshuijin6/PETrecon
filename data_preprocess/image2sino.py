import os
import torch
import numpy
import numpy as np
import scipy.io as sio
from utils.transfer_si import i2s
import matplotlib.pyplot as plt

# for UDPET-brain
path_root = "/mnt/data/linshuijin/data_PET/data_UDPET_brain/train"
file_names = os.listdir(path_root)
ldSinos = []
hdSinos = []
ldImgs = []
hdImgs = []
k = 0
n = 2
for i, file_name in enumerate(file_names):
    file_path = os.path.join(path_root, file_name)
    raw_data = sio.loadmat(file_path)['img']
    ldImg = raw_data[:, 0:128, :]
    hdImg = raw_data[:, 128:256, :]
    ldImgs.append(ldImg), hdImgs.append(hdImg)
    torch_ldImgs = torch.from_numpy(np.array(ldImg)).to('cuda').squeeze(1)
    dSinos = i2s(torch_ldImgs, 0, sinogram_nAngular=180)
    if n*k <= i < n*(k+1):
        ldImgs.append(ldImg), hdImgs.append(hdImg)
    if i > n*(k+1):
        break

torch_ldImgs = torch.from_numpy(np.array(ldImgs)).to('cuda').squeeze(1)
torch_hdImgs = torch.from_numpy(np.array(hdImgs)).to('cuda').squeeze(1)

ldSinos = i2s(torch_ldImgs, 0, sinogram_nAngular=180)
hdSinos = i2s(torch_hdImgs, 0, sinogram_nAngular=180)

ldSino_np = ldSinos.cpu().numpy()
hdSino_np = hdSinos.cpu().numpy()
np.save('../data/ldSino.npy', ldSino_np)
np.save('../data/hdSino.npy', hdSino_np)
np.save('../data/ldImg.npy', ldImgs)
np.save('../data/hdImg.npy', hdImgs)
