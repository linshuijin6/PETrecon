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
n = 50000
temPath = '/mnt/data/linshuijin/PETrecon/tmp_180_128*128/'
geoMatrix = []
geoMatrix.append(np.load(temPath + 'geoMatrix-0.npy', allow_pickle=True))
for i, file_name in enumerate(file_names):
    file_path = os.path.join(path_root, file_name)
    raw_data = sio.loadmat(file_path)['img']
    ldImg = raw_data[:, 0:128, :]
    hdImg = raw_data[:, 128:256, :]
    # ldImgs.append(ldImg), hdImgs.append(hdImg)
    torch_ldImgs = torch.from_numpy(np.array(ldImg)).to('cuda').squeeze(1)
    # dSinos = i2s(torch_ldImgs, 0, sinogram_nAngular=180, geoMatrix=geoMatrix)
    if n*k <= i < n*(k+1):
        ldImgs.append(ldImg), hdImgs.append(hdImg)
    if i > n*(k+1):
        break

# torch_ldImgs = torch.from_numpy(np.array(ldImgs)).to('cuda').squeeze(1)
torch_hdImgs = torch.from_numpy(np.array(hdImgs)).to('cuda').squeeze(1)

# ldSinos = i2s(torch_ldImgs, 0, sinogram_nAngular=180, geoMatrix=geoMatrix)
# ldSino_np = ldSinos.cpu().numpy()
# np.save(f'/mnt/data/linshuijin/PETrecon/simulation_angular/angular_180/transverse_sinoLD.npy', ldSino_np)
# del ldSinos, ldSino_np, torch_ldImgs

hdSinos = i2s(torch_hdImgs, 0, sinogram_nAngular=180, geoMatrix=geoMatrix)
hdSino_np = hdSinos.cpu().numpy()
np.save(f'/mnt/data/linshuijin/PETrecon/simulation_angular/angular_180/transverse_sinoHD.npy', hdSino_np)
np.save(f'/mnt/data/linshuijin/PETrecon/simulation_angular/angular_180/transverse_picLD.npy', ldImgs)
np.save(f'/mnt/data/linshuijin/PETrecon/simulation_angular/angular_180/transverse_picHD.npy', hdImgs)
# np.save(f'/mnt/data/linshuijin/PETrecon/simulation_angular/angular_180/transverse_p{k+1}_4_sinoLD.npy', ldSino_np)
# np.save(f'/mnt/data/linshuijin/PETrecon/simulation_angular/angular_180/transverse_p{k+1}_4_sinoHD.npy', hdSino_np)
# np.save(f'/mnt/data/linshuijin/PETrecon/simulation_angular/angular_180/transverse_p{k+1}_4_picLD.npy', ldImgs)
# np.save(f'/mnt/data/linshuijin/PETrecon/simulation_angular/angular_180/transverse_p{k+1}_4_picHD.npy', hdImgs)
