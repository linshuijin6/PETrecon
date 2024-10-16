import time

import numpy as np
import scipy.io as sio
import torch
from matplotlib import pyplot as plt
from skimage.transform import iradon, radon
from geometry.BuildGeometry_v4 import BuildGeometry_v4
from utils.transfer_si import i2s, s2i, i2s_radon

path = r"E:\dataset_pet\UDPET_Brain\dataset\dataset\train_mat\100_070722_1_20220707_162729_55.mat"
path2 = r"E:\dataset_pet\UDPET_Brain\dataset\dataset\train_mat\100_070722_1_20220707_162729_34.mat"
data = np.load('./simulation_angular/angular_180/sino_HD.npy', allow_pickle=True)
# data1 = sio.loadmat(path)['img'][:, 128:, :]
# data2 = sio.loadmat(path2)['img'][:, 128:, :]
# data = np.concatenate((data1, data2), axis=0)
# radon_data = radon(data[0, :, :], theta=np.linspace(0, 180, 180), circle=True)
iradon_data = iradon(data[0, :, :], theta=np.linspace(0, 180, 180), circle=True)
plt.imshow(iradon_data, 'gray')
plt.show()
plt.imshow(data[0, :, :], 'gray')
plt.show()
im_data = i2s_radon(data[0, :, :], 180)
img = torch.from_numpy(data[0, :, :])
img = img.repeat(16, 1, 1)
# temPath = r'C:\pythonWorkSpace\tmp'
geoPath = './tmp_180_172172/geoMatrix-0.npy'
# # phanPath = r'E:\PET-M\Phantoms\Brainweb'
#
# radialBinCropFactor = 0
# PET = BuildGeometry_v4('mmr',radialBinCropFactor)
# PET.loadSystemMatrix(geoPath,is3d=False)
geoMatrix = []
geoMatrix.append(np.load(geoPath, allow_pickle=True))

time_s = time.time()
sino = i2s(img, 0, geoMatrix, 180)
plt.imshow(sino[0, :, :].detach().cpu())
plt.show()
sino = sino[:, None, :, :]

im = s2i(sino, geoMatrix, 0)
plt.imshow(im[0, :, :].detach().cpu())
plt.show()
print(time.time() - time_s)
plt.imshow(sino[0, :, :])
plt.show()
