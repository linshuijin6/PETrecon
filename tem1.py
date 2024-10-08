import time

import numpy as np
import scipy.io as sio
import torch
from matplotlib import pyplot as plt

from geometry.BuildGeometry_v4 import BuildGeometry_v4
from utils.transfer_si import i2s

path = r"E:\dataset_pet\UDPET_Brain\dataset\dataset\train_mat\100_070722_1_20220707_162729_0.mat"
path2 = r"E:\dataset_pet\UDPET_Brain\dataset\dataset\train_mat\100_070722_1_20220707_162729_1.mat"

data1 = sio.loadmat(path)['img'][:, 0:128, :]
data2 = sio.loadmat(path2)['img'][:, 0:128, :]
data = np.concatenate((data1, data2), axis=0)
img = torch.from_numpy(data)
img = img.repeat(16, 1, 1)
# temPath = r'C:\pythonWorkSpace\tmp'
geoPath = './tmp_180_128128/geoMatrix-0.npy'
# # phanPath = r'E:\PET-M\Phantoms\Brainweb'
#
# radialBinCropFactor = 0
# PET = BuildGeometry_v4('mmr',radialBinCropFactor)
# PET.loadSystemMatrix(geoPath,is3d=False)
geoMatrix = []
geoMatrix.append(np.load(geoPath, allow_pickle=True))

time_s = time.time()
sino = i2s(img, 0, geoMatrix, 180)
print(time.time() - time_s)
plt.imshow(sino[0, :, :])
plt.show()
