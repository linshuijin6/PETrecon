import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch_radon import Radon

from model.whole_network import normalization2one
from utils.radon import Radon as MeRadon
from result_eval.evaluate import show_images, calculate_psnr

angles = np.linspace(0, np.pi, 180, endpoint=False)
device = 'cpu'
radon = Radon(resolution=128, angles=angles, clip_to_circle=True)
me_torch = MeRadon(n_theta=180, circle=True, device=device)
data_n = np.load('./angular_180/test_transverse_sinoLD.npy', allow_pickle=True)
imgLD = np.load('./angular_180/test_transverse_picLD.npy', allow_pickle=True)

with torch.no_grad():

    picLD = torch.from_numpy(imgLD).float().to(device)

    x = torch.FloatTensor(imgLD).to(device)
    # time_s = time.time()
    sinogram = radon.forward(x)
    # t_2 = time.time()
    # print('Time:', t_2 - time_s)
    t_3 = time.time()
    filtered_sinogram = radon.filter_sinogram(sinogram)
    # print(time.time() - t_3)
    fbp = radon.backprojection(filtered_sinogram)
    print(time.time() - t_3)
    print(calculate_psnr(normalization2one(picLD), normalization2one(fbp)))
    # show_images([x, sinogram, filtered_sinogram, fbp], ['Original', 'Sinogram', 'Filtered Sinogram', 'FBP'], keep_range=False)
    # plt.show()

    sino_recon = me_torch(picLD)
    t_3 = time.time()
    reconLD = me_torch.filter_backprojection(sino_recon)
    print(time.time() - t_3)
    print(calculate_psnr(normalization2one(picLD), normalization2one(reconLD)))


    # show_images([picLD.squeeze(), sino_recon.squeeze(), reconLD.squeeze()], ['Original', 'Sinogram', 'FBP'], keep_range=False)
    # plt.show()


