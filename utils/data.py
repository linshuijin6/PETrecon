import numpy as np
import torch
import torch.utils.data as data
from recon_astraFBP import sino2pic as s2p


def tv_loss(img):
    # 假设 img 的形状为 (batch_size, channel=1, height, width)
    # 因为 channel 为 1，可以直接在高度和宽度维度上计算总变差

    # 计算 X 方向的总变差 (沿着宽度方向的差异)
    n_pixel = img.shape[0] * img.shape[1] * img.shape[2] * img.shape[3]

    tv_x = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])
    n_tv_x = tv_x.shape[0] * tv_x.shape[1] * tv_x.shape[2] * tv_x.shape[3]
    v_tv_x = torch.sum(tv_x)

    # 计算 Y 方向的总变差 (沿着高度方向的差异)
    tv_y = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])
    n_tv_y = tv_y.shape[0] * tv_y.shape[1] * tv_y.shape[2] * tv_y.shape[3]
    v_tv_y = torch.sum(tv_y)

    mean_tv = (v_tv_x+v_tv_y)/(n_tv_x+n_tv_y)

    return mean_tv


def set_random_pixels_to_zero(data, ratio):
    """
    将输入张量中的部分像素值随机设为0.

    参数:
        data (torch.Tensor): 输入张量，形状为 (batchsize, c, h, w).
        ratio (float): 要设为0的像素占比（0到1之间的浮点数）。

    返回:
        torch.Tensor: 修改后的张量.
    """
    # 确保 ratio 在 [0, 1] 范围内
    if not (0 <= ratio <= 1):
        raise ValueError("Ratio must be between 0 and 1.")

    batchsize, c, h, w = data.shape
    # 计算要设为0的像素数量
    num_pixels_to_zero = int(batchsize * c * h * w * ratio)

    # 随机选择要设为0的索引
    indices = torch.randperm(batchsize * c * h * w)[:num_pixels_to_zero]

    # 将对应的像素值设为0
    data_flat = data.view(-1)  # 展平张量
    data_flat[indices] = 0
    return data_flat.view(batchsize, c, h, w)  # 还原原始形状


# 加噪声函数（适用于 PyTorch tensor）
def add_noise(img, radon, ratio):
    """
    img: 输入的图像张量，假设值在 [0, 1] 范围内，形状为 [batch_size, channels, height, width]
    mode: 噪声类型 'poisson+gaussian' 或 'gaussian'
    gauss_mean: 高斯噪声的均值
    gauss_std: 高斯噪声的标准差
    """
    img = img.to(radon.device).float()
    img = img[:, None, :, :] if len(img.shape)==3 else img
    noisy_image = img.clone()
    noisy_image = set_random_pixels_to_zero(noisy_image, ratio)
    noisy_sino = radon(noisy_image.to(img.device))
    return noisy_sino.squeeze(1).cpu()
    #
    #
    # if mode == 'p+g':
    #     # 1. 泊松噪声：Poisson分布中的数值是整数，因此需要将图像值扩展为较大范围
    #     noisy_poisson = torch.poisson(img * 255.0) / 255.0  # 归一化回 [0, 1]
    #
    #     # 2. 高斯噪声：生成高斯噪声并加到泊松噪声图像上
    #     noise_gauss = torch.normal(mean=gauss_mean, std=gauss_std / 255.0, size=noisy_poisson.shape).to(img.device)
    #     noisy_gaussian = torch.clamp(noisy_poisson + noise_gauss, 0.0, 1.0)
    #
    #     return noisy_gaussian
    #
    # elif mode == 'g':
    #     # 仅添加高斯噪声
    #     noise_gauss = torch.normal(mean=gauss_mean, std=gauss_std, size=img.shape).to(img.device)
    #     noisy_gaussian = img + noise_gauss
    #     # noisy_gaussian = torch.clamp(img + noise_gauss, 0.0, 1.0)
    #
    #     return noisy_gaussian
    #
    # else:
    #     raise ValueError("Invalid noise mode. Choose 'p+g' or 'g'.")


def load_data(dir_path, name_pre):
    file_path_pre = dir_path + '/' + name_pre
    file_sinoLD = np.load(file_path_pre + '_sinoLD.npy', allow_pickle=True)
    file_sinoHD = np.load(file_path_pre + '_sinoHD.npy', allow_pickle=True)
    file_imageLD = np.load(file_path_pre + '_picLD.npy', allow_pickle=True)
    file_imageHD = np.load(file_path_pre + '_picHD.npy', allow_pickle=True)

    # file_imageLD = np.rot90(file_imageLD, -1, (2, 3))
    # file_imageHD = np.rot90(file_imageHD, -1, (2, 3))

    # X_all = np.expand_dims(np.transpose(file_sinoLD, (0, 1, 2)), -1)
    # Y_all = np.expand_dims(np.transpose(file_imageHD, (0, 1, 2)), -1)
    X_all = file_sinoLD
    Y_all = file_imageHD


    return X_all, Y_all, file_imageLD, file_sinoHD


def generate_mask(dimensions, sigma, column=True):
    """
    生成batchsize个mask，对应的列置为0。

    参数:
    dimensions: tuple，图像尺寸，格式为(batchsize, radical, angular)
    sigma: float，置0的列占总列数的比值。

    输出:
    mask: np.array, 尺寸与输入尺寸一致的mask。
    """
    batchsize, _, radical, angular = dimensions
    # 初始化mask为全1
    mask = np.ones((batchsize, radical, angular))

    # 计算每个batch中需要置0的列数
    num_zero_columns = int(sigma * angular)

    for i in range(batchsize):
        # 随机选择需要置0的列索引
        zero_columns = np.random.choice(angular, num_zero_columns, replace=False)
        # 将对应列的值置为0
        if column:
            mask[i, :, zero_columns] = 0
        else:
            mask[i, zero_columns, :] = 0

    mask_p1 = mask
    mask_p2 = np.ones_like(mask) - mask

    return mask_p1, mask_p2


class DatasetPETRecon(data.Dataset):
    def __init__(self, file_path, radon, ratio):
        super().__init__()
        self.file_path = file_path
        self.radon = radon
        self.ratio = ratio
        self.x1_noisy, self.x2_noisy, self.Y_train, self.sino_label = self.prep_data()

    def __getitem__(self, index):
        x1 = self.x1_noisy[index, :, :, :]
        x2 = self.x2_noisy[index, :, :, :]
        Y = self.Y_train[index, :, :, :]
        sino_label = self.sino_label[index, :, :]
        X = (x1, x2)
        # elif self.phase == 'test':
        #     X = self.X_test
        #     Y = self.Y_test
        # elif self.phase == 'val':
        #     X = self.X_val
        #     Y = self.Y_val
        return X, Y, sino_label

    def __len__(self):
        return self.Y_train.shape[0]

    def prep_data(self):
        file_path = self.file_path
        # 数据
        name_pre = 'transverse'
        # X, sinogram; Y, pic
        X_train, Y_train, picLD_train, sino_label = load_data(file_path, name_pre)
        X_train, Y_train, picLD_train = torch.from_numpy(X_train), torch.from_numpy(Y_train), torch.from_numpy(picLD_train)
        # X_train_noisy1, X_train_noisy2 = add_noise(X_train, mode='g'), add_noise(X_train, mode='g')
        Y_train = Y_train.squeeze() if X_train.shape[0] != 1 else Y_train
        # gau_std = torch.std(X_train).item()*0.1
        X_train_noisy1, X_train_noisy2 = add_noise(picLD_train, self.radon, self.ratio), X_train  # noise2noise策略
        X_train_noisy1 = torch.unsqueeze(X_train_noisy1, 1)
        X_train_noisy2 = torch.unsqueeze(X_train_noisy2, 1)
        Y_train = torch.unsqueeze(Y_train, 1)

        return X_train_noisy1, X_train_noisy2, Y_train, sino_label

