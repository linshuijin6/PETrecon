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


# 加噪声函数（适用于 PyTorch tensor）
def add_noise(img, mode='p+g', gauss_mean=0, gauss_std=11):
    """
    img: 输入的图像张量，假设值在 [0, 1] 范围内，形状为 [batch_size, channels, height, width]
    mode: 噪声类型 'poisson+gaussian' 或 'gaussian'
    gauss_mean: 高斯噪声的均值
    gauss_std: 高斯噪声的标准差
    """
    if mode == 'p+g':
        # 1. 泊松噪声：Poisson分布中的数值是整数，因此需要将图像值扩展为较大范围
        noisy_poisson = torch.poisson(img * 255.0) / 255.0  # 归一化回 [0, 1]

        # 2. 高斯噪声：生成高斯噪声并加到泊松噪声图像上
        noise_gauss = torch.normal(mean=gauss_mean, std=gauss_std / 255.0, size=noisy_poisson.shape).to(img.device)
        noisy_gaussian = torch.clamp(noisy_poisson + noise_gauss, 0.0, 1.0)

        return noisy_gaussian

    elif mode == 'g':
        # 仅添加高斯噪声
        noise_gauss = torch.normal(mean=gauss_mean, std=gauss_std, size=img.shape).to(img.device)
        noisy_gaussian = img + noise_gauss
        # noisy_gaussian = torch.clamp(img + noise_gauss, 0.0, 1.0)

        return noisy_gaussian

    else:
        raise ValueError("Invalid noise mode. Choose 'p+g' or 'g'.")


def load_data(dir_path, name_pre):
    file_path_pre = dir_path + '/' + name_pre
    file_sinoLD = np.load(file_path_pre + '_sinoHD.npy', allow_pickle=True)
    file_imageHD = np.load(file_path_pre + '_picHD.npy', allow_pickle=True)
    file_AN = np.load(file_path_pre + '_AN.npy', allow_pickle=True)

    # X_all = np.expand_dims(np.transpose(file_sinoLD, (0, 1, 2)), -1)
    # Y_all = np.expand_dims(np.transpose(file_imageHD, (0, 1, 2)), -1)
    X_all = file_sinoLD
    Y_all = file_imageHD
    AN_all = file_AN
    train_size = int(0.8 * len(X_all))
    test_size = int(0.1 * len(X_all))
    validation_size = len(X_all) - train_size - test_size  # 剩下的作为验证集

    # 按比例分割数据
    X_train = X_all[:train_size]
    Y_train = Y_all[:train_size]
    AN_train = AN_all[:train_size]
    X_test = X_all[train_size:train_size + test_size]
    Y_test = Y_all[train_size:train_size + test_size]
    AN_test = AN_all[train_size:train_size + test_size]
    X_validation = X_all[train_size + test_size:]
    Y_validation = Y_all[train_size + test_size:]
    AN_validation = AN_all[train_size + test_size:]
    return X_train, AN_train, Y_train, X_test, AN_test, Y_test, X_validation, AN_validation, Y_validation


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
    def __init__(self, file_path, phase):
        super().__init__()
        self.phase = phase
        self.file_path = file_path
        self.x1_noisy, self.x2_noisy, self.AN_train, self.Y_train = self.prep_data()

    def __getitem__(self, index):
        if self.phase == 'train':
            x1 = self.x1_noisy[index, :, :, :]
            x2 = self.x2_noisy[index, :, :, :]
            # mask_1 = self.mask_1[index, :, :, :]
            # mask_2 = self.mask_2[index, :, :, :]
            Y = self.Y_train[index, :, :, :]
            # x_o = self.X_train[index, :, :, :]
            AN = self.AN_train[index, :, :, :]
            X = (x1, x2, AN)
        # elif self.phase == 'test':
        #     X = self.X_test
        #     Y = self.Y_test
        # elif self.phase == 'val':
        #     X = self.X_val
        #     Y = self.Y_val
        return X, Y

    def __len__(self):
        return self.Y_train.shape[0]

    def prep_data(self):
        file_path = self.file_path
        # 数据
        name_pre = 'transverse'
        # X, sinogram; Y, pic
        X_train, AN_train, Y_train, X_test, AN_test, Y_test, X_validation, AN_validation, Y_validation = load_data(file_path, name_pre)
        X_train, Y_train, AN_train = torch.from_numpy(X_train), torch.from_numpy(Y_train), torch.from_numpy(AN_train)
        # X_train_noisy1, X_train_noisy2 = add_noise(X_train, mode='g'), add_noise(X_train, mode='g')
        X_train_noisy1, X_train_noisy2 = add_noise(X_train, mode='g'), X_train  # noise2noise策略
        X_train_noisy1 = torch.unsqueeze(X_train_noisy1, 1)
        X_train_noisy2 = torch.unsqueeze(X_train_noisy2, 1)
        Y_train = torch.unsqueeze(Y_train, 1)
        AN_train = torch.unsqueeze(AN_train, 1)

        return X_train_noisy1, X_train_noisy2, AN_train, Y_train

        mask_1, mask_2 = generate_mask(X_train.shape, sigma=0.1, column=True)
        X1_train, X2_train = X_train * mask_1, X_train * mask_2
        x1_pic, x2_pic = [], []
        for sino_o1 in X1_train:
            x1_pic.append(s2p(sino_o1, 168).numpy())
        for sino_o2 in X2_train:
            x2_pic.append(s2p(sino_o2, 168).numpy())
        x1_input = torch.from_numpy(np.expand_dims(np.array(x1_pic), 1))
        x2_input = torch.from_numpy(np.expand_dims(np.array(x2_pic), 1))
        mask_1 = torch.from_numpy(np.expand_dims(mask_1, 1))
        mask_2 = torch.from_numpy(np.expand_dims(mask_2, 1))
        Y_train = torch.from_numpy(np.expand_dims(Y_train, 1))
        AN_train = torch.from_numpy(np.expand_dims(AN_train, 1))
        AN_test = torch.from_numpy(np.expand_dims(AN_test, 1))
        AN_validation = torch.from_numpy(np.expand_dims(AN_validation, 1))
        X_test = torch.from_numpy(np.expand_dims(X_test, 1))
        X_train = torch.from_numpy(np.expand_dims(X_train, 1))
        Y_test = torch.from_numpy(np.expand_dims(Y_test, 1))
        X_validation = torch.from_numpy(np.expand_dims(X_validation, 1))
        Y_validation = torch.from_numpy(np.expand_dims(Y_validation, 1))
        return x1_input, AN_train, x2_input, Y_train, X_test, AN_test, Y_test, X_validation, AN_validation, Y_validation, mask_1, mask_2, X_train
