import numpy as np
import torch
import torch.utils.data as data
from recon_astraFBP import sino2pic as s2p


def load_data(dir_path, name_pre):
    file_path_pre = dir_path + '/' + name_pre
    file_sinoLD = np.load(file_path_pre + '_sinoHD.npy', allow_pickle=True)[:, 0:168, :]
    file_imageHD = np.load(file_path_pre + '_picHD.npy', allow_pickle=True)[:, 2:170, 2:170]
    file_AN = np.load(file_path_pre + '_AN.npy', allow_pickle=True)[:, 0:168, :]

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
    batchsize, radical, angular = dimensions
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
        self.x1_train, self.AN_train, self.x2_train, self.Y_train, self.X_test, self.AN_test, self.Y_test, self.X_val, self.AN_val, self.Y_val, self.mask_1, self.mask_2, self.X_train = self.prep_data()

    def __getitem__(self, index):
        if self.phase == 'train':
            x1 = self.x1_train[index, :, :, :]
            x2 = self.x2_train[index, :, :, :]
            mask_1 = self.mask_1[index, :, :, :]
            mask_2 = self.mask_2[index, :, :, :]
            Y = self.Y_train[index, :, :, :]
            x_o = self.X_train[index, :, :, :]
            AN = self.AN_train[index, :, :, :]
            X = (x1, x2, AN, mask_1, mask_2, x_o)
        elif self.phase == 'test':
            X = self.X_test
            Y = self.Y_test
        elif self.phase == 'val':
            X = self.X_val
            Y = self.Y_val
        return X, Y

    def __len__(self):
        return self.x1_train.shape[0]

    def prep_data(self):
        file_path = self.file_path
        # 数据
        name_pre = 'transverse'
        # X, sinogram; Y, pic
        X_train, AN_train, Y_train, X_test, AN_test, Y_test, X_validation, AN_validation, Y_validation = load_data(file_path, name_pre)
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
