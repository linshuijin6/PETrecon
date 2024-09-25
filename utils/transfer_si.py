import numpy as np
import torch
from skimage.transform import iradon


def gaussFilter(img, fwhm, is3d=False, batch_size=1, image_voxelSizeCm=(0.41725 / 2, 0.41725 / 2, 0.40625 / 2)):
    # 3D aniso/isotropic Gaussian filtering

    fwhm = np.array(fwhm)
    if np.all(fwhm == 0):
        return img
    from scipy import ndimage
    image_matrixSize = img.shape
    if is3d:
        img = img.reshape(image_matrixSize, order='F')
        voxelSizeCm = image_voxelSizeCm
    else:
        img = img.reshape(image_matrixSize[:2], order='F')
        voxelSizeCm = image_voxelSizeCm[:2]
    if fwhm.shape == 1:
        if is3d:
            fwhm = fwhm * np.ones([3, ])
        else:
            fwhm = fwhm * np.ones([2, ])
    sigma = fwhm / voxelSizeCm / np.sqrt(2 ** 3 * np.log(2))
    imOut = ndimage.filters.gaussian_filter(img, sigma)
    return imOut.flatten('F')


def i2s(img, AN, sinogram_nAngular=180, psf=0):
    # img.shape = (batchsize, h, w)
    img = img.squeeze(1) if len(img.shape)==4 else img
    img = img.float()
    assert isinstance(img, torch.Tensor)
    sinogram_nRadial = img.shape[1]
    # img = img*AN
    if img.ndimension() == 2:
        batch_size = 1
        img = img.unsqueeze(0)
    else:
        batch_size = img.shape[0]
    img = img.view(batch_size, -1)
    if psf:
        for b in range(batch_size):
            img[b, :] = gaussFilter(img[b, :], psf)
    dims = [batch_size, sinogram_nRadial, sinogram_nAngular]
    # if tof: dims.append(self.sinogram.nTofBins)
    y = torch.zeros(dims, dtype=torch.float32, device=img.device)
    matrixSize = [sinogram_nRadial, sinogram_nRadial]
    q = sinogram_nAngular // 2

    geoMatrix = []
    geoMatrix.append(np.load("./tmp_1/" + 'geoMatrix-0.npy', allow_pickle=True))

    for i in range(sinogram_nAngular // 2):
        for j in range(sinogram_nRadial):
            M0 = geoMatrix[0][i, j]
            if not np.isscalar(M0):
                M = torch.tensor(M0[:, 0:3], dtype=torch.int32, device=img.device)  # 转为 tensor
                G = torch.tensor(M0[:, 3] / 1e4, dtype=torch.float32, device=img.device)  # 转为 tensor
                idx1 = M[:, 0] + M[:, 1] * matrixSize[0]
                idx2 = M[:, 1] + matrixSize[0] * (matrixSize[0] - 1 - M[:, 0])
                for b in range(batch_size):
                    y[b, j, i] = torch.dot(G, img[b, idx1.long()])
                    y[b, j, i + q] = torch.dot(G, img[b, idx2.long()])
    if batch_size == 1:
        y = y[0, :, :]
    # print(f'{batch_size} batches forward-projected\n')
    return y


def _sinogram_circle_to_square(sinogram, device='cuda'):
    import torch.nn.functional
    # 计算对角线长度（向上取整）
    diagonal = int(torch.ceil(torch.sqrt(torch.tensor(2.0, device=device)) * sinogram.shape[0]))

    # 计算需要填充的大小
    pad = diagonal - sinogram.shape[0]
    old_center = sinogram.shape[0] // 2
    new_center = diagonal // 2
    pad_before = new_center - old_center

    # 定义填充的宽度，先在上下方向进行填充
    pad_width = (pad_before, pad - pad_before)

    # 使用 torch.nn.functional.pad 进行填充
    sinogram_padded = torch.nn.functional.pad(sinogram, (0, 0, pad_width[0], pad_width[1]), mode='constant', value=0)

    return sinogram_padded


def convert_to_float(image, preserve_range=True):
    # 检查是否是 float16 类型
    if image.dtype == torch.float16:
        return image.to(torch.float32)  # 转换为 float32

    if preserve_range:
        # 仅当图像不是 float32 或 float64 时，转换为 float
        if image.dtype not in [torch.float32, torch.float64]:
            image = image.to(torch.float32)

    return image


def _get_fourier_filter(size, filter_name, device='cuda'):
    # 创建频率数组 n
    n = torch.cat((torch.arange(1, size / 2 + 1, 2, device=device, dtype=torch.int),
                   torch.arange(size / 2 - 1, 0, -2, device=device, dtype=torch.int)))
    pi = torch.tensor(torch.acos(torch.zeros(1)).item() * 2)  # pi ≈ 3.141592653589793
    # 初始化滤波器数组 f
    f = torch.zeros(size, device=device)
    f[0] = 0.25
    f[1::2] = -1 / (pi * n) ** 2

    # 通过傅里叶变换计算 ramp 滤波器
    fourier_filter = 2 * torch.real(torch.fft.fft(f))  # ramp filter

    if filter_name == "ramp":
        pass
    elif filter_name == "shepp-logan":
        # Shepp-Logan 滤波器，避免除以零，从第二个元素开始
        omega = pi * torch.fft.fftfreq(size, device=device)[1:]
        fourier_filter[1:] *= torch.sin(omega) / omega
    elif filter_name == "cosine":
        # Cosine 滤波器
        freq = torch.linspace(0, pi, size, device=device, dtype=torch.float32, requires_grad=False)
        cosine_filter = torch.fft.fftshift(torch.sin(freq))
        fourier_filter *= cosine_filter
    elif filter_name == "hamming":
        # Hamming 滤波器
        fourier_filter *= torch.fft.fftshift(torch.hamming_window(size, device=device, dtype=torch.float32))
    elif filter_name == "hann":
        # Hann 滤波器
        fourier_filter *= torch.fft.fftshift(torch.hann_window(size, device=device, dtype=torch.float32))
    elif filter_name is None:
        fourier_filter[:] = 1

    return fourier_filter[:, None]  # 增加维度以保持与输入一致


def interp(x, xp, fp):
    """
    实现 1D 线性插值功能，类似于 numpy.interp。

    :param x: 要插值的位置
    :param xp: 已知数据点的 x 坐标
    :param fp: 已知数据点的 y 坐标
    :return: 在 x 位置的插值值
    """
    x = torch.tensor(x, dtype=torch.float32, device=xp.device)
    xp = torch.tensor(xp, dtype=torch.float32, device=xp.device)
    fp = torch.tensor(fp, dtype=torch.float32, device=fp.device)

    # 确保 xp 是递增的，否则排序
    indices = torch.argsort(xp)
    xp = xp[indices]
    fp = fp[indices]

    # 计算插值
    left = torch.searchsorted(xp, x, right=True) - 1
    right = left + 1

    left = torch.clamp(left, 0, len(xp) - 1)
    right = torch.clamp(right, 0, len(xp) - 1)

    x_left = xp[left]
    x_right = xp[right]
    y_left = fp[left]
    y_right = fp[right]

    slope = (y_right - y_left) / (x_right - x_left)
    return y_left + slope * (x - x_left)


def s2i_batch(radon_image):
    # radon_image: batchsize, channel=1, h, w
    assert len(radon_image.shape) == 4
    radon_image = radon_image.squeeze(1)
    sino_t = []
    for sino in radon_image:
        sino_t.append(s2i(sino))
    out_sino = torch.stack(sino_t, 0)
    out_sino = out_sino.unsqueeze(1)
    return out_sino


def s2i(radon_image, theta=None, output_size=None,
        filter_name="ramp", interpolation="linear", circle=True,
        preserve_range=True, device='cuda'):
    import torch
    import numpy as np
    from torch.fft import fft, ifft
    from functools import partial
    from scipy.interpolate import interp1d
    import torch.nn.functional
    if radon_image.ndim != 2:
        raise ValueError('The input image must be 2-D')

    # 将 radon_image 转换为 GPU 上的 float tensor
    radon_image = radon_image.to(device).float()

    if theta is None:
        theta = torch.linspace(0, 180, radon_image.shape[1], device=device, dtype=torch.float32,
                               requires_grad=False)

    angles_count = len(theta)
    if angles_count != radon_image.shape[1]:
        raise ValueError("The given ``theta`` does not match the number of "
                         "projections in ``radon_image``.")

    interpolation_types = ('linear', 'nearest', 'cubic')
    if interpolation not in interpolation_types:
        raise ValueError(f"Unknown interpolation: {interpolation}")

    filter_types = ('ramp', 'shepp-logan', 'cosine', 'hamming', 'hann', None)
    if filter_name not in filter_types:
        raise ValueError(f"Unknown filter: {filter_name}")

    # 保留范围处理 (需要自行定义 convert_to_float 函数)
    radon_image = convert_to_float(radon_image, preserve_range)
    dtype = radon_image.dtype

    img_shape = radon_image.shape[0]
    if output_size is None:
        # 如果没有指定输出大小，根据输入的 radon image 来估计
        if circle:
            output_size = img_shape
        else:
            output_size = int(torch.floor(torch.sqrt(torch.tensor((img_shape) ** 2 / 2.0, device=device))))

    if circle:
        radon_image = _sinogram_circle_to_square(radon_image)
        img_shape = radon_image.shape[0]

    # 将图像大小调整为 2 的幂次以加速 Fourier 变换
    projection_size_padded = max(64, int(2 ** torch.ceil(torch.log2(torch.tensor(2 * img_shape, device=device)))))
    pad_width = ((0, projection_size_padded - img_shape), (0, 0))
    img = torch.nn.functional.pad(radon_image, (0, 0, 0, projection_size_padded - img_shape), 'constant', 0)

    # 在傅里叶域应用滤波器 (需要自行定义 _get_fourier_filter 函数)
    fourier_filter = _get_fourier_filter(projection_size_padded, filter_name, device=device)
    projection = fft(img, dim=0) * fourier_filter
    radon_filtered = torch.real(ifft(projection, dim=0)[:img_shape, :])

    # 通过插值重建图像
    reconstructed = torch.zeros((output_size, output_size), dtype=dtype, device=device)
    radius = output_size // 2
    xpr, ypr = torch.meshgrid(torch.arange(output_size, device=device) - radius,
                              torch.arange(output_size, device=device) - radius)
    x = torch.arange(img_shape, device=device) - img_shape // 2

    for col, angle in zip(radon_filtered.T, torch.deg2rad(theta)):
        t = ypr * torch.cos(angle) - xpr * torch.sin(angle)
        if interpolation == 'linear':
            interpolant = partial(interp, xp=x, fp=col)
        else:
            col_np = col.cpu().numpy()  # 需要将数据移到CPU上，使用 interp1d
            interpolant = interp1d(x.cpu().numpy(), col_np, kind=interpolation, bounds_error=False, fill_value=0)
            t_np = t.cpu().numpy()
        reconstructed += torch.tensor(interpolant(x=t), device=device)

    if circle:
        out_reconstruction_circle = (xpr ** 2 + ypr ** 2) > radius ** 2
        reconstructed[out_reconstruction_circle] = 0.

    pi = torch.tensor(torch.acos(torch.zeros(1)).item() * 2)  # pi ≈ 3.141592653589793

    return reconstructed * pi / (2 * angles_count)
    # iradon()
