import argparse
import logging
import random
import time
import numpy as np
import torch
import yaml
from PIL import Image
from torch import nn, optim
from torch.utils.data import DistributedSampler, DataLoader, random_split
import matplotlib.pyplot as plt
from geometry.BuildGeometry_v4 import BuildGeometry_v4
from utils.radon import Radon
from utils.data import DatasetPETRecon, tv_loss
from utils.data import load_data, generate_mask
# from recon_astraFBP import sino2pic as s2p
from model.whole_network import PETReconNet, PETDenoiseNet, SwinDenoise
# from utils.transfer_si import s2i, i2s, s2i_batch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
from model.whole_network import normalization2one
# from model.network_swinTrans import SwinIR
from modelSwinUnet.SUNet import SUNet_model
from torch.utils.tensorboard import SummaryWriter
from result_eval.evaluate import calculate_metrics


def preset_seed(seed_number):
    import torch.backends.cudnn
    seed = seed_number
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果有多个 GPU
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def simulate_geometry(device):
    temPath = './tmp_1'
    PET = BuildGeometry_v4('mmr', device, 0.5)  # scanner mmr, with radial crop factor of 50%
    PET.loadSystemMatrix(temPath, is3d=False)
    return PET


def train(model_pre, radon, train_loader, criterion, optimizer, rank, epoch, log_writer):
    model_pre.train()
    # model_recon.train()
    running_loss = 0.0

    for iteration, (inputs, Y, sino_label, picLD) in enumerate(train_loader):

        # if (iteration % 10 == 0) and sign:

        time_s = time.time()
        #     sign = not sign
        x1, x2 = inputs
        x1, x2 = x1.to(rank), x2.to(rank)
        bs = x1.shape[0]
        # print(torch.cuda.memory_summary())

        # sinogram去噪，noise2noise训练
        x1_denoised = model_pre(x1)
        # x2_denoised = model_pre(x2)

        x2_denoised = x2
        # 平均输出的sinogram
        aver_x = (x1_denoised + normalization2one(x2_denoised)) / 2
        mid_recon = normalization2one(radon.filter_backprojection(aver_x))

        # 以下【去除】PET去噪器的原自监督设计
        # 计算mask角度下的loss

        # mask_p1, mask_p2 = generate_mask(mid_recon.shape, 0.5)
        #
        # # mask_p1, _ = generate_mask(aver_x.shape, 0.1)
        # # _, mask_p2 = generate_mask(aver_x.shape, 0.1)
        # mask_p1, mask_p2 = torch.from_numpy(mask_p1).unsqueeze(1).float().to(rank), torch.from_numpy(mask_p2).unsqueeze(
        #     1).float().to(rank)
        # i_in_m1, i_in_m2 = mid_recon * mask_p1, mid_recon * mask_p2
        # s_in_1, s_in_2 = radon(i_in_m1), radon(i_in_m2)
        # s_out_1, s_out_2 = model_pre(s_in_1), model_pre(s_in_2)
        # i_out_1, i_out_2 = radon.filter_backprojection(s_out_1), radon.filter_backprojection(s_out_2)
        # i_out_m12, i_out_m21 = i_out_1 * mask_p2, i_out_2 * mask_p1
        # lsm1, lsm2 = criterion(normalization2one(i_out_m12), normalization2one(i_in_m2)), criterion(normalization2one(i_out_m21), normalization2one(i_in_m1))
        # lsi = criterion(normalization2one(i_out_1), normalization2one(i_out_2))
        # lsm = lsm1 + lsm2
        # ls_post = lsm + lsi
        # ls_sino = criterion(x1_denoised, x2_denoised)
        # loss = ls_sino + ls_post + args.alpha * tv_loss(x1_denoised) - args.beta * tv_loss(mid_recon)
        loss = criterion(aver_x, x2) + args.alpha * tv_loss(aver_x) - args.beta * tv_loss(mid_recon)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for name, param in model_pre.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).sum() > 0:
                print(f"NaN detected in gradient of {name}")


        loss_cur = loss.item() / bs
        running_loss += loss_cur
        if iteration % 100 == 0:
            logger.info(
                f'Epoch:{epoch}, Iteration: {iteration}/{len(train_loader)}, Loss: {loss_cur:.4f}, Time/p_i: {time.time() - time_s:.4f}')

            # 定义图像数据和标题
            pics = [x1, x2, x1_denoised, aver_x, picLD, mid_recon,
                    aver_x, sino_label, mid_recon, Y]
            titles = ["x1", "x2", "x1_denoised", "aver_x", "input_LD", "mid_recon",
                      'aver_recon_sino', 'sino_label', 'pic_recon', 'pic_label']
            # 设置图像整体大小
            fig, axes = plt.subplots(5, 2, figsize=(12, 18))
            fig.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.05, hspace=0.3, wspace=0.2)  # 减少图像间的间距

            # 绘制每个子图
            for i, (pic, title) in enumerate(zip(pics, titles)):
                ax = axes[i // 2, i % 2]  # 计算当前的行列位置
                pic_t = pic[0, 0, :, :].cpu().detach().numpy() if len(pic.shape) == 4 else pic[0, :, :].numpy()
                ax.imshow(pic_t)  # 使用灰度显示图像
                ax.set_title(title, fontsize=10)  # 设置子图标题=
                # 以下代码组合移除坐标轴、刻度和边框
                ax.axis('off')  # 移除边框和坐标轴
                ax.get_xaxis().set_ticks([])  # 移除 x 轴刻度
                ax.get_yaxis().set_ticks([])  # 移除 y 轴刻度
                ax.spines['top'].set_visible(False)  # 隐藏顶部边框
                ax.spines['bottom'].set_visible(False)  # 隐藏底部边框
                ax.spines['left'].set_visible(False)  # 隐藏左侧边框
                ax.spines['right'].set_visible(False)  # 隐藏右侧边框
            # 设置大标题
            fig.suptitle(
                f'Epoch:{epoch}, Iteration: {iteration}, a_loss: {loss_cur:.4f}, seed: {seed}',
                fontsize=14)
            # 调整布局，避免重叠

            plt.tight_layout(pad=2.0, rect=[0, 0.03, 1, 0.95])
            os.mkdir(os.path.join(args.log_dir, 'pic_visual')) if not os.path.exists(
                os.path.join(args.log_dir, 'pic_visual')) else None
            pic_path = os.path.join(args.log_dir, 'pic_visual', f'tr_{iteration}_{loss_cur:.4f}_s{seed}.png')
            plt.savefig(pic_path, format='png', dpi=300, bbox_inches='tight')  # 300 dpi保证高分辨率

            if args.show_tr:
                # 显示图像
                plt.show()
                plt.close()
                # 将 matplotlib 图像转换为 PIL 图像
                image = Image.open(pic_path)  # 使用 PIL 打开图像
                image = np.array(image)  # 转换为 NumPy 数组

                # 转换为 Tensor 格式，TensorBoard 需要的形状是 (C, H, W)
                image_tensor = torch.from_numpy(image).permute(2, 0, 1)  # 从 (H, W, C) 转换为 (C, H, W)

                # 记录图像到 TensorBoard
                # log_writer.add_image('train duration', image_tensor, epoch)
        # logger.info(f'Train Loss: {running_loss:.4f}')

    loss_average = running_loss / len(train_loader)
    log_writer.add_scalar('train/loss', loss_average, epoch)  # 在一个 tag 下面添加多个折线图
    return loss_average


def validate(model_pre, radon, val_loader, criterion, rank, epoch, log_writer):
    model_pre.eval()
    running_loss = 0.0

    with torch.no_grad():
        for iteration, (inputs, Y, sino_label, picLD) in enumerate(val_loader):
            x1, x2 = inputs
            x1, x2 = x1.to(rank), x2.to(rank)
            Y = Y.to(rank).float()

            # sinogram去噪，noise2noise训练
            x1_denoised = x1
            x2_denoised = model_pre(x2)
            # x2_denoised = x2
            # 平均输出的sinogram
            aver_x = x2_denoised
            mid_recon = normalization2one(radon.filter_backprojection(aver_x))

            # PET图去噪
            # p_out = model_recon(mid_recon, aver_x, torch.ones_like(aver_x))
            # pic_recon = (p_out + mid_recon) / 2
            # sino_recon = radon(pic_recon)

            loss = criterion(mid_recon, Y)
            loss_cur = loss.item() / x1.shape[0]
            running_loss += loss.item()
            if iteration % 40 == 0:
                logger.info(f'Epoch:{epoch}, Validation Loss: {loss_cur:.4f}')
                # 定义图像数据和标题
                pics = [x1, x2, x1_denoised, aver_x, x2_denoised, aver_x, mid_recon, picLD, aver_x, sino_label, mid_recon, Y]
                titles = ["x1", "x2", "x1_denoised", "aver_x", "x2_denoised", "aver_x", "mid_recon", "input_LD", "sino_recon",
                          'label_sino', 'pic_recon', 'label']
                # 设置图像整体大小
                fig, axes = plt.subplots(6, 2, figsize=(12, 18))
                fig.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.05, hspace=0.3, wspace=0.2)  # 减少图像间的间距

                # 绘制每个子图
                for i, (pic, title) in enumerate(zip(pics, titles)):
                    ax = axes[i // 2, i % 2]  # 计算当前的行列位置
                    pic_t = pic[0, 0, :, :].cpu().detach().numpy() if len(pic.shape) == 4 else pic[0, :, :].numpy()
                    ax.imshow(pic_t)  # 使用灰度显示图像
                    ax.set_title(title, fontsize=10)  # 设置子图标题=
                    # 以下代码组合移除坐标轴、刻度和边框
                    ax.axis('off')  # 移除边框和坐标轴
                    ax.get_xaxis().set_ticks([])  # 移除 x 轴刻度
                    ax.get_yaxis().set_ticks([])  # 移除 y 轴刻度
                    ax.spines['top'].set_visible(False)  # 隐藏顶部边框
                    ax.spines['bottom'].set_visible(False)  # 隐藏底部边框
                    ax.spines['left'].set_visible(False)  # 隐藏左侧边框
                    ax.spines['right'].set_visible(False)  # 隐藏右侧边框
                # 设置大标题
                fig.suptitle(
                    f'Validate, Epoch:{epoch}, loss_cur: {loss_cur:.4f}', fontsize=14)
                # 调整布局，避免重叠

                plt.tight_layout(pad=2.0, rect=[0, 0.03, 1, 0.95])
                # os.mkdir(os.path.join(args.log_dir, 'pic_visual')) if not os.path.exists(
                #     os.path.join(args.log_dir, 'pic_visual')) else None
                pic_path = os.path.join(args.log_dir, 'pic_visual', f'val_{iteration}_{loss_cur:.4f}_s{seed}.png')
                # pic_path = f'/home/ssddata/linshuijin/PETrecon/log_file/pic_visual/val_{iteration}_{loss_cur:.4f}.png'
                plt.savefig(pic_path, format='png', dpi=300, bbox_inches='tight')  # 300 dpi保证高分辨率

                if args.show_val:
                    # 显示图像
                    plt.show()
                    plt.close()
                    # 将 matplotlib 图像转换为 PIL 图像
                    image = Image.open(pic_path)  # 使用 PIL 打开图像
                    image = np.array(image)  # 转换为 NumPy 数组

                    # 转换为 Tensor 格式，TensorBoard 需要的形状是 (C, H, W)
                    image_tensor = torch.from_numpy(image).permute(2, 0, 1)  # 从 (H, W, C) 转换为 (C, H, W)

                    # 记录图像到 TensorBoard
                    # log_writer.add_image('val duration', image_tensor, epoch)
        avg_loss = running_loss / len(val_loader)
        log_writer.add_scalar('val/loss', avg_loss, epoch)  # 在一个 tag 下面添加多个折线图
        return avg_loss


def test(model_pre, radon, test_loader, criterion, rank, log):
    logger.info('load net parameters...')
    # model_pre.load_state_dict(torch.load(os.path.join(args.log_dir, "denoise_pre_weight_best.pth")))
    model_pre.eval()
    running_loss = 0.0

    with torch.no_grad():
        for iteration, (inputs, Y, sino_label, _) in enumerate(test_loader):
            x1, x2 = inputs
            x1, x2 = x1.to(rank), x2.to(rank)
            Y = Y.to(rank).float()

            # sinogram去噪，noise2noise训练
            x1_denoised = x1
            x2_denoised = model_pre(x2)
            # x2_denoised = x2
            # 平均输出的sinogram
            aver_x = x2_denoised
            mid_recon_t = normalization2one(radon.filter_backprojection(aver_x))
            mid_recon_list = mid_recon_t if iteration == 0 else torch.cat([mid_recon_list, mid_recon_t], dim=0)
            label_list = Y if iteration == 0 else torch.cat([label_list, Y], dim=0)
        mid_recon_list = mid_recon_list.squeeze()
        label_list = label_list.squeeze()
        psnr_l, ssim_l = calculate_metrics(mid_recon_list, label_list)
        psnr_avg = psnr_l.mean()
        ssim_avg = ssim_l.mean()
        logger.info(f'psnr: {psnr_avg}, ssim: {ssim_avg}')

        # save the visualized images
        pics = [aver_x, sino_label, mid_recon_t, Y]
        titles = ["sino_recon", 'label_sino', 'pic_recon', 'label']
        # 设置图像整体大小
        fig, axes = plt.subplots(2, 2, figsize=(12, 18))
        fig.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.05, hspace=0.3, wspace=0.2)  # 减少图像间的间距

        # 绘制每个子图
        for i, (pic, title) in enumerate(zip(pics, titles)):
            ax = axes[i // 2, i % 2]  # 计算当前的行列位置
            pic_t = pic[0, 0, :, :].cpu().detach().numpy() if len(pic.shape) == 4 else pic[0, :, :].numpy()
            ax.imshow(pic_t)  # 使用灰度显示图像
            ax.set_title(title, fontsize=10)  # 设置子图标题=
            # 以下代码组合移除坐标轴、刻度和边框
            ax.axis('off')  # 移除边框和坐标轴
            ax.get_xaxis().set_ticks([])  # 移除 x 轴刻度
            ax.get_yaxis().set_ticks([])  # 移除 y 轴刻度
            ax.spines['top'].set_visible(False)  # 隐藏顶部边框
            ax.spines['bottom'].set_visible(False)  # 隐藏底部边框
            ax.spines['left'].set_visible(False)  # 隐藏左侧边框
            ax.spines['right'].set_visible(False)  # 隐藏右侧边框
        # 设置大标题
        fig.suptitle(
            f'psnr: {psnr_avg}, ssim: {ssim_avg}', fontsize=14)
        # 调整布局，避免重叠

        plt.tight_layout(pad=2.0, rect=[0, 0.03, 1, 0.95])
        # os.mkdir(os.path.join(args.log_dir, 'pic_visual')) if not os.path.exists(
        #     os.path.join(args.log_dir, 'pic_visual')) else None
        pic_path = os.path.join(args.log_dir, 'pic_visual', f'test_psnr_{psnr_avg}_ssim_{ssim_avg}_s{seed}.png')
        # pic_path = f'/home/ssddata/linshuijin/PETrecon/log_file/pic_visual/val_{iteration}_{loss_cur:.4f}.png'
        plt.savefig(pic_path, format='png', dpi=300, bbox_inches='tight')  # 300 dpi保证高分辨率
        return psnr_avg, ssim_avg


def main(logger, args, config, log_writer):
    # # 数据
    from model.network_swinTrans import SwinIR

    file_path = os.path.join(args.root_path, f'angular_{args.n_theta}')
    # rank = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rank = torch.device("cpu")
    radon = Radon(n_theta, circle=True, device=rank)

    dataset = DatasetPETRecon(file_path, radon, args.ratio, mode=args.mode, scale_factor=args.scale_factor)
    # radon = radon.to(rank)

    # 将数据集按80/10/10比例划分为训练集、验证集和测试集
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # 使用DataLoader加载数据
    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # 模型初始化

    denoise_model_pre = SwinIR(upscale=1,
                               in_chans=1,
                               img_size=[128, 180],
                               window_size=4,
                               patch_size=[1, 45] if args.aod else [32, 1],
                               img_range=1.0,
                               depths=[2, 6, 2],
                               embed_dim=180,
                               num_heads=[3, 6, 12],
                               mlp_ratio=2.0,
                               upsampler='',
                               resi_connection='1conv', ).to(rank)
    # denoise_model_pre = PETDenoiseNet(device=rank).to(rank)
    # log_writer.add_graph(denoise_model_pre, torch.randn(args.bs, 1, 128, 180).to(rank))
    # denoise_model = PETReconNet(radon, device=rank, config=config).to(rank)
    logger.info(denoise_model_pre)
    # logger.info(denoise_model)
    # log_writer.add_graph(denoise_model, [torch.randn(args.bs, 1, 128, 128).to(rank), torch.randn(args.bs, 1, 128, 180).to(rank), torch.randn(args.bs, 1, 128, 180).to(rank)])

    if args.checkpoints:
        denoise_model_pre.load_state_dict(torch.load(f'./log_file_6034963/denoise_pre_weight_best.pth'))
        logger.info('load pre model...')
        # denoise_model.load_state_dict(torch.load(f'./model/denoise_weight_best.pth'))
    # print(torch.cuda.memory_summary())
    if args.loss == 'L1':
        criterion = nn.L1Loss()
    else:
        criterion = nn.MSELoss()

    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(denoise_model_pre.parameters(), lr=args.lr)

    # 训练
    num_epochs = 20
    val_loss_best = 1
    if not args.test:
        for epoch in range(num_epochs):
            logger.info('start train !')
            logger.info(f'{5 * "*"}Epoch {epoch + 1}/{num_epochs}{5 * "*"}')
            train_loss = train(denoise_model_pre, radon, train_loader, criterion, optimizer, rank, epoch,
                               writer_1)
            logger.info('start validate !')
            val_loss = validate(denoise_model_pre, radon, val_loader, criterion, rank, epoch, writer_1)
            logger.info(f'Epoch {epoch + 1}/{num_epochs} done! Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            writer_1.close()  # 训练结束，不再写入数据，关闭writer
            if val_loss < val_loss_best:
                val_loss_best = val_loss
                torch.save(denoise_model_pre.state_dict(), os.path.join(args.log_dir, "./denoise_pre_weight_best.pth"))
                # torch.save(denoise_model.state_dict(), os.path.join(args.log_dir, "./denoise_weight_best.pth"))
                logger.info(f'Model saved! best in {epoch} for {val_loss:.4f}')
    logger.info('start test !')
    psnr, ssim = test(denoise_model_pre, radon, test_loader, criterion, rank, writer_1)
    logger.info(f'Test Loss: psnr={psnr:.4f}, ssim={ssim:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PET reconstruction demo')
    parser.add_argument('--n_theta', default=180, type=int, help='number of theta')
    parser.add_argument('--bs', default=2, type=int, help='batch_size')
    parser.add_argument('--root_path', default='./simulation_angular/', type=str,
                        help='Input images')
    parser.add_argument('--lr', default=2e-5, type=float, help='learning rate')
    parser.add_argument('--ratio', default=0.2, type=float, help='noise ratio')

    parser.add_argument('--alpha', default=0.2, type=float, help='importance of the smooth of sinogram')
    parser.add_argument('--beta', default=0.4, type=float, help='importance of the delta of pic')
    parser.add_argument('--log_dir', default='./log_file/', type=str,
                        help='Directory for results')
    parser.add_argument('--weights',
                        default='./model/', type=str,
                        help='Path to weights')
    parser.add_argument('--mode', default='none', type=str, help='mode of noise')
    parser.add_argument('--scale_factor', default=0.5, type=float, help='counts level for poisson noise, HD*0.5=LD')
    parser.add_argument('--aod', default=True, type=bool, help='patch of the angular or distance, the former by default')
    parser.add_argument('--CUDA_VISIBLE_DEVICES', default='1', type=str, help='number of CUDA_VISIBLE_DEVICES')
    parser.add_argument('--opt_path', default='./modelSwinUnet/training.yaml', type=str,
                        help='path of SwinUnet preset file')
    parser.add_argument('--loss', default='L1', type=str, help='loss mode, L1 or L2')
    parser.add_argument('--show_tr', default=True, type=bool, help='whether to show results')
    parser.add_argument('--show_val', default=True, type=bool, help='whether to show results')
    parser.add_argument('--checkpoints', default=True, type=bool, help='whether to continue the last training')
    parser.add_argument('--test', default=True, type=bool, help='only for test or not')

    args = parser.parse_args()
    seed = random.randint(0, 10000000)
    preset_seed(seed)
    n_theta = 180
    recon_size = 128
    # 配置logging模块
    args.log_dir = f'./log_file_{seed}'
    log_file_path = f'./log_file_{seed}/log_file_{seed}.txt'
    os.mkdir(args.log_dir) if not os.path.exists(args.log_dir) else None

    # 创建一个logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # 设置日志级别为INFO

    # 创建一个handler，用于将日志写入文件
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)

    # 再创建一个handler，用于将日志输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 定义日志输出格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 将两个handler都添加到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA_VISIBLE_DEVICES
    with open(args.opt_path, 'r') as config:
        opt = yaml.safe_load(config)
    logger.info(f'seed = {seed}')
    logger.info(args)
    logger.info(opt)
    # 初始化 SummaryWriter
    writer_1 = SummaryWriter(log_dir=os.path.join(args.log_dir, f"experiment_{seed}"))  # 可以指定日志路径

    main(logger, args, opt, writer_1)
