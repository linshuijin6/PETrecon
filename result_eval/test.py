import sys
import os

import torch
import yaml
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Subset
from evaluate import calculate_metrics, plot_box
from utils.data import DatasetPETRecon
from utils.normalize import normalization2one
from utils.radon import Radon

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '/')))
from train1 import validate
from model.whole_network import PETReconNet, PETDenoiseNet


def eval_test(model_pre, model_recon, radon, val_loader, rank):
    model_pre.eval()
    model_recon.eval()
    sino_recon_list = torch.empty((0, 128, 180)).to(rank)
    pic_recon_list = torch.empty((0, 128, 128)).to(rank)
    sino_label_list = torch.empty((0, 128, 180)).to(rank)
    picLD_list = torch.empty((0, 128, 128)).to(rank)
    picHD_list = torch.empty((0, 128, 128)).to(rank)

    with torch.no_grad():
        for iteration, (inputs, Y, sino_label, picLD) in enumerate(val_loader):
            x1, x2 = inputs
            x1, x2 = x1.to(rank), x2.to(rank)
            Y = Y.to(rank).float().squeeze()
            picLD = picLD.to(rank)
            sino_label = sino_label.to(rank)

            # sinogram去噪，noise2noise训练
            x1_denoised = model_pre(x1)
            x2_denoised = x2
            # x2_denoised = x2
            # 平均输出的sinogram
            aver_x = (x1_denoised + normalization2one(x2_denoised)) / 2
            mid_recon = normalization2one(radon.filter_backprojection(aver_x))

            # PET图去噪
            p_out = model_recon(mid_recon, aver_x, torch.ones_like(aver_x))
            pic_recon = ((p_out + mid_recon) / 2)
            sino_recon = radon(pic_recon).squeeze()
            sino_recon_list = torch.cat([sino_recon_list, sino_recon], dim=0)
            pic_recon_list = torch.cat([pic_recon_list, pic_recon.squeeze()], dim=0)
            picLD_list = torch.cat([picLD_list, picLD.squeeze()], dim=0)
            sino_label_list = torch.cat([sino_label_list, sino_label], dim=0)
            picHD_list = torch.cat([picHD_list, Y], dim=0)

        return pic_recon_list, picLD_list, picHD_list, sino_label_list, sino_recon_list


if __name__ == '__main__':
    with torch.no_grad():
        # 数据导入
        radon_me = Radon(n_theta=180, circle=True, device='cuda')
        test_dataset = DatasetPETRecon(file_path='/home/ssddata/linshuijin/PETrecon/simulation_angular/angular_180',
                                       radon=radon_me, ratio=0.1, name_pre='test_transverse')
        all_in, all_label = test_dataset.get_all_in()
        all_in, all_label = all_in[0:200].to('cuda'), all_label[0:200].to('cuda')
        radon_pic_recon = radon_me.filter_backprojection(all_in)
        test_dataset = Subset(test_dataset, range(200))
        test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
        # 模型导入

        with open('../modelSwinUnet/training.yaml', 'r') as config:
            opt = yaml.safe_load(config)
        device = 'cuda:0'
        model_pre = PETDenoiseNet(device).to(device)
        model_recon = PETReconNet(radon_me, device, opt).to(device)
        model_pre.load_state_dict(torch.load('../model/denoise_pre_weight_0.0946_epoch2.pth'))
        model_recon.load_state_dict(torch.load('../model/denoise_weight_0.0946_epoch2.pth'))
        pic_recon_list, picLD_list, picHD_list, sino_label_list, sino_recon_list = eval_test(model_pre, model_recon, radon_me, test_loader, 'cuda:0')
        pic_me_psnr, pic_me_ssim = calculate_metrics(pic_recon_list, picHD_list)
        pic_radon_psnr, pic_radon_ssim = calculate_metrics(radon_pic_recon.squeeze(), picHD_list)
        for i in range(10):
            plt.imshow(pic_recon_list[i].cpu().numpy()), plt.title('me_recon'), plt.show()
            plt.imshow(radon_pic_recon.squeeze()[i].cpu().numpy()), plt.title('radon_recon'), plt.show()
            plt.imshow(picHD_list.squeeze()[i].cpu().numpy()), plt.title('pic_HD'), plt.show()
            plt.imshow(picLD_list.squeeze()[i].cpu().numpy()), plt.title('pic_LD'), plt.show()
        data_list = [pic_me_psnr, pic_radon_psnr]
        label_list = ['me', 'radon']
        plot_box(data_list, label_list, 'PSNR (dB)', 'PSNR')
        data_list = [pic_me_ssim, pic_radon_ssim]
        plot_box(data_list, label_list, 'SSIM', 'SSIM')
        sino_psnr_l, sino_ssim_l = calculate_metrics(sino_recon_list, sino_recon_list)

