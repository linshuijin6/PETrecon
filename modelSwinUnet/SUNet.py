import numpy as np
import torch.nn as nn
from modelSwinUnet.SUNet_detail import SUNet
import torch.nn.functional as F
import torch

class SUNet_model(nn.Module):
    def __init__(self, config):
        super(SUNet_model, self).__init__()
        self.config = config
        self.swin_unet = SUNet(img_size=config['SWINUNET']['IMG_SIZE'],
                               patch_size=config['SWINUNET']['PATCH_SIZE'],
                               in_chans=3,
                               out_chans=1,
                               embed_dim=config['SWINUNET']['EMB_DIM'],
                               depths=config['SWINUNET']['DEPTH_EN'],
                               num_heads=config['SWINUNET']['HEAD_NUM'],
                               window_size=config['SWINUNET']['WIN_SIZE'],
                               mlp_ratio=config['SWINUNET']['MLP_RATIO'],
                               qkv_bias=config['SWINUNET']['QKV_BIAS'],
                               qk_scale=config['SWINUNET']['QK_SCALE'],
                               drop_rate=config['SWINUNET']['DROP_RATE'],
                               drop_path_rate=config['SWINUNET']['DROP_PATH_RATE'],
                               ape=config['SWINUNET']['APE'],
                               patch_norm=config['SWINUNET']['PATCH_NORM'],
                               use_checkpoint=config['SWINUNET']['USE_CHECKPOINTS'])

    def forward(self, x):
        x_size = (x.size(2), x.size(3))
        # x = self.padIn(x)
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = self.swin_unet(x)
        # x = self.padOut(x_size, x)
        return x

    def padIn(self, in_data):
        h, w = in_data.size(2), in_data.size(3)
        full_size = 2 ** np.ceil(np.log2(h))
        pad_h = int(max(0, full_size-h))
        pad_w = int(max(0, full_size-w))
        padding_size = (pad_w//2, pad_w-pad_w//2, pad_h//2, pad_h-pad_h//2)
        padded = F.pad(in_data, padding_size, mode='constant', value=0)
        return padded

    def padOut(self, x_size, in_data):
        h, w = x_size
        ph, pw = in_data.size(2), in_data.size(3)
        cut_size_h = (ph-h)//2
        cut_size_w = (pw-w)//2
        out_data = in_data[:, :, cut_size_h:cut_size_h+h, cut_size_w:cut_size_w+w]
        return out_data






    
if __name__ == '__main__':
    from utils.model_utils import network_parameters
    import torch
    import yaml
    from thop import profile
    from utils.model_utils import network_parameters

    ## Load yaml configuration file
    with open('../training.yaml', 'r') as config:
        opt = yaml.safe_load(config)
    Train = opt['TRAINING']
    OPT = opt['OPTIM']

    height = 256
    width = 256
    x = torch.randn((1, 156, height, width))  # .cuda()
    model = SUNet_model(opt)  # .cuda()
    out = model(x)
    flops, params = profile(model, (x,))
    print(out.size())
    print(flops)
    print(params)
