""" Full assembly of the parts to form the complete network """
import torch.nn.functional as F
import torch.nn as nn

from .unet.unet_model import UNet


class SiamSO(nn.Module):
    def __init__(self, n_channels, bilinear=False):
        super(SiamSO, self).__init__()
        self.bilinear = bilinear
        self.unet = UNet(n_channels, 1, bilinear)

    def forward(self, batch):
        template_feature = self.unet(batch['template'])
        search_feature = self.unet(batch['search']).squeeze(1)
        # print(template_feature.shape)
        # print(search_feature.shape)
        out = F.conv2d(search_feature, weight = template_feature, stride = 1, padding = 0, groups = template_feature.shape[0])
        # print(out.shape)
        return out
