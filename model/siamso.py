""" Full assembly of the parts to form the complete network """
import torch.nn.functional as F
import torch.nn as nn

from .unet.unet_model import UNet


class SiamSO(nn.Module):
    def __init__(self, n_channels, classes, device, bilinear=False):
        super(SiamSO, self).__init__()
        self.bilinear = bilinear
        self.classes = classes
        self.unet = UNet(n_channels, classes, bilinear).to(device = device)
        self.outConv = nn.Conv2d(classes, 1, kernel_size = 1)

    def forward(self, batch):
        batch_size = batch['template'].shape[0]
        # print(batch_size)
        template_feature = self.unet(batch['template'])
        search_feature = self.unet(batch['search'])
        # print(template_feature.shape)
        # print(search_feature.shape)
        out_channels = F.conv2d(search_feature.view(1, -1, search_feature.shape[-2], search_feature.shape[-1]),
                         weight = template_feature.view(-1, 1, template_feature.shape[-2], template_feature.shape[-1]),
                         stride = 1, padding = 0, groups = batch_size * self.classes)
        # print(out.shape)
        out = self.outConv(out_channels.view(batch_size, -1, out_channels.shape[-2], out_channels.shape[-1]))

        return out
