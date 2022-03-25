import torch
import torch.nn as nn
import torch.nn.functional as F




a = torch.FloatTensor([
                    [
                    [[1,1,1],
                    [1,1,1],
                    [1,1,1]],

                    [[2,2,2],
                    [2,2,2],
                    [2,2,2]],

                    [[3,3,3],
                    [3,3,3],
                    [3,3,3]],

                    [[4,4,4],
                    [4,4,4],
                    [4,4,4]]
                            ],
                    [
                    [[1,1,1],
                    [1,1,1],
                    [1,1,1]],

                    [[2,2,2],
                    [2,2,2],
                    [2,2,2]],

                    [[3,3,3],
                    [3,3,3],
                    [3,3,3]],

                    [[4,4,4],
                    [4,4,4],
                    [4,4,4]]
                            ]])


b = torch.FloatTensor([
                    [
                    [[1,1],
                    [1,1]],

                    [[2,2],
                    [2,2]],

                    [[3,3],
                    [3,3]],

                    [[4,4],
                    [4,4]]
                            ],
                    [
                    [[1,1],
                    [1,1]],

                    [[2,2],
                    [2,2]],

                    [[3,3],
                    [3,3]],

                    [[4,4],
                    [4,4]]
                            ]])

search_feature = a
template_feature = b
classes = 4
outConv = nn.Conv2d(classes, 1, kernel_size = 1)
batch_size = 2

out_channels = F.conv2d(search_feature.view(1, -1, search_feature.shape[-2], search_feature.shape[-1]).contiguous(),
                         weight = template_feature.view(-1, 1, template_feature.shape[-2], template_feature.shape[-1]).contiguous(),
                         stride = 1, padding = 0, groups = batch_size * classes)
print(out_channels)
print(out_channels.view(batch_size, -1, out_channels.shape[-2], out_channels.shape[-1]))
out = outConv(out_channels.view(batch_size, -1, out_channels.shape[-2], out_channels.shape[-1]))
print(out)