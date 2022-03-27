import torch
import torch.nn as nn
import torch.nn.functional as F



'''
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

out_channels = F.conv2d(search_feature.view(1, -1, search_feature.shape[-2], search_feature.shape[-1]),
                         weight = template_feature.view(-1, 1, template_feature.shape[-2], template_feature.shape[-1]),
                         stride = 1, padding = 0, groups = batch_size * classes)
print(out_channels)

out_channels_reshape = out_channels.view(batch_size, -1, out_channels.shape[-2], out_channels.shape[-1])
out = outConv(out_channels_reshape)
print(out)
print(outConv.weight)
'''



import PIL.Image as Image
import numpy as np

img = Image.open('C:\\Users\\57168\\Desktop\\Swiss Industrial Area Attisholz\\Attisholz/Attisholz_Flight_01_00003.jpg')
print(img.size)
img_nparray = np.asarray(img)
print(img_nparray.shape)
img_tensor = torch.as_tensor(img_nparray)
print(img_tensor.shape)