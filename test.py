import torch


a = torch.tensor([[
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

b = torch.tensor([


                [[[1,1],[1,1]]],[[[2,2],[2,2]]],[[[3,3],[3,3]]],[[[4,4],[4,4]]]


                            ])
print(a.shape)
print(b.shape)
out = torch.nn.functional.conv2d(a, weight = b, stride = 1, groups = b.shape[0])
print(out)
print(out[0].shape)