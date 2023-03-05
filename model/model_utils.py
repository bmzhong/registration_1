from torch import nn
import torch
from torch.nn.functional import affine_grid


def conv3d_with_leakyReLU(*args):
    return nn.Sequential(nn.Conv3d(*args),
                         nn.LeakyReLU())


def get_deform_space(flow, image_shape):
    base_transform = generate_base_deform_space(image_shape, flow.device)
    deform_space = flow + base_transform
    return deform_space


def generate_base_deform_space(shape, device):
    """
    generate a base deform space to calculate points' offset
    """
    a = torch.tensor([[[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0]]], dtype=torch.float32, device=device)
    a = a.repeat(shape[0], 1, 1)
    affine = affine_grid(a, shape)
    return affine


# def get_deform_space_1(flow):
#     shape = flow.shape[2:]
#
#     vectors = [torch.arange(0, s) for s in shape]
#     grids = torch.meshgrid(vectors)
#     grid = torch.stack(grids)  # y, x, z
#     grid = torch.unsqueeze(grid, 0)  # add batch
#     grid = grid.type(torch.FloatTensor)
#     grid = grid.to(flow.device)
#
#     new_locs = grid + flow
#
#     for i in range(len(shape)):
#         new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
#
#     if len(shape) == 2:
#         new_locs = new_locs.permute(0, 2, 3, 1)
#         new_locs = new_locs[..., [1, 0]]
#     elif len(shape) == 3:
#         new_locs = new_locs.permute(0, 2, 3, 4, 1)
#         new_locs = new_locs[..., [2, 1, 0]]
#     return new_locs

