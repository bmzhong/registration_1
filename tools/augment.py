from torch import rand, pow
import torch
from torch.nn.functional import affine_grid, grid_sample
from random import randint


class Augment():
    def __init__(self, use_deform=True, use_crop=True):
        self.aug_list = [
            self._gamma_correction
        ]
        if use_crop:
            self.aug_list.append(self._random_crop_out)
        if use_deform:
            self.aug_list.append(self._deform)

    def __call__(self, data_dict):
        fix_dict, mov_dict = data_dict["fix"], data_dict["mov"]
        for aug in self.aug_list:
            fix_dict, mov_dict = aug(fix_dict, mov_dict)
        data_dict["fix"], data_dict["mov"] = fix_dict, mov_dict
        return data_dict

    def _gamma_correction(self, fix_dict, mov_dict):
        # gamma = torch.tensor(rand(1) * (2.2 - 1/2.2) + 1/2.2, device=mov_dict["simi"]["img"].device)
        gamma = (torch.rand(1) * (2.2 - 1 / 2.2) + 1 / 2.2).to(mov_dict["simi"]["img"].device)
        if mov_dict.get("simi") is not None:
            mov_dict["simi"]["img"] = pow(mov_dict["simi"]["img"], gamma)
        return fix_dict, mov_dict

    def _deform(self, fix_dict, mov_dict):
        if torch.randn(1) > 0:
            batch = mov_dict["simi"]["img"].shape[0]
            shape = mov_dict["simi"]["img"].shape
            a = torch.tensor([[[1, 0, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 1, 0]]], dtype=torch.float32, device=mov_dict["simi"]["img"].device)
            a += (torch.rand(a.shape, device=a.device) - 0.5) * 0.2  ##0.1决定了形变的大小
            a = a.repeat(batch, 1, 1)

            affine = affine_grid(a, shape)
            for k, v in mov_dict.items():
                for sub_k in ["img", "img2"]:
                    if isinstance(v, dict) and v.get(sub_k) is not None:
                        if k in ["simi", "raw"]:
                            mov_dict[k][sub_k] = grid_sample(mov_dict[k][sub_k], affine, align_corners=True)
                        else:
                            mov_dict[k][sub_k] = grid_sample(mov_dict[k][sub_k], affine, "nearest", align_corners=True)
            return fix_dict, mov_dict
        else:
            return fix_dict, mov_dict

    def _random_crop_out(self, fix_dict, mov_dict):
        length = 10  ##each time a square with side length of 10 is cropped
        if torch.randn(1) > 0:
            mov_img = mov_dict["simi"]["img"]
            x = randint(0, mov_img.shape[2] - length)
            y = randint(0, mov_img.shape[3] - length)
            z = randint(0, mov_img.shape[4] - length)
            mov_img[:, :, x:x + length, y:y + length, z:z + length] = 0
            mov_dict["simi"]["img"] = mov_img
            return fix_dict, mov_dict
        else:
            return fix_dict, mov_dict
