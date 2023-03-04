"""
store all function used in registration, including fix, mov, deform space
gain a reg dictionary by process the mov data
"""
import torch
from torch.nn.functional import grid_sample


class DeformerZoo:
    constrain2deformer = {}

    @staticmethod
    def get_deformer_by_constrain(constrain: str):
        return DeformerZoo.constrain2deformer.get(constrain, deform_img_nearest)

    @staticmethod
    def register(*args):
        def inner_register(func):
            for arg in args:
                print(f"add deformer {arg} {func} to deformer zoo")
                DeformerZoo.constrain2deformer[arg] = func
            return func

        return inner_register


@DeformerZoo.register(
    "simi", "tra", "647"
)
def deform_img(mov: dict, deform_space: torch.Tensor) -> dict:
    """
    use the deform space to deform the moving img into registration img
    """
    reg = {"img": grid_sample(mov["img"], deform_space, align_corners=True),
           "img_raw": grid_sample(mov["img_raw"], deform_space, align_corners=True)}
    for k in mov.keys():
        if k not in reg:
            reg[k] = mov[k].clone()
    return reg


@DeformerZoo.register(
    "outline",
    "convex",
    "hpf",
    "hole",
    "cp",
    "csc",
    "bs",
    "cbx",
    "ctx",
    "cb",
    "nn"
)
def deform_img_nearest(mov: dict, deform_space: torch.Tensor) -> dict:
    """
    use the deform space to deform the moving img into registration img
    """

    reg = {"img_raw": grid_sample(mov["img_raw"], deform_space, mode='nearest', align_corners=True),
           "img": grid_sample(mov["img"], deform_space, align_corners=True)}
    for k in mov.keys():
        if k not in reg:
            reg[k] = mov[k].clone()
    return reg
