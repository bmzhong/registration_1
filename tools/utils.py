import torch
import os
import numpy as np
import random
from tools.zoo.deformer_zoo import DeformerZoo
from tools.zoo.metric_zoo import MetricZoo


def set_random_seed(seed=0):
    SEED = seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_optimizer(type: str, params: dict, model):
    """
    :param type: classifier
    :param params: parameters included
    :param model: Model that this optimizer is used
    :return:
    """
    return getattr(torch.optim, type, None)(params=model.parameters(), **params)


def compute_metric(reg: dict, fix: dict):
    metric_result_dict = {}
    for k in reg.keys():
        metric_calculator = MetricZoo.get_metric_by_constrain(k)
        if metric_calculator is not None:
            metric_result_dict[k] = metric_calculator(fix[k], reg[k])
    return metric_result_dict


def update_dict(tot_dict: dict, cur_dict: dict):
    """
    update the element of dictionary
    """
    for k in cur_dict.keys():
        if tot_dict.get(k) is None:
            if isinstance(cur_dict[k], torch.Tensor):
                tot_dict[k] = [cur_dict[k].cpu().detach().numpy()]
            else:
                tot_dict[k] = [cur_dict[k]]
        else:
            if isinstance(cur_dict[k], torch.Tensor):
                tot_dict[k].append(cur_dict[k].cpu().detach().numpy())
            else:
                tot_dict[k].append(cur_dict[k])


def average_loss(output):
    """
    calculate the average of the network output losses
    :param output:
    :return:
    """
    for k in output["loss"].keys():
        output["loss"][k] = torch.mean(output["loss"][k])
    return output


def get_deform_space(flow):
    shape = flow.shape[2:]

    vectors = [torch.arange(0, s) for s in shape]
    grids = torch.meshgrid(vectors)
    grid = torch.stack(grids)  # y, x, z
    grid = torch.unsqueeze(grid, 0)  # add batch
    grid = grid.type(torch.FloatTensor)
    grid = grid.to(flow.device)

    new_locs = grid + flow

    for i in range(len(shape)):
        new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

    if len(shape) == 2:
        new_locs = new_locs.permute(0, 2, 3, 1)
        new_locs = new_locs[..., [1, 0]]
    elif len(shape) == 3:
        new_locs = new_locs.permute(0, 2, 3, 4, 1)
        new_locs = new_locs[..., [2, 1, 0]]
    return new_locs


def fusion_loss(output: dict, loss_cfg, constrain_dict: dict):
    '''
    :param output: loss dict gain from network
    :param loss_cfg: loss config
    :param constrain_dict: which constrains are used
    :return:
    '''
    loss = 0.
    if loss_cfg["factor"].get("use_factor", True):
        for name, val in loss_cfg["factor"].items():
            if output.get(name) is not None:
                loss += val * output[name]

    for k, v in constrain_dict.items():
        if v:
            loss += output.get(k, 0) * loss_cfg["constrain"].get(k, 1)
            torch.cuda.empty_cache()
    return loss


def tensor_cuda(data, device):
    dfs_cuda(data,device)
    return data


def tensor_cpu(data):
    dfs_cpu(data)
    return data


def dfs_cuda(data, device):
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = data[k].to(device)
            else:
                dfs_cuda(data[k], device)


def dfs_cpu(data):
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = data[k].detach().cpu()
            else:
                dfs_cpu(data[k])


class ModelSaver:
    def __init__(self, max_save_num):
        """
        :param max_save_num: max checkpoint number to save
        """
        self.save_path_list = []
        self.max_save_num = max_save_num

    def save(self, path, state_dict):
        self.save_path_list.append(path)
        if len(self.save_path_list) > self.max_save_num:
            top = self.save_path_list.pop(0)
            os.remove(top)
        torch.save(state_dict, path)
