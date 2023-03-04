import torch
from torch import nn
import torch.nn.functional as F
from math import pi, exp
from ..loss_zoo import LossZoo

@LossZoo.register(("simi", "mi"), ("tra", "mi"), ("cbx", "mi"))
class MILoss(nn.Module):
    def __init__(self, bin_num=40, dropout_number=200000):
        super(MILoss, self).__init__()
        self.dropout_number = dropout_number
        self.bin_num = bin_num
        self.k = 1.0 / 2
        self.sigma = 0.5 / self.k / self.bin_num  ##use 2*k*sigma=1/self.bin_num to calculate sigma
        self.min_number = exp(-self.k ** 2) / (2 * pi * self.sigma * self.sigma)
        self.mu1_list, self.sigma1_list = [], []
        self.mu2_list, self.sigma2_list = [], []
        for i in range(self.bin_num):
            mu1 = (2 * i + 1) / (2 * self.bin_num)
            for j in range(self.bin_num):
                mu2 = (2 * j + 1) / (2 * self.bin_num)
                self.mu1_list.append(mu1)
                self.mu2_list.append(mu2)
                self.sigma1_list.append(self.sigma)
                self.sigma2_list.append(self.sigma)

    def forward(self, fix: dict, mov: dict, reg: dict, deform_space: torch.Tensor):
        batch_x, batch_y = fix["img"], reg["img"]
        b = batch_x.shape[0]
        mu1 = torch.tensor(self.mu1_list, device=batch_x.device)
        mu1 = mu1.view(-1, 1, 1)
        mu2 = torch.tensor(self.mu2_list, device=batch_x.device)
        mu2 = mu2.view(-1, 1, 1)
        sigma1 = torch.tensor(self.sigma1_list, device=batch_x.device)
        sigma1 = sigma1.view(-1, 1, 1)
        sigma2 = torch.tensor(self.sigma2_list, device=batch_x.device)
        sigma2 = sigma2.view(-1, 1, 1)
        loss = 0
        for x, y in zip(batch_x, batch_y):
            x, y = torch.unsqueeze(x, 0), torch.unsqueeze(y, 0)
            if fix.get("ignore") is not None:
                ignore = fix["ignore"].bool()
                x, y = x[~ignore], y[~ignore]
            x, y = x.contiguous().view(1, 1, -1), y.contiguous().view(1, 1, -1)
            rand_index = torch.randint(0, x.shape[2] - 1, (self.dropout_number,))
            drop_out_x = x[:, :, rand_index]
            drop_out_y = y[:, :, rand_index]
            torch.cuda.empty_cache()
            '''codes below are refer to https://matthew-brett.github.io/teaching/mutual_information.html'''
            hgram = self._gauss(drop_out_x, drop_out_y, mu1, mu2, sigma1, sigma2)  # (bin_num**2, b, xyz)
            torch.cuda.empty_cache()
            hgram = hgram - self.min_number
            hgram = F.relu(hgram, inplace=True)
            hgram = torch.sum(hgram, -1)
            hgram = hgram * (2 * pi * self.sigma * self.sigma)
            torch.cuda.empty_cache()
            # print(hgram)
            pxy = hgram / torch.sum(hgram, 0, keepdim=True)
            pxy = pxy.view(self.bin_num, self.bin_num, 1)

            px = torch.sum(pxy, 1, keepdim=True)  # marginal for x over y
            py = torch.sum(pxy, 0, keepdim=True)  # marginal for y over x
            px_py = px * py  # Broadcast to multiply marginals
            # Now we can do the calculation using the pxy, px_py 2D arrays
            torch.cuda.empty_cache()
            loss += -torch.sum(pxy * torch.log(pxy + 1e-9) - pxy * torch.log(px_py + 1e-9))
        loss /= b
        return loss

    def _gauss(self, x, y, mu1, mu2, sigma1, sigma2):
        return 1.0 / (2 * pi * sigma1 * sigma2) * torch.exp(
            -0.5 * ((x - mu1) ** 2 / (sigma1 ** 2) + (y - mu2) ** 2 / (sigma2 ** 2)))