import torch
import torch.nn as nn
from model.XMorpher import XMorpherHead
from tools.zoo.loss_zoo import LossZoo
from tools.zoo.loss.losses import gradient_loss
from tools.zoo.deformer_zoo import DeformerZoo
from tools.utils import get_deform_space


class RegistrationNet(nn.Module):
    def __init__(self, constrain, loss_config, n_channels=1, no_loss=False):
        super().__init__()

        self.constrain = constrain
        self.loss_config = loss_config
        self.xmorpher = XMorpherHead(n_channels=n_channels)
        self.no_loss = no_loss
        self.constrain_loss = dict()
        if not self.no_loss:
            for k, v in self.constrain.items():
                if v:
                    if self.loss_config.get(k) is not None:
                        self.constrain_loss[k] = LossZoo.get_loss_by_constrain_and_type(k, self.loss_config[k])()
                    else:
                        self.constrain_loss[k] = LossZoo.get_loss_by_constrain(k)()

    def forward(self, input: dict):

        flow = self.xmorpher(input["mov"]["simi"]["img"], input["fix"]["simi"]["img"])

        deform_space = get_deform_space(flow)

        output = dict()

        output["reg"] = self._register(input["mov"], deform_space)

        output["loss"] = {}

        for k in input["mov"].keys():
            if self.constrain.get(k, False) and self.no_loss is False:
                output["loss"][k] = self.constrain_loss[k](
                    input["fix"][k], input["mov"][k], output["reg"][k], deform_space
                )
            torch.cuda.empty_cache()

        output["loss"]["gradient_loss"] = gradient_loss(flow)
        torch.cuda.empty_cache()
        return output

    def _register(self, mov_dict, deform_space):
        reg_dict = {}
        for k in mov_dict.keys():
            reg_dict[k] = DeformerZoo.get_deformer_by_constrain(k)(
                mov_dict.get(k),
                deform_space
            )
        return reg_dict
