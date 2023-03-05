import torch
from model.XMorpher import XMorpherHead
from tools.zoo.loss_zoo import LossZoo
from tools.zoo.loss.losses import compute_gradient_loss
from tools.zoo.deformer_zoo import DeformerZoo
from model.model_utils import *
from torch.nn.functional import interpolate
from torch import nn, load


class RegistrationNet(nn.Module):
    def __init__(self, constrain, loss_config, n_channels, scale, median_filter_ksize=15, max_delta=0.01,
                 no_loss=False, checkpoint: str = ''):
        super().__init__()

        self.constrain = constrain
        self.loss_config = loss_config
        self.n_channels = n_channels
        self.scale = scale
        self.no_loss = no_loss
        self.xmorpher = XMorpherHead(n_channels=self.n_channels)
        self.median_filter_ksize = median_filter_ksize
        self.max_delta = max_delta

        self.constrain_loss = dict()
        if not self.no_loss:
            for k, v in self.constrain.items():
                if v:
                    if self.loss_config.get(k) is not None:
                        self.constrain_loss[k] = LossZoo.get_loss_by_constrain_and_type(k, self.loss_config[k])()
                    else:
                        self.constrain_loss[k] = LossZoo.get_loss_by_constrain(k)()

        self.downsample_module_list_list = nn.ModuleList([
            nn.ModuleList(
                [nn.Sequential(conv3d_with_leakyReLU(self.n_channels, 16, 3, 2, 1),
                               conv3d_with_leakyReLU(16, self.n_channels, 3, 1, 1))
                 for _ in range(i)]
            ) for i in range(self.scale)
        ])

        self.upsample_module_list_list = nn.ModuleList([
            nn.ModuleList(
                [conv3d_with_leakyReLU(3 + 2 * self.n_channels, 3, 3, 1, 1)
                 for _ in range(i)]
            ) for i in range(self.scale)
        ])

        self.activation = nn.Tanh()

        self.median_blur = self.median_blur_conv(self.median_filter_ksize)

        self._fix()

        if len(checkpoint) > 0:
            self._load(checkpoint)

    def forward(self, input: dict):

        mov, fix = input["mov"]["simi"]["img"], input["fix"]["simi"]["img"]

        mov, mov_scale_feat_list = self.down_sample_data(mov)

        fix, fix_scale_feat_list = self.down_sample_data(fix)

        x = self.xmorpher(mov, fix)

        x = self.up_sample_data(x, mov_scale_feat_list, fix_scale_feat_list)

        x = self.activation(x)

        gradient_loss = compute_gradient_loss(x)

        # x = self.median_blur(x)

        x *= self.max_delta

        x = x.permute(0, 2, 3, 4, 1).contiguous()

        deform_space = get_deform_space(x, input["mov"]["simi"]["img"].shape)

        deform_space = torch.clip(deform_space, -1, 1)

        output = dict()

        output["reg"] = self._register(input["mov"], deform_space)

        output["loss"] = {}

        for k in input["mov"].keys():
            if self.constrain.get(k, False) and self.no_loss is False:
                output["loss"][k] = self.constrain_loss[k](
                    input["fix"][k], input["mov"][k], output["reg"][k], deform_space
                )
            torch.cuda.empty_cache()

        output["loss"]["gradient_loss"] = gradient_loss

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

    def down_sample_data(self, x):
        """
        self.scale    scale_feat_list
        1             [[1]                              ]
        2             [[1/2], [1, 1/2]                  ]
        3             [[1/4], [1/2, 1/4], [1, 1/2, 1/4] ]
        """
        scale_feat_list = []
        for scale in range(self.scale):
            '''down sample'''
            d_sample_feat_list = [x[..., ::1 << (self.scale - scale - 1), ::1 << (self.scale - scale - 1),
                                  ::1 << (self.scale - scale - 1)]]
            d_sample_module_list = self.downsample_module_list_list[scale]
            for i, d_layer in enumerate(d_sample_module_list):
                feat = d_layer(d_sample_feat_list[-1])
                d_sample_feat_list.append(feat)
                torch.cuda.empty_cache()
            scale_feat_list.append(d_sample_feat_list)

        x = scale_feat_list[0][-1]
        for i in range(1, len(scale_feat_list)):
            x = x + scale_feat_list[i][-1]
        torch.cuda.empty_cache()
        return x, scale_feat_list

    def up_sample_data(self, x, mov_scale_feat_list, fix_scale_feat_list):
        output_list = []
        '''up sample'''
        for scale in range(self.scale):
            '''down sample'''
            feat = x
            u_sample_module_list = self.upsample_module_list_list[scale]
            for i, d_module in enumerate(u_sample_module_list):
                feat = torch.cat((feat, fix_scale_feat_list[scale][-i - 1], mov_scale_feat_list[scale][-i - 1]), 1)
                feat = d_module[0](feat)
                feat = interpolate(feat, scale_factor=2, mode='trilinear')
                feat = d_module[1](feat)
                torch.cuda.empty_cache()

            if scale < self.scale - 1:
                feat = interpolate(feat, scale_factor=1 << (self.scale - scale - 1), mode='trilinear')
            output_list.append(feat)
            torch.cuda.empty_cache()

        x = output_list[0]
        for i in range(1, len(output_list)):
            x = x + output_list[i]
        return x

    def median_blur_conv(self, kernel_size):
        conv = nn.Conv3d(3, 3, kernel_size, padding=(kernel_size - 1) // 2, bias=False, groups=3)
        conv.register_parameter(name='weight',
                                param=nn.Parameter(
                                    torch.ones([3, 1, kernel_size, kernel_size, kernel_size]) / (kernel_size ** 3)))
        for param in conv.parameters():
            param.requires_grad = False
        return conv

    def _fix(self):
        if self.scale > 1:
            for name, param in self.xmorpher.named_parameters():
                param.requires_grad = False
            for i in range(self.scale - 1):
                for name, param in self.downsample_module_list_list[i].named_parameters():
                    param.requires_grad = False
                for name, param in self.upsample_module_list_list[i].named_parameters():
                    param.requires_grad = False
        for name, param in self.named_parameters():
            if param.requires_grad is False:
                print(f"RegistrationNet {name} is fixed")
            else:
                print(f"RegistrationNet {name} is not fixed")

    def _load(self, checkpoint):
        cur_dict = self.state_dict()
        if torch.cuda.is_available():
            need_dict = load(checkpoint)["model"]
        else:
            need_dict = load(checkpoint, map_location='cpu')["model"]
        print(need_dict.keys())
        print("the network load from the ", checkpoint)
        for ck, cv in cur_dict.items():
            if need_dict.get("module." + ck) is not None:
                print(ck, " has been loader from ", checkpoint)
                cur_dict[ck] = need_dict["module." + ck]
            elif need_dict.get(ck) is not None:
                print(ck, " has been loader from ", checkpoint)
                cur_dict[ck] = need_dict[ck]
            else:
                print(ck, " random init")
        self.load_state_dict(cur_dict)
