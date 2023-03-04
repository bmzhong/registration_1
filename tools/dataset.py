from torch.utils.data.dataset import Dataset
import json
import os
from random import randint, shuffle
from copy import deepcopy
import torch


from tools.zoo.reader_zoo import ReaderZoo


class Brain(Dataset):
    def __init__(self,
                 data_json: str,
                 constrain: dict):
        """
        :param data_json: json path that contains data info
        :param constrain: such as
                        {
                              simi: True
                              outline: True
                              convex: False
                              hole_pointcloud: True
                        }
        """
        super(Brain, self).__init__()
        with open(data_json) as f:
            content = json.load(f)
        moving_list = content["moving"]
        fix_list = content["fix"]
        self.constrain = constrain
        self.reader = {}
        for k, v in self.constrain.items():
            if v:
                self.reader[k] = ReaderZoo.get_reader_by_constrain(k)
        self.moving_prefix_list = [os.path.join(i, os.path.split(i)[1])
                                  for i in moving_list]
        self.fix_prefix_list = [os.path.join(i, os.path.split(i)[1])
                               for i in fix_list]

    def __getitem__(self, index):
        """
        :param index:
        :return: dict: {
                    fix: {
                        name:  brain name of fixed image
                        simi (processed fixed image）：{
                            img: image data
                        }
                        outline (mask of outline): {
                            img: data of outline
                        }
                        hole (mask of ventricle) : {
                            img: data of ventricle
                        }
                        ...
                    }
                    mov: {
                        name:  brain name of moving image
                        simi (processed moving image）：{
                            img: image data
                        }
                        outline (marked outline of moving image): {
                            img: data of outline
                        }
                        hole (marked ventricle of moving image) : {
                            img: data of ventricle
                        }
                        ...
                    }
                }
        """

        mov_prefix = self.moving_prefix_list[index]
        fix_prefix = self.fix_prefix_list[index]
        output = {"fix": {"name": os.path.split(fix_prefix)[1]},
                  "mov": {"name": os.path.split(mov_prefix)[1]}}
        for k, v in self.reader.items():
            output["fix"][k] = self.reader[k](fix_prefix)
            output["mov"][k] = self.reader[k](mov_prefix)
        return output

    def __len__(self):
        return len(self.moving_prefix_list)