from tqdm import tqdm
import numpy as np
import os
import torch
from shutil import copyfile
import logging
from tools.dataset import Brain
from model.RegistrationNet import RegistrationNet
from tools.utils import *
from tools.visual import *


def evaluate(config, basedir, ckp_path, cfg_path):
    print(f"base dir is {basedir}")
    logging.basicConfig(filename=os.path.join(basedir, "log.txt"), filemode='w', level=logging.INFO)
    copyfile(ckp_path, os.path.join(basedir, "checkpoint", "checkpoint.pth"))
    copyfile(cfg_path, os.path.join(basedir, "config.yaml"))

    eval_dataset = Brain(os.path.join(config["TrainConfig"]["data"], "tot.json"),
                         constrain=config["DataConfig"]["constrain"])
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=1, shuffle=False)

    model = RegistrationNet(constrain=config["ModelConfig"]["backbone"]["constrain"],
                            loss_config=config["LossConfig"]["loss"],
                            n_channels=config["DataConfig"]["n_channels"],
                            scale=config["ModelConfig"]["scale"],
                            median_filter_ksize=config["ModelConfig"]["backbone"]["median_filter_ksize"],
                            max_delta=config["ModelConfig"]["backbone"]["max_delta"],
                            no_loss=True,
                            checkpoint=ckp_path)

    use_gpu = True if len(config["TrainConfig"]["gpu"]) > 0 else False
    print(f'use_gpu: {use_gpu}')
    device = torch.device("cuda:0" if use_gpu else "cpu")
    if use_gpu:
        model.to(device)

    model.eval()

    for input in tqdm(eval_loader):
        fix_name = input["fix"]["name"]
        mov_name = input["mov"]["name"]
        input["fix"].pop("name")
        input["mov"].pop("name")

        with torch.no_grad():

            if use_gpu:
                input = tensor_cuda(input, device)

            output = model(input)

            logging.info(fix_name + mov_name)

            metric = compute_metric(output["reg"], input["fix"])

            for k, v in metric.items():
                logging.info(f"{k} {v}")

            folder_name = fix_name[0] + '_' + mov_name[0]
            # save the fix and moving img after registration
            for k in input["fix"].keys():
                write_img(input["fix"][k], name=folder_name, mode="fix_" + k, basedir=basedir)
                write_img(input["mov"][k], name=folder_name, mode="mov_" + k, basedir=basedir)
                write_img(output["reg"][k], name=folder_name, mode="reg_" + k, basedir=basedir)

            # save the deform space
            if output["reg"].get("space") is not None:
                np.save(os.path.join(basedir, folder_name, "space"), output["reg"]["space"])

        del output, metric
        torch.cuda.empty_cache()
