import os
from shutil import copyfile
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
from tools.utils import *
from tools.visual import *
from tools.dataset import Brain
from tools.zoo.loss_zoo import LossZoo
from model.RegistrationNet import RegistrationNet
from torch import nn
import torch
from tools.augment import Augment


def train_one_iterator(input, model, optimizer, model_saver, config, step, augment, writer, train_loss, train_metric,
                       basedir, use_gpu, device):
    fix_name = input["fix"]["name"]
    mov_name = input["mov"]["name"]
    input["fix"].pop("name")
    input["mov"].pop("name")

    input = augment(input)
    torch.cuda.empty_cache()

    optimizer.zero_grad()

    if use_gpu:
        input = tensor_cuda(input, device)

    output = model(input)
    output = average_loss(output)
    output["loss"]["total_loss"] = fusion_loss(output["loss"], config["LossConfig"],
                                               config["DataConfig"]["constrain"])
    metric = compute_metric(output["reg"], input["fix"])

    output["loss"]["total_loss"].backward()

    optimizer.step()

    update_dict(train_loss, output["loss"])
    update_dict(train_metric, metric)

    if step % 1 == 0:
        for k in input["fix"].keys():
            visual_img("train/" + k, fix_name[0] + "_" + mov_name[0], input["fix"][k]["img_raw"][0][0],
                       input["mov"][k]["img_raw"][0][0],
                       [output["reg"][k]["img_raw"][0][0]], writer, step)
        visual_gradient(model, writer, step)

        model_saver.save(os.path.join(basedir, "checkpoint", str(step).zfill(4) + ".pth"),
                         {"model": model.state_dict(), "optim": optimizer.state_dict()})

    del input, output

    torch.cuda.empty_cache()


def evaluate_one_iterator(input, model, config, step, writer, val_loss, val_metric, use_gpu, device):
    fix_name = input["fix"]["name"]
    mov_name = input["mov"]["name"]
    input["fix"].pop("name")
    input["mov"].pop("name")

    if use_gpu:
        input = tensor_cuda(input, device)

    torch.cuda.empty_cache()

    with torch.no_grad():
        output = model(input)

        output = average_loss(output)
        output["loss"]["total_loss"] = fusion_loss(output["loss"], config["LossConfig"],
                                                   config["DataConfig"]["constrain"])
        metric = compute_metric(output["reg"], input["fix"])

        update_dict(val_loss, output["loss"])

        update_dict(val_metric, metric)

    for k in input["fix"].keys():
        visual_img("eval/" + k, fix_name[0] + "_" + mov_name[0], input["fix"][k]["img_raw"][0][0],
                   input["mov"][k]["img_raw"][0][0],
                   [output["reg"][k]["img_raw"][0][0]], writer, step)

    del input, output
    torch.cuda.empty_cache()


def train(config, basedir, config_path):
    copyfile(config_path, os.path.join(basedir, "config.yaml"))

    set_random_seed(seed=0)

    print(f"base dir is {basedir}")

    writer = SummaryWriter(log_dir=os.path.join(basedir, "logs"))

    model_saver = ModelSaver(config["TrainConfig"].get("max_save_num", 10))

    train_dataset = Brain(os.path.join(config["TrainConfig"]["data"], "train.json"),
                          constrain=config["DataConfig"]["constrain"])

    val_dataset = Brain(os.path.join(config["TrainConfig"]["data"], "test.json"),
                        constrain=config["DataConfig"]["constrain"])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config["TrainConfig"]["batch"],
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config["TrainConfig"]["batch"],
                                             shuffle=True)
    augment = Augment(config["DataConfig"]["use_deform"],
                      config["DataConfig"]["use_crop"])

    model = RegistrationNet(constrain=config["ModelConfig"]["backbone"]["constrain"],
                            loss_config=config["LossConfig"]["loss"],
                            n_channels=config["DataConfig"]["n_channels"])

    use_gpu = True if len(config["TrainConfig"]["gpu"]) > 0 else False
    print(f'use_gpu: {use_gpu}')
    device = torch.device("cuda:0" if use_gpu else "cpu")

    if use_gpu:
        gpu_num = len(config["TrainConfig"]["gpu"].split(","))
        if gpu_num > 1:
            model = torch.nn.DataParallel(
                model, device_ids=[i for i in range(gpu_num)])
        model.to(device)

    optimizer = get_optimizer(type=config["OptimConfig"]["backbone"]["optimizer"]["type"],
                              params=config["OptimConfig"]["backbone"]["optimizer"]["params"],
                              model=model)
    step = 0
    for epoch in range(1, config["TrainConfig"]["epoch"] + 1):
        model.train()
        train_loss = {}
        train_metric = {}
        for input in tqdm(train_loader):
            train_one_iterator(input, model, optimizer, model_saver, config, step, augment, writer, train_loss,
                               train_metric, basedir, use_gpu, device)
            step += 1
            del input
            torch.cuda.empty_cache()

        print(
            f"epoch: {epoch} train loss: {np.mean(train_loss['total_loss'])}")
        for k, v in train_metric.items():
            print(f"metric: {k}: {np.mean(v)}")
        for k, v in train_loss.items():
            writer.add_scalar("train/loss/" + k, np.mean(v), step)
        for k, v in train_metric.items():
            writer.add_scalar("train/metric/" + k, np.mean(v), step)

        if epoch % 10 == 0:
            val_loss = {}
            val_metric = {}
            model.eval()
            for input in tqdm(val_loader):
                evaluate_one_iterator(
                    input, model, config, step, writer, val_loss, val_metric, use_gpu, device)
                del input
                torch.cuda.empty_cache()

            print(
                f"epoch: {epoch} eval loss: {np.mean(val_loss['total_loss'])}")
            for k, v in val_metric.items():
                print(f"eval metric: {k}: {np.mean(v)}")
            for k, v in val_loss.items():
                writer.add_scalar("eval/loss/" + k, np.mean(v), step)
            for k, v in val_metric.items():
                writer.add_scalar("eval/metric/" + k, np.mean(v), step)
