import os
from typing import Dict
from functools import partial

import cv2
import numpy as np
import torch

from ..io import read_tiff_stack


class ReaderZoo:
    constrain2reader = {}

    @staticmethod
    def get_reader_by_constrain(constrain: str):
        return ReaderZoo.constrain2reader.get(constrain, partial(_read_img, type=constrain))

    @staticmethod
    def register(constrain):
        def inner_register(func):
            print(f"add constrain {constrain} {func} to reader zoo")
            ReaderZoo.constrain2reader[constrain] = func
            return func

        return inner_register


@ReaderZoo.register("simi")
def read_simi_img(prefix):
    simi_img = read_tiff_stack(prefix + "_process.tiff")
    simi_img = _gauss(simi_img)
    simi_img = simi_img.astype(np.float32)
    simi_img = simi_img[np.newaxis, ...]
    simi_img = (simi_img - np.min(simi_img)) / (np.max(simi_img) - np.min(simi_img))
    simi = {"img": simi_img}
    if os.path.exists(prefix + "_process_ignore.tiff"):
        simi["ignore"] = read_tiff_stack(prefix + "_process_ignore.tiff")
        simi["ignore"] = simi["ignore"][np.newaxis, ...]

    if os.path.exists(prefix + ".tiff"):
        img_raw = read_tiff_stack(prefix + ".tiff")
    else:
        img_raw = read_tiff_stack(prefix + "_process.tiff")
    img_raw = img_raw[np.newaxis, ...]
    img_raw = img_raw.astype(np.float32)
    minn, maxn = np.min(img_raw), np.max(img_raw)
    img_raw = (img_raw - minn) / (maxn - minn)
    simi["img_raw"] = img_raw
    simi["min"] = minn
    simi["max"] = maxn
    return simi


def _gauss(vol):
    new_vol = vol.copy()
    for i in range(len(new_vol)):
        new_vol[i] = cv2.GaussianBlur(new_vol[i], (3, 3), 0, 0)
    return new_vol


@ReaderZoo.register("tra")
def read_tra_img(prefix):
    tra_img = read_tiff_stack(prefix + "_tra_process.tiff")
    tra_img = tra_img.astype(np.float32)
    tra_img = tra_img[np.newaxis, ...]
    minn, maxn = np.min(tra_img), np.max(tra_img)
    tra_img = (tra_img - minn) / (maxn - minn)

    tra_img = {"img": tra_img}

    if os.path.exists(prefix + "_tra.tiff"):
        img_raw = read_tiff_stack(prefix + "_tra.tiff")
    else:
        img_raw = read_tiff_stack(prefix + "_tra_process.tiff")
    img_raw = img_raw[np.newaxis, ...]
    img_raw = img_raw.astype(np.float32)
    minn, maxn = np.min(img_raw), np.max(img_raw)
    img_raw = (img_raw - minn) / (maxn - minn)
    tra_img["img_raw"] = img_raw
    tra_img["min"] = minn
    tra_img["max"] = maxn
    return tra_img


@ReaderZoo.register("647")
def read_647_img(prefix):
    img = read_tiff_stack(prefix + "_647.tiff")
    img = img.astype(np.float32)
    img = img[np.newaxis, ...]
    minn, maxn = np.min(img), np.max(img)
    img = (img - minn) / (maxn - minn)
    img = {"img": img, "img_raw": img.copy(), "min": minn, "max": maxn}
    return img


@ReaderZoo.register("outline")
def read_outline_img(prefix):
    return _read_img(prefix, "outline")


@ReaderZoo.register("convex")
def read_convex_img(prefix):
    return _read_img(prefix, "convex")


@ReaderZoo.register("hole_pointcloud")
def read_pointcloud_img(prefix):
    pointcloud_img = read_tiff_stack(prefix + "_hole_pointcloud.tiff")
    pointcloud_img = pointcloud_img.astype(np.float32)
    pointcloud_img = pointcloud_img[np.newaxis, ...]
    pointcloud_img /= 255

    hole_img = read_tiff_stack(prefix + "_hole.tiff")
    hole_img = hole_img.astype(np.float32)
    hole_img = hole_img[np.newaxis, ...]
    hole_img /= 255

    pointcloud = {"img": hole_img, "img_raw": hole_img.copy(), "pc": pointcloud_img}
    return pointcloud


@ReaderZoo.register("hpf")
def read_hpf_img(prefix):
    return _read_img(prefix, "hpf")


@ReaderZoo.register("hole_landmark")
def read_hole_landmark(prefix):
    '''
    :param prefix:
    :return:
        {
        img: mask of region。
        landmark: landmark of region，shape: (n, 3)
        }
    '''
    hole_img = read_tiff_stack(prefix + "_hole.tiff")
    hole_img = hole_img.astype(np.float32)
    hole_img = hole_img[np.newaxis, ...]
    hole_img /= 255

    hole_landmark = np.load(prefix + "_hole_landmark.npy").astype(np.float32)

    lm = {"img": hole_img, "img_raw": hole_img.copy(), "landmark": hole_landmark}
    return lm


@ReaderZoo.register("hole")
def read_hole_img(prefix):
    return _read_img(prefix, "hole")


@ReaderZoo.register("cp")
def read_cp_img(prefix):
    return _read_img(prefix, "cp")


@ReaderZoo.register("csc")
def read_csc_img(prefix):
    return _read_img(prefix, "csc")


@ReaderZoo.register("bs")
def read_bs_img(prefix):
    return _read_img(prefix, "bs")


@ReaderZoo.register("cbx")
def read_cbx_img(prefix):
    # return _read_img_intensity(prefix, "cbx")
    return _read_img(prefix, "cbx")


@ReaderZoo.register("ctx")
def read_ctx_img(prefix):
    return _read_img(prefix, "ctx")


@ReaderZoo.register("cb")
def read_cb_img(prefix):
    return _read_img(prefix, "cb")


@ReaderZoo.register("nn")
def read_nn_img(prefix):
    # return _read_img(prefix, "nn")
    nn_img = read_tiff_stack(prefix + "_nn.tiff")
    nn_img = nn_img.astype(np.float32)
    nn_img = nn_img[np.newaxis, ...]
    min_val, max_val = np.min(nn_img), np.max(nn_img)
    nn_img = (nn_img - np.min(nn_img)) / (np.max(nn_img) - np.min(nn_img))
    nn_img = {"img": nn_img, "img_raw": nn_img.copy(), "min": min_val, "max": max_val}
    return nn_img


def _read_img(prefix, type):
    img = read_tiff_stack(prefix + f"_{type}.tiff")
    img = img[np.newaxis, ...]
    label = np.max(img)
    img = img.astype(np.float32)
    img /= np.max(img)
    img_raw = img.copy()
    return {"img": img, "img_raw": img_raw, "label": label}


def _read_img_intensity(prefix, type):
    img = read_tiff_stack(prefix + f"_{type}.tiff")
    img = img[np.newaxis, ...]
    label = np.max(img)
    img = img.astype(np.float32)
    img /= np.max(img)
    img_raw = img.copy()

    if os.path.exists(prefix + "_process.tiff"):
        tra_img = read_tiff_stack(prefix + "_process.tiff")
        tra_img = tra_img.astype(np.float32)
        tra_img = tra_img[np.newaxis, ...]
        tra_img = (tra_img - np.min(tra_img)) / (np.max(tra_img) - np.min(tra_img))
        tra_img[img == 0] = 0
        ignore = np.zeros_like(img, dtype=np.uint8)
        ignore[img == 0] = 1
    else:
        tra_img = img.copy()
        print(f"!!!!! {prefix + '_tra_process.tiff'} not exists")

    return {"img": tra_img, "img_outline": img, "img_raw": img_raw, "label": label, "ignore": ignore}
    # return {"img": img}


def restore_raw_image_from_output(img_dict: Dict[str, torch.Tensor]) -> np.ndarray:
    """
    reader_zoo will normalize the raw origin image, result will store into a dict
    this method reverse a dict back to raw image
    """
    img = img_dict["img_raw"].squeeze()
    img = img.detach().cpu().numpy()
    if img_dict.get("label") is not None:
        img[img > 0.5] = 1
        img[img <= 0.5] = 0
        img = img.astype(np.uint8)
        img *= img_dict["label"].squeeze().cpu().numpy().astype(np.uint8)
    elif img_dict.get("min") is not None:
        minn, maxn = img_dict["min"].squeeze().cpu().numpy(), img_dict["max"].squeeze().cpu().numpy()
        print("!!!!!!!!!!!!", minn, maxn)
        img = img * (maxn - minn) + minn
        if maxn > 255:
            img = img.astype(np.uint16)
        else:
            img = img.astype(np.uint8)
    else:
        img *= 255
        img = img.astype(np.uint8)
    return img

