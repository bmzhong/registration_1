from PIL import Image
import numpy as np


def read_tiff_stack(path):
    img = Image.open(path)
    images = []
    for i in range(img.n_frames):
        img.seek(i)
        slice = np.array(img)
        images.append(slice)

    return np.array(images)
