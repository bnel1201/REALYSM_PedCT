import numpy as np
import re
import skimage
from skimage.filters import threshold_otsu
import torch


class globalNormalize(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        sample = sample/torch.max(sample) * 255
        return (sample - torch.mean(sample)) / torch.std(sample)


def read_mhd(filename):
    data = {}
    with open(filename, "r") as f:
        for line in f:
            s = re.search("([a-zA-Z]*) = (.*)", line)
            data[s[1]] = s[2]

            if " " in data[s[1]]:
                data[s[1]] = data[s[1]].split(" ")
                for i in range(len(data[s[1]])):
                    if data[s[1]][i].replace(".", "").replace("-", "").isnumeric():
                        if "." in data[s[1]][i]:
                            data[s[1]][i] = float(data[s[1]][i])
                        else:
                            data[s[1]][i] = int(data[s[1]][i])
            else:
                if data[s[1]].replace(".", "").replace("-", "").isnumeric():
                    if "." in data[s[1]]:
                        data[s[1]] = float(data[s[1]])
                    else:
                        data[s[1]] = int(data[s[1]])
    return data


def crop_image1(image):
    # crop to largest connected component
    # tol  is tolerance
    blur = skimage.filters.gaussian(image, sigma=(5, 5))
    thresh = threshold_otsu(blur)
    img_bw = blur > thresh
    labels = skimage.measure.label(img_bw, return_num=False)

    maxCC_withbcg = labels == np.argmax(np.bincount(labels.flat))
    maxCC_nobcg = labels == np.argmax(np.bincount(labels.flat, weights=img_bw.flat))
    mask = maxCC_nobcg
    bounds = np.ix_(mask.any(1), mask.any(0))
    return image[bounds], bounds

def searchForLine(filename, search_string):
    all_results = []
    for line in open(filename, "r"):
        if re.search(search_string, line):
            all_results.append(line)
    return all_results
