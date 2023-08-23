import util.config as config
import util.util as util
import numpy as np
from PIL import Image
import torch
import os
import util.util_classifier as util_classifier
from util.config import *
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torchvision
import torchvision.transforms as tt
from torchvision.datasets import ImageFolder, DatasetFolder
import timm
import pytorch_lightning as pl
from tqdm import tqdm
from torchvision import transforms
import os
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import WeightedRandomSampler
import sys
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
import util.util as util
import util.util_classifier as util_classifier
import util.util_preprocessing as util_preprocessing
from util.config import *

import time
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
import skimage
from skimage.filters import threshold_otsu


import time
from sklearn.metrics import classification_report

def get_log_name(out, exampleID, logDir='log/'):
#     return ''
    logFile = (
        config.log_dir_testing + logDir
        + str(out).split(" ")[3][2:-2]
        + "."
        + str(out).split(" ")[2].split("-")[0].split(".")[0]
        + "."
        + str(exampleID)
        + ".log"
    )
    return logFile

def get_log_name_test(out, exampleID, logDir='log_victre/'):
    tmp = out.split("_")
#     return ''
    logFile = (
        config.log_dir_testing + logDir
        + tmp[5] + '_' 
        + tmp[6] + '_' 
        + tmp[7].split(".")[0] + '.' 
        + tmp[7].split(".")[1] + '.' 
        + str(exampleID)
        + ".log"
    )
    return logFile

def get_dose(out, logDir='log/'):
    exampleID = 1
    logFile = get_log_name_test(out, exampleID,logDir=logDir)
    with open(logFile, "r") as fp:
        # read all lines in a list
        lines = fp.readlines()
        for line in lines:
            # check if string present on a current line
            if "dose  " in line:
                DOSE = "{:.2e}".format(float(line.strip().split(" ")[2]))
                #                 print('DOSE ' + DOSE)
                break
    return float(line.strip().split(" ")[2])

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

def preprocess_raw_file(filename_mhd):
    filename_raw = filename_mhd.replace(".mhd", ".raw")
    data = util.read_mhd(filename_mhd)
    pixel_array = np.fromfile(filename_raw, dtype="float32").reshape(
        data["NDims"], data["DimSize"][1], data["DimSize"][0]
    )
    tmp = pixel_array[0]
    if 'dense' in filename_mhd:
        bounds_name = 'bounds_dense.npy'
    elif 'fatty' in filename_mhd:
        bounds_name = 'bounds_fatty.npy'
    elif 'hetero' in filename_mhd:
        bounds_name = 'bounds_hetero.npy'
    elif 'scattered' in filename_mhd:
        bounds_name = 'bounds_scattered.npy'
    else:
        raise Exception("Bounds not found!") 
        
    bounds_saved = np.load('/home/niloufar.saharkhiz/code/realysm/image_classification/data/bounds/' + bounds_name,allow_pickle=True)
    tmp = tmp[bounds_saved[0], bounds_saved[1]]
    X = np.std(tmp) * 2
    tmp[tmp < np.mean(tmp) - X] = 0
    tmp[tmp > np.mean(tmp) + X] = X + np.mean(tmp)
#     tmp = pixel_array[0]
#     tmp = tmp[0:1300, :]  # this is only for dense size breast !!! #1300
#     tmp, bounds = crop_image1(tmp)
#     X = np.std(tmp) * 2
#     tmp[tmp < np.mean(tmp) - X] = 0
#     tmp[tmp > np.mean(tmp) + X] = X + np.mean(tmp)
    return tmp

def get_lesion_label(filename_mhd):
    locFile = filename_mhd.replace(".mhd", ".loc")
    if os.path.isfile(locFile):
        lesion_present = 1.0
    else:
        lesion_present = 0.0
    return lesion_present

def evaluate_models_on_log(model_names, FLNAME, test_log_ID_list, DEBUG, NEXAMPLES = 100, dict_dir='', logDir='log/'):
    l_save_dict_name = []
    #print("DEBUG ", DEBUG)
    for best_path in model_names:
        dd_name = get_dict_names([best_path], FLNAME, dict_dir)
#         if os.path.isfile(dd_name[0]):
#             print("save dict exists!")
#             continue

        model = util_classifier.Classifier.load_from_checkpoint(best_path)
        model.eval()
        if CUDA:
            model.cuda()

        # plot images of different doses
#         l_increasing_dose,l_out = get_data_from_out2(FLNAME, NEXAMPLES=NEXAMPLES)
        start_time = time.time()
        l_ba = []
        l_outputs = []
        i = 0
#         if DEBUG:
#             l_out = [l_out[i] for i in [1]]#,6,7,8]]
#             l_out = l_out
#        for out in l_out:
#        for out in FLNAME:
        for i in range (1):
            out = FLNAME
            print("out:", out)
            (
                true_labels0,
                predicted_labels0,
                prob_y0,
                l_images0,
                l_lesion_masks,
                _,
                num_examples,
            ) = process_output(out, model, example_IDs=test_log_ID_list,DOSEID=i, logDir=logDir)

            balanced_accuracy = get_ba(true_labels0, predicted_labels0) # calculate balanced accuracy
            print(str(i) + " " + str(balanced_accuracy))
            l_ba.append(balanced_accuracy)
            doseval = get_dose(out,logDir=logDir)

            d_output = {
                "true_labels": true_labels0,
                "predicted_labels0": predicted_labels0,
                "prob_y0": prob_y0,
                "l_images0": [x.numpy() for x in l_images0],
                "l_lesion_masks": l_lesion_masks,
                "out": out,
                "balanced_accuracy": balanced_accuracy,
                "dose": doseval,
                "mean_time": [],
                "num_examples": num_examples,
            }
            l_outputs.append(d_output)
            i += 1
        print("time: " + str(time.time() - start_time))
        save_dict_name = save_dict(l_outputs, best_path, FLNAME, NEXAMPLES, dict_dir)
        print(save_dict_name)
        l_save_dict_name.append(save_dict_name)
    return l_save_dict_name

def process_output(out, classifier_model,example_IDs, num_examples=50, DOSEID=1,logDir='log/'):
    ############################################################
    ## get accuracy
    l_raw_images0 = []
    l_raw_images = []
    true_labels = []
    l_lesion_masks = []
#     for exampleID in tqdm(range(1, num_examples + 1)):
#     from pdb import set_trace
#     set_trace()
#    consideredRange = range(1, num_examples + 1)
#     if 'P1_5.0_scattered' in out: # skip broken phantom
#         consideredRange = [i for i in range(1, num_examples + 2) if i!=100]
    for exampleID in tqdm(example_IDs):
        logFile = get_log_name_test(out, exampleID,logDir=logDir)

        try:
            flOpen = open(logFile, "r")
            lines = flOpen.readlines()
            flOpen.close()
            for lineID in range(len(lines)):
                if "results directory" in lines[lineID]:
                    result_dir = lines[lineID + 1]
                    break
            filename_mhd = (
                result_dir.strip()
                + "/"
                + str(exampleID)
                + "/projection_DM"
                + str(exampleID)
                + ".mhd"
            )
            #print(filename_mhd)

            save_fileName = filename_mhd.replace(".mhd", ".npy")
            if os.path.isfile(save_fileName):
                tmp = np.load(save_fileName)
            else:
                tmp = preprocess_raw_file(filename_mhd)
                np.save(save_fileName, tmp)
            if FLOAT:
                tmp = tmp
            else:
                tmp = tmp.astype(np.uint8) #DEBUG: float
            tmp = Image.fromarray(tmp)
            l_raw_images.append(tmp)
            lesion_present = get_lesion_label(filename_mhd)
            true_labels.append(lesion_present)
        except:
            continue

    # load images and run through network
    l_images = []
    transform = util_classifier.data_transforms["test"]
    img_id = 0
    for raw_img in l_raw_images:
        image_data = transform(raw_img)
        l_images.append(image_data)
        img_id+=1
        
    tens_images = torch.stack(l_images, dim=0)
    if CUDA:
        tens_images = tens_images.cuda()
    classifier_model.model.eval()
    output = classifier_model.model(tens_images)#.float())
    prob_y_t = classifier_model.sigmoid(output)
    output_sigm_round = torch.round(prob_y_t)
    if CUDA:
        prob_y = prob_y_t.cpu().detach().numpy()[:, 0]
    else:
        prob_y = prob_y_t.detach().numpy()[:, 0]
    predicted_labels = output_sigm_round.data.cpu().numpy().flatten().tolist()

    return (
        true_labels,
        predicted_labels,
        prob_y,
        l_images,
        l_lesion_masks,
        l_raw_images0,
        num_examples,
    )

def get_ba(true_labels0, predicted_labels0):
    report = classification_report(
        true_labels0, predicted_labels0, output_dict=True
    )
    sensitivity = report["1.0"]["recall"]
    specificity = report["0.0"]["recall"]
    balanced_accuracy = (sensitivity + specificity) * 0.5
    return balanced_accuracy


def get_model_nickname(DENSITY,SIZE,DETECTOR,LESIONDENSITY,DOSE):
    nickname = (
    "out_victre_"
    + DENSITY
    + "_spic"
    + SIZE
    + "_id2_"
    + DETECTOR
    + "_"
    + LESIONDENSITY
    + "_"
    + DOSE
    + ".out"
)
    return nickname


def get_source_dirs(dir_training_data,LESIONDENSITY,DENSITY,SIZE,DETECTOR,DOSE):
    sourceDir00 = (
        dir_training_data+"/device_data_VICTREPhantoms_spic_"
        + LESIONDENSITY
        + "/"
    )
    sourceDir0 = sourceDir00 + DOSE + "/" + DENSITY + "/2/" + SIZE + "/" + DETECTOR + "/"
    return sourceDir00,sourceDir0

def get_save_dir(dir_training_data_preprocessed, DENSITY,SIZE,LESIONDENSITY,DOSE,DETECTOR):
    saveDir00 = dir_training_data_preprocessed +"/device_data_VICTREPhantoms_"
    saveDir00 += (
        DENSITY
        + "_spic"
        + SIZE
        + "_id2_"
        + DETECTOR
        + "_"
        + LESIONDENSITY
        + "_"
        + DOSE
        + "_preprocessed1/"
    )
    return saveDir00



def save_dict(l_outputs, mpath, FLNAME, NEXAMPLES, dict_dir):
    l_outputs_to_save = []
    for dic in l_outputs:
        dic1 = {
            "dose": dic["dose"],
            "balanced_accuracy": dic["balanced_accuracy"],
            "true_labels": dic["true_labels"],
            "predicted_labels0": dic["predicted_labels0"],
            "prob_y0": dic["prob_y0"],
            "mpath": mpath,
            "FLNAME": FLNAME,
            "NEXAMPLES": NEXAMPLES,
        }
        l_outputs_to_save.append(dic1)
    saveName = get_saveDict_name_test(mpath, FLNAME, FLOAT, dict_dir)
    print(saveName)
    np.save(saveName, l_outputs_to_save)
    return saveName


def get_mean_std(l_outs_all):
    l_out_0 = l_outs_all[0]
    l_dose0 = [l_out_0[i]["dose"] for i in range(len(l_out_0))]
    l_metric_all = [
        [l_out00[i]["balanced_accuracy"] for i in range(len(l_out00))]
        for l_out00 in l_outs_all
    ]

    l_metric_all_mean = [
        np.mean([l_metric_all[i][j] for i in range(len(l_metric_all))])
        for j in range(len(l_metric_all[0]))
    ]
    l_metric_all_std = [
        np.std([l_metric_all[i][j] for i in range(len(l_metric_all))])
        for j in range(len(l_metric_all[0]))
    ]
    return l_dose0, l_metric_all_mean, l_metric_all_std


def plot_l_outputs(l_outputs):
    i = 0
    for l_out00 in [l_outputs]:  # ,l_outputs_to_save1,l_outputs_to_save2]:
        l_increasing_dose = [l_out00[i]["dose"] for i in range(len(l_out00))]
        l_ba = [l_out00[i]["balanced_accuracy"] for i in range(len(l_out00))]
        plt.semilogx(
            l_increasing_dose, l_ba, base=10, linewidth=0, markersize=4, marker="o"
        )
        i += 1
    plt.axvline(x=7.8e9, color="r", label="optimal dose, dense")
    plt.xlabel("Dose")
    plt.ylabel("Balanced Accuracy")
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        fancybox=True,
        shadow=True,
        ncol=1,
    )

def get_saveDict_name_test(mpath,FLNAME,FLOAT=False, dict_dir=''):
     #"tmp/dicts/"
    tmp = FLNAME.split("_")
    saveName = (
        dict_dir
        + tmp[5] + '_' + tmp[6] + '_' + tmp[4] + '_'+ tmp[0] + '_'+ tmp[7].split(".")[0] + '.' + tmp[7].split(".")[1]
        + "__"
        + "__".join(mpath.replace(".ckpt", "").split("/")[-3:])
        + "__" + 'FLOAT' + str(int(FLOAT))  
        + ".npy"
    )
    return saveName

def get_dict_names(model_names, FLNAME, dict_dir):
    l_save_dict_name = []
    for mpath in model_names:
        saveName = get_saveDict_name_test(mpath,FLNAME,FLOAT=False, dict_dir=dict_dir)
        l_save_dict_name.append(saveName)
    return l_save_dict_name

def get_data_from_out(FLNAME, NEXAMPLES=300):
    flOpen = open(FLNAME + ".out", "r")
    lines = flOpen.readlines()
    flOpen.close()
    l_increasing_dose = []
    l_out = []
    for line_id in range(len(lines)):
        if "DOSE" in lines[line_id] and "DOSENUM" not in lines[line_id]:
            DOSE = float(lines[line_id].split()[1])
            l_increasing_dose.append(DOSE)
            out = lines[line_id + 2]
            l_out.append(out)
    print(len(l_out))
    return l_increasing_dose,l_out 

def get_data_from_out2(FLNAME, NEXAMPLES=300):
    flOpen = open(FLNAME + ".out", "r")
    lines = flOpen.readlines()
    flOpen.close()
    l_increasing_dose = []
    l_out = []
    for line_id in range(len(lines)):
        if "DOSE" in lines[line_id] and "DOSENUM" not in lines[line_id]:
            DOSE = float(lines[line_id].split()[1])
            l_increasing_dose.append(DOSE)
        if "job-array" in lines[line_id]:
            out = lines[line_id]
            l_out.append(out)
    print(len(l_out))
    return l_increasing_dose,l_out


def run_dict_script(l_FLNAME, test_log_ID_list, model_runs, dict_dir, logDir='log/',NEXAMPLES=100):
    model_names_all = []
    for model_runs_name in model_runs:
        flLog = open(model_runs_name, "r")
        lines = flLog.readlines()
        flLog.close()

        #lines_model = [line for line in lines if "Restoring" in line]
        lines_model = [line for line in lines if "Restoring" in line and 'efficientnetb0' in line]
        model_names = [line_model.strip().split(" ")[-1] for line_model in lines_model]
        model_names_all = model_names_all + model_names

        for FLNAME in l_FLNAME:
            if DEBUG:
                model_names_all = model_names_all
            l_save_dict_name = evaluate_models_on_log(model_names_all, 
                                                      FLNAME,
                                                      test_log_ID_list,
                                                      DEBUG=DEBUG, 
                                                      dict_dir=dict_dir,
                                                      logDir=logDir,
                                                      NEXAMPLES=NEXAMPLES)

            