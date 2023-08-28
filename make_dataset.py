from pathlib import Path
import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt
from argparse import ArgumentParser

import pydicom
from pydicom.data import get_testdata_file
from pydicom import dcmread, dcmwrite
import cv2

from scipy.ndimage import zoom
from scipy.ndimage import affine_transform
from mpl_toolkits.axes_grid1 import make_axes_locatable


def lesion_present(lesion_name): return Path(str(lesion_name)).exists()


def load_volume(path, shape=(512, 512), dtype='float32'):
    input_image = np.fromfile(path, dtype=dtype)
    input_image = input_image.reshape(shape)
    return(input_image)


def normalize(image): return 1 - (image.max() - image) / (image.max() - image.min())


def get_ground_truth(patient):
    slice_id = f'{patient.slice:03d}'
    patient_name = Path(patient.results_name).parent.stem
    ground_truth_name = Path(patient.results_name).parents[2] / 'phantoms' / patient_name / slice_id / f'ground_truth_{slice_id}.dcm'
    return pydicom.read_file(ground_truth_name).pixel_array - 1024


def get_segmentation_mask(patient):
    slice_id = patient.slice
    patient_name = Path(patient.results_name).parent.stem
    seg_file = Path(patient.results_name).parent / f'{patient_name}_{slice_id}_segmentation_labels.dcm'
    if seg_file.exists():
        return pydicom.read_file(seg_file).pixel_array - 1024
    else:
        return None


def get_lesion_mask(patient):
    if patient['lesion present']:
        return pydicom.read_file(patient.lesion_name).pixel_array[::2, ::2] - 1024
    else:
        return None


def get_registration_transform(moving_img, ref_img):
    #from https://stackoverflow.com/questions/72907431/opencv-error-215assertion-failed-m0-type-cv-32f-m0-type-cv-64
    # Open the image files.
    img1 = (255*normalize(moving_img)).astype('uint8')  # Image to be aligned.
    img2 = (255*normalize(ref_img)).astype('uint8')   # Reference image.
    height, width = img2.shape

    # Create ORB detector with 5000 features.
    ## used to creates keypoints on the reference image
    orb_detector = cv2.ORB_create(5000)

    # Find keypoints and descriptors.
    # The first arg is the image, second arg is the mask
    #  (which is not required in this case).
    kp1, d1 = orb_detector.detectAndCompute(img1, None)
    kp2, d2 = orb_detector.detectAndCompute(img2, None)

    # Match features between the two images.
    # We create a Brute Force matcher with
    # Hamming distance as measurement mode.

    #Brute-Force matcher is simple. 
    #It takes the descriptor of one feature in first set and is matched with all other features in second set using some distance calculation. And the closest one is returned.
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

    # Match the two sets of descriptors.
    matches = matcher.match(d1, d2)

    # Sort matches on the basis of their Hamming distance.
    matches = sorted(matches, key = lambda x:x.distance, reverse=False)

    # Take the top 90 % matches forward.
    matches = matches[:int(len(matches)*0.9)]
    no_of_matches = len(matches)

    # Define empty matrices of shape no_of_matches * 2.
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))

    for i in range(len(matches)):
        p1[i, :] = kp1[matches[i].queryIdx].pt
        p2[i, :] = kp2[matches[i].trainIdx].pt

    # Find the homography matrix.
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)
    return homography


def get_registered_ground_truths(patient):      
    ground_truth_image = get_ground_truth(patient)[::2, ::2]
    ct_image = load_volume(patient.results_name, shape=(512, 512))
    transform = get_registration_transform(ground_truth_image, ct_image)
    
    ground_truth_transformed = cv2.warpPerspective(ground_truth_image, transform, ct_image.shape)
      
    seg_mask = get_segmentation_mask(patient)
    if seg_mask is not None:
        seg_mask = seg_mask[::2,::2]
        seg_mask_transformed = cv2.warpPerspective(seg_mask, transform, ct_image.shape)
    else:
        seg_mask_transformed = None

    lesion_mask = get_lesion_mask(patient)
    lesion_mask_transformed = lesion_mask
    if lesion_mask is not None:
        lesion_mask_transformed = cv2.warpPerspective(lesion_mask, transform, ct_image.shape)
    
    return ct_image, ground_truth_transformed, seg_mask_transformed, lesion_mask_transformed


def norm(x,y):
    diff = x.ravel() - y.ravel()
    return np.sqrt(np.dot(diff, diff.T)/x.size)


def convert_to_dicom(img_slice, phantom_path):
    fpath = get_testdata_file("CT_small.dcm")
    ds = dcmread(fpath)
    img_slice = img_slice.astype('int16')
    ds.Rows, ds.Columns = img_slice.shape
    ds.PixelData = img_slice.tobytes()
    dcmwrite(phantom_path, ds)


class ImageGetter():
    
    def __init__(self, dataset_dir):
        
        self.dataset_dir = Path(dataset_dir)
        self.image_fnames = list((self.dataset_dir / 'images').glob('image_*.dcm'))
    
    def __getitem__(self, index):
        idx = int(self.image_fnames[index].stem.split('_')[1].split('.dcm')[0])
        ct_image = dcmread(self.dataset_dir / 'images' / f'image_{idx:05d}.dcm').pixel_array
        
        ground_truth = dcmread(self.dataset_dir / 'ground_truth' / f'truth_{idx:05d}.dcm').pixel_array
        
        segmentation = dcmread(self.dataset_dir / 'segmentation_masks' / f'segmentation_{idx:05d}.dcm').pixel_array

        lesion_fname = self.dataset_dir / 'lesion_masks' / f'lesion_{idx:05d}.dcm'
        lesion_mask = dcmread(lesion_fname).pixel_array if lesion_fname.exists() else None
        return ct_image, ground_truth, segmentation, lesion_mask

    def __len__(self):
        return len(self.image_fnames)


def make_dataset(dataset_dir, summary, max_images=4000, max_lesions=2000, rmse_reject_threshold = 200):
    assert max_images > max_lesions

    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(exist_ok=True, parents=True)
    summary.to_csv(dataset_dir / 'summary.csv')

    image_dir = dataset_dir / 'images'
    image_dir.mkdir(exist_ok=True)
    ground_truth_dir = dataset_dir / 'ground_truth'
    ground_truth_dir.mkdir(exist_ok=True)
    lesion_dir = dataset_dir / 'lesion_masks'
    lesion_dir.mkdir(exist_ok=True)
    segmentation_dir = dataset_dir / 'segmentation_masks'
    segmentation_dir.mkdir(exist_ok=True)
    
    print(f'writing dataset of {max_images} to: {dataset_dir} with subdirectories:')
    print(image_dir.stem)
    print(ground_truth_dir.stem)
    print(lesion_dir.stem)
    print(segmentation_dir.stem)
    print('...')

    image_count = 0
    lesion_count = 0
    idx = 0
    while image_count < max_images:
        patient = summary.iloc[idx]
        ct_image, ground_truth_transformed, seg_mask_transformed, lesion_mask_transformed = get_registered_ground_truths(patient) 
        idx += 1
        if seg_mask_transformed is None:
            continue
        rmse = norm(ct_image, ground_truth_transformed)
        if rmse > rmse_reject_threshold:
            continue
        
        ct_fname = image_dir / f'image_{idx:05d}.dcm'
        convert_to_dicom(ct_image, ct_fname)

        gt_fname = ground_truth_dir / f'truth_{idx:05d}.dcm'
        convert_to_dicom(ground_truth_transformed, gt_fname)
        image_count += 1

        if (lesion_mask_transformed is not None) & (lesion_count < max_lesions):
                lesion_fname = lesion_dir / f'lesion_{idx:05d}.dcm'
                convert_to_dicom(lesion_mask_transformed, lesion_fname)
                lesion_count += 1

        seg_fname = segmentation_dir / f'segmentation_{idx:05d}.dcm'
        convert_to_dicom(seg_mask_transformed, seg_fname)
        
        print_freq = 100 if max_images > 100 else 1
        if image_count % (max_images//print_freq) == 0:
            print(f'images: {image_count}, lesions: {lesion_count}')

    
def plot_patient_images(ct_image, ground_truth, seg_mask=None, lesion_mask=None):
    if lesion_mask is not None:
        f, axs = plt.subplots(2, 2, figsize=(6, 6), dpi=150)
        axs = axs.flatten()
    else:
        f, axs = plt.subplots(1, 3, figsize=(8, 4), dpi=150)
    axs[0].imshow(ct_image, cmap='gray')
    axs[0].set_title('CT Image')
    axs[1].imshow(ground_truth, cmap='gray')
    axs[1].set_title('Phantom Image')
    if seg_mask is not None:
        im = axs[2].imshow(seg_mask)
        axs[2].set_title('Segmentation Mask')
        divider = make_axes_locatable(axs[2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
    list(map(lambda ax: ax.set_axis_off(), axs))
    if lesion_mask is not None:
        axs[3].imshow(lesion_mask)
        axs[3].set_title('Lesion Mask')


if __name__ == '__main__':
    parser = ArgumentParser(description='Program to organize XCIST simulations into a dataset')
    parser.add_argument('base_directory', nargs='?', default="", help='directory containing original XCIST simulations images to be processed')
    parser.add_argument('-o', '--output_directory', type=str, required=False, default="", help='directory to dataset to')

    args = parser.parse_args()
    dirP = args.base_directory or Path('/projects01/didsr-aiml/brandon.nelson/XCAT_body/peds_abdomens_08-25-2023_00-10/')
    dataset_dir = args.output_directory or Path('/projects01/didsr-aiml/brandon.nelson/XCAT_body/segmentation_dataset')

    csv_file = dirP/'summary.csv'
    if csv_file.exists():
        summary = pd.read_csv(dirP/'summary.csv')
    else:
        summary = pd.concat([pd.read_csv(f) for f in dirP.glob('*.csv')]).reset_index()
        summary['lesion present'] = list(map(lesion_present, summary.lesion_name))
        summary.to_csv(dirP/'summary.csv', index=False)
        summary.to_csv(csv_file, index=False)
    
    make_dataset(dataset_dir, summary, max_images=4000, max_lesions=2000)
