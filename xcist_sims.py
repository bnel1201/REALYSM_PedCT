# %%
from pathlib import Path
import os
from glob import glob

import numpy as np
from pydicom.data import get_testdata_file
from pydicom import dcmread, dcmwrite
import gecatsim as xc
import pandas as pd
import pydicom

from lesions import add_random_circle_lesion
from XCIST.gecatsim.reconstruction.pyfiles import recon
import DICOM_to_voxelized_phantom


def load_volume(path, shape=(501, 1024, 1024), dtype='float32'):
  input_image = np.fromfile(path, dtype=dtype)
  input_image = input_image.reshape(shape)
  return(input_image)


def get_attenuation_coefficients(logfile):
    "returns attenuation coefficients (1/pixel) from xcat logfile as a dict"
    with open(logfile) as f:
        text = f.read()
    atten_coeffs = text.split('Linear Attenuation Coefficients (1/pixel):\n')[1].split('\n\n')[0].split('\n')
    atten_coeffs_dict = {ac.split('=')[0].strip(): float(ac.split('=')[1]) for ac in atten_coeffs}
    atten_coeffs_dict['Air'] = 0 # add manually because its assumed in the log file and not included
    return atten_coeffs_dict


def mu_to_HU(vol, mu_water): return 1000*(vol - mu_water)/mu_water


def convert_to_dicom(img_slice, phantom_path):
    fpath = get_testdata_file("CT_small.dcm")
    ds = dcmread(fpath)

    img_slice += 1024 #check this <-- I shouldnt need this bias
    img_slice = img_slice.astype('int16')
    ds.Rows, ds.Columns = img_slice.shape
    ds.PixelData = img_slice.tobytes()
    dcmwrite(phantom_path, ds)


def voxelize_ground_truth(dicom_path, save_path, material_threshold_dict=None):
   """
   Used to convert ground truth image into segmented volumes used by XCIST to run simulations
   
   Inputs:
   phantom_path [str]: directory containing ground truth dicom images, these are typically the output of `convert_to_dicom`
   material_threshold_dict [dict]: dictionary mapping XCIST materials to appropriate lower thresholds in the ground truth image, see the .cfg here for examples <https://github.com/xcist/phantoms-voxelized/tree/main/DICOM_to_voxelized>
   """
   if not material_threshold_dict:
    material_threshold_dict = dict(zip(
       ['ICRU_lung_adult_healthy', 'ICRU_adipose_adult2', 'ICRU_liver_adult', 'water', 'ICRU_skeleton_cortical_bone_adult'],
       [-1000, -200, 0, 100, 300]))

    cfg_file_str = f"""
# Path where the DICOM images are located:
phantom.dicom_path = '{dicom_path}'
# Path where the phantom files are to be written (the last folder name will be the phantom files' base name):
phantom.phantom_path = '{save_path}'
phantom.materials = {list(material_threshold_dict.keys())}
phantom.mu_energy = 60                  # Energy (keV) at which mu is to be calculated for all materials.
phantom.thresholds = {list(material_threshold_dict.values())}	# Lower threshold (HU) for each material.
phantom.slice_range = [[0,10]]			  # DICOM image numbers to include.
phantom.show_phantom = False                # Flag to turn on/off image display.
phantom.overwrite = True                   # Flag to overwrite existing files without warning.
    """

    dicom_to_voxel_cfg = dicom_path / 'dicom_to_voxelized.cfg'

    with open(dicom_to_voxel_cfg, 'w') as f:
        f.write(cfg_file_str)
    
    DICOM_to_voxelized_phantom.run_from_config(dicom_to_voxel_cfg)


def get_effective_diameter(ground_truth_mu):
    pixel_width_mm = 480/1024
    A = np.sum(ground_truth_mu>0)*pixel_width_mm**2
    return 2*np.sqrt(A/np.pi)


def get_patient_name(phantom_df, code):
    if phantom_df['Code #'].dtype != int: code = str(code)
    idx = phantom_df[phantom_df['Code #'] == code].index[0]
    patient_num = phantom_df['Code #'][idx]
    gender = phantom_df['gender'][idx]
    gender_str = 'female' if gender == 'F' else 'male'
    if str(patient_num).split(' ')[0] == 'Reference':
        age = int(phantom_df['age (year)'][idx])
        age_str = 'infant' if age < 1 else f'{age}yr'
        patient = f'{gender_str}_{age_str}_ref'
    else:
        patient = f'{gender_str}_pt{patient_num}'
    return patient


df = pd.read_csv('selected_xcat_patients.csv')
patient_name_dict = {get_patient_name(df, code): code for code in df['Code #']}


def get_patient_info(phantom_id):
    code = patient_name_dict[phantom_id]
    patient_df = df[df['Code #'] == code]
    patient_df.pop('effective diameter (cm)')
    patient_df.pop('liver location (relative to height)')
    return patient_df
 

def make_summary_df(phantom_id, **kwargs):
    patient_info_df = get_patient_info(phantom_id).reset_index()
    additional_info_df = pd.DataFrame(kwargs)
    return pd.concat([patient_info_df, additional_info_df], axis=1)


def load_dicom_image(dicom_filename): return pydicom.read_file(dicom_filename).pixel_array


def load_organ_mask(phantom_path, slice_id, organ='liver'):
    mask_filename = list((phantom_path / f'voxelized_{slice_id}').glob(f'*{organ}*x1.raw'))[0]
    mask = load_volume(mask_filename, shape=(1024,1024))
    if organ == 'liver':
        mask = mask == 1.049
    return mask


def cleanup_simulation(resultsName): [os.remove(f) for f in glob(f'{resultsName}.*')]


def run_simulation(datadir, output_dir, phantom_id, slice_id=0, mA=200, kVp=120, FOV=None, add_lesion=False):
    # prepare output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    # prepare phantom for simulation
    ground_truth = np.copy(load_volume(datadir/f'{phantom_id}_atn_1.bin')[slice_id]) #whole phantom loaded as attenuation coefficients mu [1/pixels]
    seg_file = datadir/f'{phantom_id}_act_1.bin'
    if seg_file.exists():
        ground_truth_labels = np.copy(load_volume(seg_file)[slice_id])
    atten_coeffs = get_attenuation_coefficients(datadir / f'{phantom_id}_log')
    mu_water = atten_coeffs['Body (water)']

    ground_truth_image = mu_to_HU(ground_truth, mu_water)

    phantom_path = output_dir / 'phantoms' / f'{phantom_id}' / f'{slice_id:03d}'
    phantom_path.mkdir(exist_ok=True, parents=True)
 
    dicom_filename = phantom_path / f'ground_truth_{slice_id:03d}.dcm'
    convert_to_dicom(ground_truth_image, dicom_filename)

    processed_phantom_path = phantom_path / f'voxelized_{slice_id:03d}'
    voxelize_ground_truth(phantom_path, processed_phantom_path)

    lesion_coords = (0, 0)
    lesion_filename = ''
    if add_lesion:
        radius = 20
        contrast = -100
        img = load_dicom_image(dicom_filename)
        organ_mask = load_organ_mask(phantom_path, slice_id, organ='liver')
        print(f'Pixels in organ mask: {organ_mask.sum()}')
        if organ_mask.sum() > 1:
            img_w_lesion, lesion_image, lesion_coords = add_random_circle_lesion(img, organ_mask, radius=radius, contrast=contrast)
            lesion_path = phantom_path / 'lesions'
            lesion_path.mkdir(exist_ok=True)
            lesion_filename = lesion_path / f'lesion_{slice_id:03d}.dcm'
            img_w_lesion -= 1024 # <-- check if convert_to_dicom has bias added or not
            convert_to_dicom(lesion_image, lesion_filename)
            convert_to_dicom(img_w_lesion, dicom_filename)

    voxelize_ground_truth(phantom_path, processed_phantom_path)

    # load defaults
    ct = xc.CatSim('defaults/Phantom_Default',
                   'defaults/Physics_Default',
                   'defaults/Protocol_Default',
                   'defaults/Recon_Default',
                   'defaults/Scanner_Default')
    
    # change relevant parameters for the experiment
    ct.cfg.phantom.filename = str(processed_phantom_path / f'voxelized_{slice_id:03d}.json')

    results_dir = output_dir / 'simulations' / f'{phantom_id}'
    results_dir.mkdir(exist_ok=True, parents=True)

    if seg_file.exists():
        seg_dicom_file = results_dir / f'{phantom_id}_{slice_id}_segmentation_labels.dcm'
        convert_to_dicom(ground_truth_labels, seg_dicom_file)
        print(f'organ segmentations saved to: {seg_dicom_file}')

    ct.resultsName = str(results_dir / f'{phantom_id}_{slice_id}_{mA}mA_{kVp}kV')
    ct.cfg.experimentDirectory = str(results_dir)
    ct.cfg.protocol.mA = mA
    kVp_options = [80, 90, 100, 110, 120, 130, 140]
    if kVp not in kVp_options:
       raise ValueError(f'Selected kVP [{kVp}] not available, please choose from {kVp_options}')
    ct.cfg.protocol.spectrumFilename = f'tungsten_tar7_{kVp}_unfilt.dat'
    ct.cfg.waitForKeypress=False

    pixel_width_mm = 480 / ground_truth_image.shape[0]
    mu_water_mm = mu_water / pixel_width_mm #in units of 1/mm as opposed to 1/pixel
    ct.cfg.recon.mu = mu_water_mm

    if not FOV:
       FOV = 1.4*get_effective_diameter(ground_truth)

    print(f'FOV size: {FOV}')

    ct.run_all()

    ct.cfg.waitForKeypress=False
    ct.cfg.do_Recon = True
    ct.add_lesion = add_lesion
    ct.lesion_filename = lesion_filename

    ct.cfg.recon.fov = FOV
    recon.recon(ct.cfg)
    return ct

# %%
if __name__ == "__main__":
    datadir = Path('/gpfs_projects/brandon.nelson/REALYSM_peds/test_torsos/anthropomorphic/phantoms/full_fov')

    phantom_ids = [o.stem.split('_log')[0] for o in datadir.glob('*_log')]
    phantom_ids

    kVp_options = [80, 90, 100, 110, 120, 130, 140]
    mA_options = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    slice_id = 100
    mA = 200
    kV = 120

    output_dir = 'output_images'
    for phantom in phantom_ids[1:]:
        print(phantom)
        run_simulation(datadir, output_dir, phantom, slice_id, mA=mA, kVp=kV, FOV=350)
# %%
