import argparse
from pathlib import Path
import os

import pandas as pd

XCAT_dir = '/gpfs_projects/brandon.nelson/XCAT/XCAT_V2_LINUX/'
XCAT_MODELFILES_DIR='/gpfs_projects/brandon.nelson/XCAT/modelfiles'
XCAT = 'dxcat2_linux_64bit'


XCAT_MODELFILES_DIR = Path(XCAT_MODELFILES_DIR)

def get_diameter(df, code, units='mm'):
    diameter = float(df[df['Code #'] == code]['effective diameter (cm)'])
    if units == 'mm':
        diameter *= 10
    return diameter

def get_nrb_filenames(phantom_df, code):
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
    patient_nrb_file = XCAT_MODELFILES_DIR / f'{patient}.nrb'
    patient_heart_nrb_file =  XCAT_MODELFILES_DIR / f'{patient}_heart.nrb'
    return patient_nrb_file, patient_heart_nrb_file, patient


def make_phantom(patient_code, output_dir, phantom_file=None, fov=None, array_size = 1024, energy=60):
    """
    energy [keV]
    """
    GENERAL_PARAMS_FILE = os.path.abspath(Path(__file__).parent / 'general.samp.par')
    output_dir = Path(os.path.abspath(output_dir))
    

    phantom_df = pd.read_csv('selected_xcat_patients.csv')
    patient_nrb_file, patient_heart_nrb_file, patient = get_nrb_filenames(phantom_df, patient_code)
    (output_dir/patient).mkdir(exist_ok=True, parents=True)
    if phantom_file:
        patient_nrb_file = os.path.abspath(phantom_file)

    patient_info = phantom_df[phantom_df['Code #'] == patient_code]
    gender = 0 if patient_info['gender'].iloc[0] == 'M' else 1

    height = patient_info['height (cm)'].iloc[0]

    estimated_eff_diameter = get_diameter(phantom_df, patient_code, units='cm')
    fov = fov or min(1.1*estimated_eff_diameter, 48) #in cm
    pixel_width_cm = fov / array_size


    # liver_location = phantom_df[phantom_df['Code #']==patient_code]['liver location (relative to height)'].to_numpy()[0]
    # midslice = round(height / pixel_width_cm * liver_location)

    cmd = f'cd {XCAT_dir}\n./{XCAT} {GENERAL_PARAMS_FILE}\
             --organ_file {patient_nrb_file}\
             --pixel_width {pixel_width_cm}\
             --slice_width {pixel_width_cm}\
             --array_size {array_size}\
             --energy {energy}\
             --startslice {int(height/pixel_width_cm)-250}\
             --endslice {int(height/pixel_width_cm)}\
             --arms_flag 1\
             --gender {gender}\
             {output_dir}/{patient}'
    print(cmd)
    os.system(cmd)
# %%

def main(patient_code, output_dir, phantom_file=None, fov=25):
    make_phantom(patient_code, output_dir, phantom_file=phantom_file, fov=fov, array_size=1024)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make Anthropomorphic Phantoms using XCAT')
    parser.add_argument('patient_code',
                        help="patient code from XCAT csv")
    parser.add_argument('output_dir',
                        help="output directory to save XCAT phantom bin files")
    parser.add_argument('-f', '--phantom_file', required=False,
                        help="nrb file containing virtual patient in XCAT format")
    args = parser.parse_args()
    main(args.patient_code, args.output_dir, args.phantom_file)
