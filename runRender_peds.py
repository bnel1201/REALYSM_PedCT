import os
import time
import random
from pathlib import Path
from argparse import ArgumentParser

import pandas as pd

import dxcist
from dxcist.xcist_sims import run_simulation, make_summary_df, cleanup_simulation, load_volume

phantom_dir = Path('/projects01/didsr-aiml/brandon.nelson/XCAT_body/full_fov')
save_dir = Path('/projects01/didsr-aiml/brandon.nelson/XCAT_body/Peds_w_liver_lesions/simulations')


def renderImage(**kwargs): return run_simulation(output_dir=save_dir, **kwargs)   


def run_task(SGE_TASK_ID):
    print('SGE_TASK_ID ' + str(SGE_TASK_ID))
    [phantom_id, kVp_id, mA_id, slice_id, lesion_id, simulation_id] = l_parameter_comb[SGE_TASK_ID]

    print('phantom_id ' + str(phantom_id))
    print('kVp_id ' + str(kVp_id))
    print('mA_id ' + str(mA_id))
    print('slice_id ' + str(slice_id))
    print('lesion_id ' + str(lesion_id))
    print('simulation_id ' + str(simulation_id))

    start_time = time.time()

    phantom = load_volume(phantom_dir/f'{phantom_id}_atn_1.bin') #whole phantom loaded as attenuation coefficients mu [1/pixels]
    atten_coeffs = dxcist.xcist_sims.get_attenuation_coefficients(phantom_dir / f'{phantom_id}_log')
    mu_water = atten_coeffs['Body (water)']
    ground_truth_image = dxcist.xcist_sims.mu_to_HU(phantom[slice_id], mu_water)

    lesion_coords = (0, 0)
    lesion_filename = ''
    if lesion_id:
        radius = 20
        contrast = -100
        organ_mask = dxcist.xcist_sims.load_organ_mask(phantom_dir/f'{phantom_id}_act_1.bin', slice_id=slice_id, organ='liver')
        print(f'Pixels in organ mask: {organ_mask.sum()}')
        if organ_mask.sum() > 1:
            img_w_lesion, lesion_image, lesion_coords = dxcist.lesions.add_random_circle_lesion(ground_truth_image, organ_mask, radius=radius, contrast=contrast)
            ground_truth_image = img_w_lesion
            ct_lesion = renderImage(ground_truth_image=lesion_image, phantom_id=f'{phantom_id}_lesion_only', slice_id=slice_id, mA=mA_id, kVp=kVp_id)

    pixel_width_mm = 480 / ground_truth_image.shape[0]
    mu_water_mm = mu_water / pixel_width_mm #in units of 1/mm as opposed to 1/pixel
    ct = renderImage(ground_truth_image=ground_truth_image, phantom_id=phantom_id, slice_id=slice_id, mA=mA_id, kVp=kVp_id, mu_water=mu_water_mm) #NOTE AEC is not on! smaller patients will enherently be less noisy, this is not clinically representative
    total_time =  time.time() - start_time
    print(total_time)
    ct.add_lesion = lesion_id
    return ct


if __name__ == "__main__":

    parser = ArgumentParser(description='Runs XCIST CT simulations on XCAT datasets')
    parser.add_argument('--phantom_dir', type=str, default="", help='directory containing `.bin` voxelized XCAT phantoms')
    parser.add_argument('--save_dir', type=str, default="", help='directory to save simulation results')
    args = parser.parse_args()

    phantom_dir = args.phantom_dir or '/gpfs_projects/brandon.nelson/REALYSM_peds/torsos/'
    save_dir = args.save_dir or '/gpfs_projects/brandon.nelson/REALYSM_peds/torsos/output_images_2023-08-10'

    phantom_dir = Path(phantom_dir)
    save_dir = Path(save_dir)

    # find parameter
    phantom_list = [o.stem.split('_log')[0] for o in phantom_dir.glob('*_log')]
    kVp_list = list(range(80, 150, 10))
    mA_list = list(range(50, 550, 50))
    slice_list = list(range(501))
    simulations_list = list(range(1)) #This can be increased to enable multiple scans (different noise realizations of the same slice and settings)

    l_parameter_comb = []
    for phantom_id in phantom_list:
        for kVp_id in kVp_list:
            for mA_id in mA_list:
                for slice_id in slice_list:
                    for lesion_id in [True]:
                        for simulation_id in simulations_list:
                            l_parameter_comb.append([phantom_id, kVp_id, mA_id, slice_id, lesion_id, simulation_id])
    random.shuffle(l_parameter_comb)
    
    run_in_parallel = True
    try:
        SGE_TASK_ID = int(os.environ['SGE_TASK_ID'])-1 # since tasks start from 1
    except:
        run_in_parallel = False    

    if run_in_parallel:
        ct = run_task(SGE_TASK_ID)
        params = l_parameter_comb[SGE_TASK_ID]
        sim_summary_df = make_summary_df(phantom_id = params[0],
                                         kVp = [params[1]],
                                         mA = [params[2]],
                                         slice = [params[3]],
                                         fov = [ct.cfg.recon.fov],
                                         add_lesion = [params[4]],
                                         simulation = [params[5]],
                                         lesion_name = [ct.lesion_filename],
                                         results_name = [f'{ct.resultsName}_512x512x1.raw'])
        sim_summary_df.to_csv(save_dir/f'{SGE_TASK_ID}_summary.csv')
        cleanup_simulation(ct.resultsName)
    else:
        print('SGE_TASK_ID not set, running in serial')
        n_params = len(l_parameter_comb)
        for SGE_TASK_ID in range(n_params):
            print(f'{SGE_TASK_ID}/{n_params}')
            params = l_parameter_comb[SGE_TASK_ID]
            ct = run_task(SGE_TASK_ID)
            if SGE_TASK_ID == 0:
                sim_summary_df = make_summary_df(phantom_id = params[0],
                                                 kVp = [params[1]],
                                                 mA = [params[2]],
                                                 slice = [params[3]],
                                                 fov = [ct.cfg.recon.fov],
                                                 add_lesion = [params[4]],
                                                 simulation = [params[5]],
                                                 lesion_name = [ct.lesion_filename],
                                                 results_name = [f'{ct.resultsName}_512x512x1.raw'])
            else:
                temp = make_summary_df(phantom_id = params[0],
                                                 kVp = [params[1]],
                                                 mA = [params[2]],
                                                 slice = [params[3]],
                                                 fov = [ct.cfg.recon.fov],
                                                 add_lesion = [params[4]],
                                                 simulation = [params[5]],
                                                 lesion_name = [ct.lesion_filename],
                                                 results_name = [f'{ct.resultsName}_512x512x1.raw'])
                sim_summary_df = pd.concat([sim_summary_df, temp])
            cleanup_simulation(ct.resultsName)

            sim_summary_df.to_csv(save_dir/'summary.csv')
