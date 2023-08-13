#!/bin/sh

SIM_NAME=Peds_w_liver_lesions_$(date +'%m-%d-%Y_%H-%M')
PHANTOM_DIR=/projects01/didsr-aiml/brandon.nelson/XCAT_body/full_fov/
SAVE_DIR=/projects01/didsr-aiml/brandon.nelson/XCAT_body/$SIM_NAME
LOG_DIR=/home/brandon.nelson/Dev/REALYSM_peds/logs/$SIM_NAME


N_PHANTOMS=$(ls $PHANTOM_DIR*.bin | wc -l)
echo $N_PHANTOMS phantoms found

N_mAs=10
N_kVp=2
N_slice=10
N_sims=1
N_lesion_conditions=2

COUNT=$[$N_PHANTOMS * $N_mAs * $N_kVp * N_slice * N_sims * N_lesion_conditions]
echo Running $COUNT simulation conditions

START_TASK=1
END_TASK=$COUNT
qsub -N $SIM_NAME -t $START_TASK-$END_TASK submit_one_render.sge $LOG_DIR $PHANTOM_DIR $SAVE_DIR
