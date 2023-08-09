#!/bin/sh
############################################
#            FOR ALL MATERIALS             
############################################




# COUNT=0
# for ID_PHANTOM in {1..3} #5
# do
# for ID_KVP in {1..7} #7
# do
# for ID_MA in {1..10} #10
# do
# for ID_SLICE in {1..10} #501 
# do
# for ID_SIMULATION in {1..1} #2
# do
# COUNT=$[$COUNT +1]
# echo setting SGE_TASK_ID = $COUNT
# qsub submit_one_render.sge $COUNT
# done
# done
# done
# done
# done

PHANTOM_DIR=/projects01/didsr-aiml/brandon.nelson/XCAT_body/full_fov/

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
qsub -N XCIST_REALSYM -t $START_TASK-$END_TASK submit_one_render.sge
