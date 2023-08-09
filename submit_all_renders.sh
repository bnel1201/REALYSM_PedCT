#!/bin/sh
############################################
#            FOR ALL MATERIALS             
############################################




COUNT=0
for ID_PHANTOM in {1..3} #5
do
for ID_KVP in {1..7} #7
do
for ID_MA in {1..10} #10
do
for ID_SLICE in {1..10} #501 
do
for ID_SIMULATION in {1..1} #2
do
COUNT=$[$COUNT +1]
echo setting SGE_TASK_ID = $COUNT
qsub submit_one_render.sge $COUNT
done
done
done
done
done
