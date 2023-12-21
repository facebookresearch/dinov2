#!/bin/bash
### chdir in the correct folder and activate the correct python environment. Then run on login node with: qsub ./train.sh <args>
### some explanation:
### -V: currently set env variables (e.g. conda) are transferred
### -cwd: job is started in current directory
### -l h=maxg03,h=maxg04,h=maxg05,h=maxg06,h=maxg07,h=maxg08,gpu=1: nodes with the V100 GPUs installed (not T4) - will change soon!


### $ -l gpu=1 -l cuda_name="Tesla-V100-SXM2-32GB"
### $ -l gpu=1 -l cuda_name="Tesla-V100-PCIE-32GB"
### $ -l gpu=1 -l cuda_name="Tesla-V100-SXM2-16GB"
### $ -l h=maxg03,h=maxg04,h=maxg05,h=maxg06,h=maxg07,h=maxg08,gpu=1
#$ -l gpu=8
#$ -l cuda_memory=45G
#$ -pe smp 4
#$ -cwd
#$ -V
#$ -e error_log_$JOB_ID
#$ -o out_log_$JOB_ID
#$ -l h_rt=96:00:00
#$ -A kainmueller

torchrun "$@"
retVal=$?
if [ $retVal -ne 0 ]; then
    echo "Error"
    exit 100
fi