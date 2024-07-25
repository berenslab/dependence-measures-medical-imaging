#!/bin/bash
#SBATCH --cpus-per-task=16          # Number of CPU cores per task
#SBATCH --partition=partition-name  # Request a specific partition for the resource allocation
#SBATCH --nodes=1                   # Ensure that all cores are on one machine
#SBATCH --time=0-18:00              # Runtime in D-HH:MM
#SBATCH --gres=gpu:1                # optionally type and number of gpus
#SBATCH --mem=100G                  # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=logfiles/%j.out    # File to which STDOUT will be written - make sure this is not on $HOME
#SBATCH --error=logfiles/%j.err     # File to which STDERR will be written - make sure this is not on $HOME
#SBATCH --mail-type=END             # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=anonymous@domain.com  # Email to which notifications will be sent

# print info about current job
scontrol show job $SLURM_JOB_ID 

PROJECT_DIR=/absolute/path/to/code/

echo "start training"
python_path=/path/to/python/

if [ "${2}" == "" ]; then
    $python_path $PROJECT_DIR/src/train_k_fold_cv.py -tc $1
else
    $python_path $PROJECT_DIR/src/train_k_fold_cv.py -tc $1 -sid $2
fi