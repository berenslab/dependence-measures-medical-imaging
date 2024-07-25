#! /bin/bash
#SBATCH --output=logfiles/%j.out  # File to which STDOUT will be written - make sure this is not on $HOME
#SBATCH --error=logfiles/%j.err   # File to which STDERR will be written - make sure this is not on $HOME

# $1 amount of jobs
# $2 train config file
# $3 sweep id

# Example call: 
# scripts/sweep.sh 20 configs/morpho-mnist/mine.yaml enr822ae

PROJECT_DIR=/absolute/path/to/code/
train_script=$PROJECT_DIR/scripts/train.sh

for (( i=0; i<$1; ++i)); do
  sbatch $train_script $2 $3; sleep 5
done