#!/bin/bash

#SBATCH -J test_GPU
#SBATCH -p dl
#SBATCH -o /****/GPU_slurm_test/test_GPU_%j.txt
#SBATCH -e /****/GPU_slurm_test/test_GPU_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=****@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node p100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=02:00:00

#Modules loaded
module load python
module load deeplearning

#Run your program
srun python /****/GPU_slurm_test/test_GPU.py