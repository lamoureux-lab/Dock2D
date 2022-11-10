#!/bin/bash
#SBATCH --partition=p_ccib_1
##SBATCH --partition=gpu
##SBATCH --job-name=MCFI_2samp_1000ep
##SBATCH --job-name=expC_MCFI_2samp
##SBATCH --job-name=expB_MCFI_2samp
##SBATCH --job-name=expA_MCFI_2samp


##SBATCH --nodes=1
##SBATCH --ntasks=1
##SBATCH --tasks-per-node=1
##SBATCH --cpus-per-task=12
##SBATCH --gres=gpu:2 # Number of GPUs
##SBATCH --constraint=volta
##SBATCH --constraint=oarc
##SBATCH --time=14-00:00:00
##SBATCH --output=slurm_log/slurm.%N.%j.%x.out
##SBATCH --error=slurm_log/slurm.%N.%j.%x.err
##SBATCH --export=ALL
##SBATCH --nodelist=gpuc002

#SBATCH --partition=gpu
#SBATCH --job-name=ep113res99step_MCFI_2samp
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:2 # Number of GPUs
### tesla
#SBATCH --cpus-per-task=12
#SBATCH --constraint=oarc
#SBATCH --mem=6400
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm_log/slurm.%N.%j.%x.out
#SBATCH --error=slurm_log/slurm.%N.%j.%x.err
#SBATCH --export=ALL

pwd
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun -N1 -n1 python train_montecarlo_FI.py;
