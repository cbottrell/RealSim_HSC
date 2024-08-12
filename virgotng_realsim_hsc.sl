#!/bin/bash -l 
#SBATCH --job-name=realsim
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --array=0-7
#SBATCH --partition=p.large
#SBATCH --mail-type=END
#SBATCH --error='/u/bconn/Scratch/%x-%A_%a.err' 
#SBATCH --output='/u/bconn/Scratch/%x-%A_%a.out' 
#SBATCH --mail-user=connor.bottrell@icrar.org

module purge
source $HOME/.bashrc
conda activate tf39_cpu

export JOB_ARRAY_NJOBS=8
export JOB_ARRAY_INDEX=$SLURM_ARRAY_TASK_ID

# export UNIVERSE='Simba'
# export SIMULATION='L100n1024FP'
# export SNAPMIN=145
# export SNAPMAX=145

export UNIVERSE='IllustrisTNG'
export SIMULATION='TNG50-1'
export SNAPMIN=68
export SNAPMAX=71

cd /u/bconn/Projects/Simulations/IllustrisTNG/Scripts/RealSim_HSC
python virgotng_realsim_hsc.py

