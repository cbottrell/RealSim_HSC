#!/bin/bash -l 
#SBATCH --job-name=realsim
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --array=0-299%8
#SBATCH --partition=p.large
#SBATCH --mail-type=END
#SBATCH --error='/u/bconn/Scratch/%x-%A_%a.err' 
#SBATCH --output='/u/bconn/Scratch/%x-%A_%a.out' 
#SBATCH --mail-user=connor.bottrell@ipmu.jp

module purge
source $HOME/.bashrc
conda activate tf39_cpu

export JOB_ARRAY_NJOBS=300
export JOB_ARRAY_INDEX=$SLURM_ARRAY_TASK_ID
export SIM='TNG50-1'

cd /u/bconn/Projects/Simulations/IllustrisTNG/Scripts/RealSim_HSC
python RealSim_HSC.py

