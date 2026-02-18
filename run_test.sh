#!/bin/bash
#SBATCH --job-name=NS_Test
#SBATCH --time=02:30:00
#SBATCH --partition=dcgp_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=128GB
#SBATCH --account=EUHPC_TDEMO_26

source /leonardo/home/userexternal/vsironi0/Navier-Stokes_equations/load_env.sh

cd /leonardo/home/userexternal/vsironi0/Navier-Stokes_equations/build

echo "Lancio simulazione di TEST su 32 core..."
srun ./navier_stokes
echo "Finito!"
