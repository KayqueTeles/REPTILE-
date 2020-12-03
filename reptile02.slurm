#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --job-name=REPTILES
#SBATCH --cpus-per-task=40
#SBATCH --partition=cosmoobs

cd /home/kayque/LENSLOAD/SCRIPTS/REPTILE/CODE-PACK/

export OMP_NUM_THREADS=$SLURM_NTASKS

echo -e "\n\n\n ## Job iniciado em $(date +'%d-%m-%Y as %T') #####################\n"

# Informacoes do job impressos no arquivo de saida.
echo -e "\n ## Jobs ativos: \n"
squeue --jobs
echo -e "\n ## Node de execucao do job:         $(hostname -s) \n"
echo -e "\n ## Numero de tarefas para este job: $SLURM_NTASKS \n"

module load softwares/anaconda3/5.0-intel-2018.0
module load compilers/cuda/10.0
module load cuda/9.0.176-gcc-5.3.0
source activate LEGIOND

python reptile03.py

echo -e "\n\n\n ## Job finalizado em $(date +'%d-%m-%Y as %T') ###################\n"
