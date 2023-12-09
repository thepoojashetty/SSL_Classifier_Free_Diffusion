#!/bin/bash
#SBATCH --job-name=SSL_CFG
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12000
#SBATCH --gres=gpu:1
#SBATCH -o /cluster/ix87iquc/SSL_CFG.out
#SBATCH -e /cluster/ix87iquc/SSL_CFG.err
#SBATCH --mail-user=pooja.shetty@fau.de
#SBATCH --mail-type=ALL
#Timelimit format: "hours:minutes:seconds" -- max is 24h
#SBATCH --time=24:00:00
### Choose a specific GPU: #SBATCH --gres=gpu:q5000:1
### Run `sinfo -h -o "%n %G"` for GPU types

cp -r /cluster/ix87iquc/data/SSE_Classifier_Free_Diffusion /scratch/$SLURM_JOB_ID/SSE_Classifier_Free_Diffusion

#export PATH="/cluster/ix87iquc/miniconda/bin:$PATH"
source /cluster/ix87iquc/miniconda/etc/profile.d/conda.sh
conda activate nirenv

cd /scratch/$SLURM_JOB_ID/SSE_Classifier_Free_Diffusion
python main.py
