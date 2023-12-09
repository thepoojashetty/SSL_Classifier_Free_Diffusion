import os

#Hyperparams
BATCH_SIZE=64
NUM_EPOCHS=50
LEARNING_RATE=0.001
NUM_WORKERS=4
TIMESTEPS=100

#Data
DATA_DIR="/scratch/"+os.environ["SLURM_JOB_ID"]+"/data/"
IMAGE_SIZE=32
CHANNELS=3

#Model
CKPT_DIR_PATH="/cluster/ix87iquc/data/SSE_Classifier_Free_Diffusion/Model/Checkpoints"

#logging
LOG_PATH="/cluster/ix87iquc/data/SSE_Classifier_Free_Diffusion/tb_logs"