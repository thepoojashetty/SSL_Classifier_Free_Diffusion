from torchvision import transforms
import pytorch_lightning as pl
import config
from pytorch_lightning.callbacks import LearningRateMonitor,ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from datamodule import CifarDataModule
from model import UNet
from helpers import *


if __name__=="__main__":
    
    transform=transforms.Compose([
        transforms.ToTensor(),
        #scale values between -1 and 1
        transforms.Lambda(scale)
    ])
    """
    diff_model=UNet(
        dim=config.IMAGE_SIZE,
        channels=config.CHANNELS,
        p_uncond=0.2,
        learning_rate=config.LEARNING_RATE,
        timesteps=config.TIMESTEPS
    )
    """

    #use pretrained model
    diff_model = UNet.load_from_checkpoint(config.CKPT_DIR_PATH+"/model_val_loss(val_loss=0.09)_epoch(epoch=81).ckpt")
    
    dm = CifarDataModule(
                        data_dir=config.DATA_DIR,
                        batch_size=config.BATCH_SIZE,
                        num_workers=config.NUM_WORKERS,
                        transform=transform,
                        p_uncond=0.2
                )
    
    logger = TensorBoardLogger(save_dir=config.LOG_PATH,name="ssl_cfg_logs")
    trainer = pl.Trainer(
        logger=logger,
        min_epochs=1,
        max_epochs=config.NUM_EPOCHS,
        callbacks=[
            LearningRateMonitor(logging_interval='epoch'),
            ModelCheckpoint(
                dirpath=config.CKPT_DIR_PATH,
                filename="model_minval_loss({val_loss:.2f})_epoch({epoch})",
                monitor="val_loss",
                mode="min"
            ),
            ModelCheckpoint(
                dirpath=config.CKPT_DIR_PATH,
                filename="modelA_val_loss({val_loss:.2f})_epoch({epoch})",
            )
        ]
    )

    trainer.fit(diff_model,dm)
    #trainer.validate(model,dm)
    #trainer.test(model,dm)
    """
     ModelCheckpoint(
                dirpath=config.CKPT_DIR_PATH,
                filename="model_val_loss({val_loss:.2f})_epoch({epoch})",
                monitor="val_loss",
                mode="min"
        ),
    """

    