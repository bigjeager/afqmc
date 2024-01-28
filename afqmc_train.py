from model import AFQData, AFQModel
import lightning as L
import numpy as np
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

if __name__ == '__main__':
    config = {
        "batch_size": 32,
        "lr": 1e-4,
        "weight_decay": 0.01,
        "epoch": 10
    }

    data = AFQData(config)
    model = AFQModel(config)

    logger = TensorBoardLogger("logs", name="afq", default_hp_metric=False)
    trainer = L.Trainer(
        strategy=DDPStrategy(find_unused_parameters=False),
        devices="auto",
        accelerator="auto",
        #enable_progress_bar=False,
        max_epochs=config['epoch'],
        logger=logger,
        log_every_n_steps=1,
        enable_checkpointing=False)
    trainer.fit(model, datamodule = data)
