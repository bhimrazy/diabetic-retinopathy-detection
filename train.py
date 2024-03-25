import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

from src.dataset import DRDataModule
from src.model import DRModel

# seed everything for reproducibility
SEED = 42
L.seed_everything(SEED, workers=True)
torch.set_float32_matmul_precision("high")


# Init DataModule
dm = DRDataModule(batch_size=96, num_workers=8)
dm.setup()

# Init model from datamodule's attributes
model = DRModel(
    num_classes=dm.num_classes, learning_rate=3e-5, class_weights=dm.class_weights
)

# Init logger
logger = TensorBoardLogger("lightning_logs", name="dr_model")

# Init callbacks
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    mode="min",
    save_top_k=3,
    dirpath="checkpoints",
)

# Init LearningRateMonitor
lr_monitor = LearningRateMonitor(logging_interval="step")

# Init trainer
trainer = L.Trainer(
    max_epochs=20,
    accelerator="auto",
    devices="auto",
    logger=logger,
    callbacks=[checkpoint_callback, lr_monitor],
    enable_checkpointing=True,
)

# Pass the datamodule as arg to trainer.fit to override model hooks :)
trainer.fit(model, dm)
