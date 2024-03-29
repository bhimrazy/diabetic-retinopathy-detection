from os.path import join

import hydra
import lightning as L
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig
from src.data_module import DRDataModule
from src.model import DRModel
from src.utils import generate_run_id


@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg: DictConfig) -> None:
    # generate unique run id based on current date & time
    run_id = generate_run_id()

    # Seed everything for reproducibility
    L.seed_everything(cfg.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    # Initialize DataModule
    dm = DRDataModule(
        train_csv_path=cfg.train_csv_path,
        val_csv_path=cfg.val_csv_path,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        use_class_weighting=cfg.use_class_weighting,
        use_weighted_sampler=cfg.use_weighted_sampler,
    )
    dm.setup()

    # Init model from datamodule's attributes
    model = DRModel(
        num_classes=dm.num_classes,
        model_name=cfg.model_name,
        learning_rate=cfg.learning_rate,
        class_weights=dm.class_weights,
        use_scheduler=cfg.use_scheduler,
    )

    # Init logger
    logger = TensorBoardLogger(save_dir=cfg.logs_dir, name="", version=run_id)
    # Init callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=2,
        dirpath=join(cfg.checkpoint_dirpath, run_id),
        filename="{epoch}-{step}-{val_loss:.2f}-{val_acc:.2f}-{val_kappa:.2f}",
    )

    # Init LearningRateMonitor
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # early stopping
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        verbose=True,
        mode="min",
    )

    # Initialize Trainer
    trainer = L.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator="auto",
        devices="auto",
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor, early_stopping],
    )

    # Train the model
    trainer.fit(model, dm)


if __name__ == "__main__":
    train()
