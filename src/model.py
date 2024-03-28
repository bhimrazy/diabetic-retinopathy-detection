import lightning as L
import torch
from torch import nn
from torchmetrics.functional import accuracy, cohen_kappa
from src.models.factory import ModelFactory


class DRModel(L.LightningModule):
    def __init__(
        self,
        num_classes: int,
        model_name: str = "densenet121",
        learning_rate: float = 3e-4,
        class_weights=None,
        use_scheduler: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.use_scheduler = use_scheduler

        # Define the model
        self.model = ModelFactory(name=model_name, num_classes=num_classes)()

        # Define the loss function
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, task="multiclass", num_classes=self.num_classes)
        kappa = cohen_kappa(
            preds,
            y,
            task="multiclass",
            num_classes=self.num_classes,
            weights="quadratic",
        )
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_kappa", kappa, on_step=True, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=0.05
        )

        configuration = {
            "optimizer": optimizer,
            "monitor": "val_loss",  # monitor validation loss
        }

        if self.use_scheduler:
            # Add lr scheduler
            # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",  # or "max" if you're maximizing a metric
                factor=0.1,  # factor by which the learning rate will be reduced
                patience=5,  # number of epochs with no improvement after which learning rate will be reduced
                verbose=True,  # print a message when learning rate is reduced
                threshold=0.001,  # threshold for measuring the new optimum, to only focus on significant changes
            )

            configuration["lr_scheduler"] = scheduler

        return configuration
