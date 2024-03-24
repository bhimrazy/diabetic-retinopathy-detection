import lightning as L
import torch
from torch import nn
from torchmetrics.functional import accuracy
from torchvision import models


class DRModel(L.LightningModule):
    def __init__(
        self, num_classes: int, learning_rate: float = 2e-4, class_weights=None
    ):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.learning_rate = learning_rate

        # Define the model
        # self.model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        self.model = models.densenet169(weights=models.DenseNet169_Weights.DEFAULT)
        # self.model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        # freeze the feature extractor
        for param in self.model.parameters():
            param.requires_grad = False
        # Change the output layer to have the number of classes
        in_features = self.model.classifier.in_features  
        # in_features = 768  
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_features // 2, num_classes),
        )

        # Define the loss function
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, task="multiclass", num_classes=self.num_classes)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val_loss",
            },
        }
        # return optimizer
