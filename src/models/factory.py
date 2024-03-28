from torchvision import models
from torch import nn

model_mapping = {
    "densenet121": (
        models.densenet121,
        {"weights": models.DenseNet121_Weights.DEFAULT, "family": "densenet"},
    ),
    "densenet161": (
        models.densenet161,
        {"weights": models.DenseNet161_Weights.DEFAULT, "family": "densenet"},
    ),
    "densenet169": (
        models.densenet169,
        {"weights": models.DenseNet169_Weights.DEFAULT, "family": "densenet"},
    ),
    "densenet201": (
        models.densenet201,
        {"weights": models.DenseNet201_Weights.DEFAULT, "family": "densenet"},
    ),
    "resnet50": (
        models.resnet50,
        {"weights": models.ResNet50_Weights.IMAGENET1K_V2, "family": "resnet"},
    ),
    "resnet101": (
        models.resnet101,
        {"weights": models.ResNet101_Weights.IMAGENET1K_V2, "family": "resnet"},
    ),
    "resnet152": (
        models.resnet152,
        {"weights": models.ResNet152_Weights.IMAGENET1K_V2, "family": "resnet"},
    ),
    "vit-b-16": (
        models.vit_b_16,
        {"weights": models.ViT_B_16_Weights.DEFAULT, "family": "vit"},
    ),
    "vit-b-32": (
        models.vit_b_32,
        {"weights": models.ViT_B_32_Weights.DEFAULT, "family": "vit"},
    ),
    # Add more models as needed with their respective configurations.
}


class Model(nn.Module):
    """Moodel definition."""

    def __init__(self, model_name: str, num_classes: int):
        """
        Initialize Model instance.

        Args:
            model_name (str): Name of the model architecture.
            num_classes (int): Number of output classes.
        """
        super(Model, self).__init__()

        model_class, model_config = model_mapping[model_name]
        self.model = model_class(weights=model_config["weights"])

        # Freeze model parameters
        for param in self.model.parameters():
            param.requires_grad = False

        in_features = self._get_in_features(model_config["family"])

        if model_config["family"] == "densenet":
            self.model.classifier = self._create_classifier(in_features, num_classes)
        elif model_config["family"] == "resnet":
            self.model.fc = self._create_classifier(in_features, num_classes)
        elif model_config["family"] == "vit":
            self.model.heads = self._create_classifier(in_features, num_classes)

    def forward(self, x):
        """Forward pass through the model."""
        return self.model(x)

    def _get_in_features(self, family: str) -> int:
        """Return the number of input features for the classifier."""
        if family == "densenet":
            return self.model.classifier.in_features
        elif family == "resnet":
            return self.model.fc.in_features
        elif family == "vit":
            return self.model.heads.head.in_features

    def _create_classifier(self, in_features: int, num_classes: int) -> nn.Sequential:
        """Create the classifier module."""
        return nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features // 2, num_classes),
        )


class ModelFactory:
    """
    Factory for creating different models based on their names.

    Args:
        name (str): The name of the model factory.
        num_classes (int): The number of output classes.

    Raises:
        ValueError: If the specified model factory is not implemented.
    """

    def __init__(self, name: str, num_classes: int):
        """
        Initialize ModelFactory instance.

        Args:
            name (str): The name of the model.
            num_classes (int): The number of output classes.
        """
        self.name = name
        self.num_classes = num_classes

    def __call__(self):
        """
        Create a model instance based on the provided name.

        Args:
            model_name (str): Name of the model architecture.
            num_classes (int): Number of output classes.

        Returns:
            Model: An instance of the selected model.
        """
        if self.name not in model_mapping:
            valid_options = ", ".join(model_mapping.keys())
            raise ValueError(
                f"Invalid model name: '{self.name}'. Available options: {valid_options}"
            )

        return Model(self.name, self.num_classes)


if __name__ == "__main__":
    model = ModelFactory("resnet50", 5)()