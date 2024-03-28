from dataclasses import dataclass

@dataclass
class Config:
    # Data paths
    train_csv_path: str = "data/train.csv"
    val_csv_path: str = "data/val.csv"
    test_csv_path: str = "data/test.csv"  # Added for test data
    
    # Model parameters
    input_size: tuple[int, int] = (224, 224)  # Input image size
    num_classes: int = 5  # Number of output classes
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 0.001
    model_architecture: str = "PretrainedResNet50"  # Specify the backbone architecture
    loss_function: str = "cross_entropy"
    optimizer: str = "Adam"
    lr_scheduler: str = "StepLR"
    dropout_rate: float = 0.5
    weight_decay: float = 0.001
    early_stopping: bool = True
    use_gpu: bool = True
    random_seed: int = 42
    data_augmentation: bool = True
    
    # Model paths
    model_save_path: str = "models/model.pth"
    logs_path: str = "logs/"
