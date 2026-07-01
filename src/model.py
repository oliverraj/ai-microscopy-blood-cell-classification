import torch.nn as nn
from torchvision import models


NUM_CLASSES = 8


def build_model():
    """
    Build a pretrained ResNet18 model for blood cell classification.
    """

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Freeze pretrained layers
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Replace the classification layer
    num_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, NUM_CLASSES),
    )

    return model


if __name__ == "__main__":

    model = build_model()

    print(model)