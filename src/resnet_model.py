"""
ResNet18 adapted for small (64x64) die-face crops.

Standard torchvision ResNet18 assumes ~224x224 ImageNet input and
aggressively downsamples in the stem (7x7 stride-2 conv + maxpool), which
throws away most of the signal on a small crop. We swap the stem for a
3x3 stride-1 conv and drop the maxpool — the same trick commonly used for
CIFAR-scale ResNets — so the network keeps enough spatial resolution to
learn from crops this size.
"""
import torch.nn as nn
from torchvision.models import resnet18


def build_model(num_classes: int = 6, pretrained: bool = False) -> nn.Module:
    model = resnet18(weights="IMAGENET1K_V1" if pretrained else None)

    # Replace the stem for small inputs. Skips loading pretrained conv1
    # weights (shape wouldn't match anyway) — pretrained=True still helps
    # via the deeper layers, but for this task random init tends to be
    # just as good given how different die-face crops are from ImageNet.
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model