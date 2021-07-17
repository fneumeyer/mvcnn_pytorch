import torch.nn as nn
from .Model import Model

# The "Baseline segmentation CNN" from the PointNet paper
class BaselineSegmentationCNN(Model):

    def __init__(self, name, num_classes):
        super().__init__(name)

        self.segmentation_layers = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(32, num_classes, kernel_size=1)
        )

    def forward(self, x):
        x = self.segmentation_layers(x)
        return x
