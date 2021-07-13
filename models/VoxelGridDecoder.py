import torch.nn as nn
from .Model import Model

class VoxelGridDecoder(Model):

    def __init__(self, name, in_channels):
        super().__init__(name)

        self.reconstruction_layers = nn.Sequential(
            nn.ConvTranspose3d(in_channels, 512, 2, stride=1), # 1x1x1/4096 -> 2x2x2/512
            nn.Conv3d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(512, 64, 4, stride=4), # 2x2x2/512 -> 8x8x8/64
            nn.Conv3d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 8, 2, stride=2), # 8x8x8/64 -> 16x16x16/8
            nn.Conv3d(8, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(8, 8, 3, padding=1),
            nn.ReLU(),            
            nn.ConvTranspose3d(8, 4, 2, stride=2), # 16x16x16/8 -> 32x32x32/4
            nn.Conv3d(4, 4, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(4, 4, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(4, 2, 2, stride=2), # 32x32x32/4 -> 64x64x64/2
            nn.Conv3d(2, 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(2, 2, 3, padding=1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = x.view((1, -1, 1, 1, 1)) # (number of channels, size in x-, y-, z- direction)

        x = self.reconstruction_layers(x)
        return x
