import torch
import torch.nn as nn
import torchvision.models as models
from .Model import Model

class SpaPGNet(Model):

    def __init__(self, name, latent_space = 2048):
        super().__init__(name)

        self.encoder = models.resnet50(pretrained=self.pretraining)
        self.encoder.fc = nn.Linear(2048, latent_space)


        #Inverse-decoder scheme
        #64, kernel = 3, stride = 2, padding = 1 =>  33 //32 x 33 x 33 x 33 
        #33, kernel = 3, stride = 2, padding = 0 =>  15,//64 x 15 x 15 x 15 
        #15, kernel = 3, stride = 2, padding = 0 => 6   //128 x 6 x 6 x 6
        #6, kernel = 3, stride = 1, padding = 0 => 4    //256 x 4 x 4 x 4
        #4, kernel = 2, stride = 1, padding = 0 => 3    //512 x 3 x 3 x 3
        #3, kernel = 2, stride = 1, padding = 0 => 2    //1024 x 2 x 2 x 2 
        #2, kernel = 2, stride = 1, padding = 0 => 1    //2048 x 1 x 1 x 1

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(latent_space, 1024, 2),
            nn.ReLU(),
            nn.ConvTranspose3d(1024, 512, 2),
            nn.ReLU(),
            nn.ConvTranspose3d(512, 256, 2),
            nn.ReLU(),
            nn.ConvTranspose3d(256, 128, 3),
            nn.ReLU(),
            nn.ConvTranspose3d(128, 64, 3, stride = 2),
            nn.ReLU(),            
            nn.ConvTranspose3d(64, 32, 3, stride = 2),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 1, 3, stride = 2, padding = 1),
            nn.ReLU(),
        )



    def forward(self, x):
        
        #Batch size, number of images, number of channels per image, height, width
        B, L, C, H, W = x.shape

        x = torch.cat([self.encoder(x[:,i]) for i in range (L)])
        
        x = torch.sum(x, dim = 0)

        x = x.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
        x = self.decoder(x)

        return x
