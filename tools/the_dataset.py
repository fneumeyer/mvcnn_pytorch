import torch
import os

from PIL import Image
from torchvision import transforms
import numpy as np

class TheDataset(torch.utils.data.Dataset):
    def __init__(self, data_path_2D, data_path_3D, num_views=12, num_models = 0):
        self.classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
                    'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
                    'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
                    'person','piano','plant','radio','range_hood','sink','sofa','stairs',
                    'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']
        self.data_path_2D = data_path_2D
        self.data_path_3D = data_path_3D
        self.num_views = num_views
        self.filepaths_2D = []

        self.filepaths_3D = []

        self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(), # do we want this? 
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], # not sure what those are doing
                                     std=[0.229, 0.224, 0.225])
            ])

        for classname in self.classnames:    
            for split in ["train", "test"]:
                for filename in os.listdir(os.path.join(self.data_path_2D, classname, split)):                
                    self.filepaths_2D.append(os.path.join(self.data_path_2D, classname, split, filename))    

        if num_models != 0: # Use only a part of the dataset
            self.filepaths_2D = self.filepaths_2D[:min(num_models,len(self.filepaths_2D))]

        # Select subset for different number of views
        if self.num_views == 12:
            pass
        elif self.num_views in [6,4,3,2,1]:
            stride = int(12/self.num_views)
            self.filepaths_2D = self.filepaths_2D[::stride]
        else:
            raise Exception("Invalid number of views")

        # 3D
        for classname in self.classnames:    
            for split in ["train", "test"]:
                for filename in os.listdir(os.path.join(self.data_path_3D, classname, split)):                
                    self.filepaths_3D.append(os.path.join(self.data_path_3D, classname, split, filename))

    def __len__(self):
        return int(len(self.filepaths_3D)) 

    def __getitem__(self, idx):
        # 2D        
        path = self.filepaths_2D[idx*self.num_views - self.num_views + 1] 
        path = os.path.normpath(path) 
        class_name = path.split(os.sep)[-3]
        #class_id = self.classnames.index(class_name)                        

        imgs = []
        for i in range(self.num_views):
            im = Image.open(self.filepaths_2D[idx*self.num_views - self.num_views + 1 + i]).convert('RGB')            
            if self.transform:
                im = self.transform(im)
            imgs.append(im)
        
        stacked_images = torch.stack(imgs)

        # 3D
        grid = np.load(self.filepaths_3D[idx])
        grid = torch.from_numpy(grid)
        
        return (class_name, stacked_images, grid)


