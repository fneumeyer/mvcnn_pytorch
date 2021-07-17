import torch
import os

from PIL import Image
from torchvision import transforms
import numpy as np

class TheDataset(torch.utils.data.Dataset):
    def __init__(self, data_path_2D, data_path_3D, split, num_views=12, num_models = 0):
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
            if not os.path.exists(os.path.join(self.data_path_2D, classname, split)):
                continue

            for filename in os.listdir(os.path.join(self.data_path_2D, classname, split)):                
                self.filepaths_2D.append(os.path.join(self.data_path_2D, classname, split, filename))
        self.filepaths_2D.sort()

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
            if not os.path.exists(os.path.join(self.data_path_2D, classname, split)):
                continue
            
            for filename in os.listdir(os.path.join(self.data_path_3D, classname, split)):                
                self.filepaths_3D.append(os.path.join(self.data_path_3D, classname, split, filename))
        self.filepaths_3D.sort()

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

        # center the shape in the voxel grid: first find the bounding box...
        x_occupied = np.argwhere(grid.max((1,2)))
        y_occupied = np.argwhere(grid.max((0,2)))
        z_occupied = np.argwhere(grid.max((0,1)))
        corner1 = np.array([x_occupied.min(), y_occupied.min(), z_occupied.min()])
        corner2 = np.array([x_occupied.max()+1, y_occupied.max()+1, z_occupied.max()+1])

        # ...find the corners of the new bounding box...
        center = np.array(grid.shape) // 2
        new_corner1 = center - (corner2 - corner1) // 2
        new_corner2 = center + (corner2 - corner1 + 1) // 2

        # ...and write the shape into the new bounding box
        centered_grid = np.zeros_like(grid)
        centered_grid[new_corner1[0]:new_corner2[0], new_corner1[1]:new_corner2[1], new_corner1[2]:new_corner2[2]
            ] = grid[corner1[0]:corner2[0], corner1[1]:corner2[1], corner1[2]:corner2[2]]

        grid = torch.from_numpy(centered_grid).unsqueeze(0)
        
        return (class_name, stacked_images, grid)


