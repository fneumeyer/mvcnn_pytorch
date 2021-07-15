import torch
import os

class TheDataset(torch.utils.data.Dataset):
    def __init__(self, data_path_2D, data_path_3D='', num_views=12, num_models = 0):
        self.classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
                    'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
                    'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
                    'person','piano','plant','radio','range_hood','sink','sofa','stairs',
                    'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']
        self.data_path_2D = data_path_2D
        self.data_path_3D = data_path_3D
        self.num_views = num_views
        self.filepaths_2D = []

        for classname in self.classnames:    
            for split in ["train", "test"]:
                for filename in os.listdir(os.path.join(self.data_path_2D, classname, split)):                
                    self.filepaths_2D.append(os.path.join(self.data_path_2D, classname, split, filename))    

        if num_models != 0: # Use only a part of the dataset
            self.filepaths_2D = self.filepaths_2D[:min(num_models,len(self.filepaths_2D))]

        # Select subset for different number of views
        # stride = int(12/self.num_views) # 12 6 4 3 2 1
        # all_files = all_files[::stride]
        
        # print(self.filepaths)

    def __len__(self):
        return int(len(self.filepaths_2D)/self.num_views) 

    def __getitem__(self, idx):
        pass



data = TheDataset(data_path_2D=r"C:\Projects\mvcnn_pytorch\ModelNet40")