import sys
import os 
import numpy as np

def center_z(grid):
    z_sum = 0
    voxels = 0
    z_max = 0
    for x in range(64):
        for y in range(64):
            for z in range(64):
                if grid[x][y][z]:
                    z_sum += (32-z)
                    voxels += 1
                    if z > z_max:
                        z_max = z

    z_avrg = int(z_sum/voxels)    

    xyz = []
    for x in range(64):
        for y in range(64):
            for z in range(64):        
                if grid[x][y][z]: 
                    grid[x][y][z] = False
                    xyz.append((x,y,z))

    for point in xyz:
        grid[point[0]][point[1]][point[2]+min(z_avrg, 63 - z_max)] = True
        
def center_y(grid):
    y_sum = 0
    voxels = 0
    y_max = 0
    for x in range(64):
        for y in range(64):
            for z in range(64):
                if grid[x][y][z]:
                    y_sum += (32-y)
                    voxels += 1
                    if y > y_max:
                        y_max = y

    y_avrg = int(y_sum/voxels)    

    xyz = []
    for x in range(64):
        for y in range(64):
            for z in range(64):        
                if grid[x][y][z]: 
                    grid[x][y][z] = False
                    xyz.append((x,y,z))

    for point in xyz:
        grid[point[0]][point[1]+min(y_avrg, 63 - y_max)][point[2]] = True

def center_x(grid):
    x_sum = 0
    voxels = 0
    x_max = 0
    for x in range(64):
        for y in range(64):
            for z in range(64):
                if grid[x][y][z]:
                    x_sum += (32-x)
                    voxels += 1
                    if x > x_max:
                        x_max = x

    x_avrg = int(x_sum/voxels)    

    xyz = []
    for x in range(64):
        for y in range(64):
            for z in range(64):        
                if grid[x][y][z]: 
                    grid[x][y][z] = False
                    xyz.append((x,y,z))

    for point in xyz:
        grid[point[0]+min(x_avrg, 63 - x_max)][point[1]][point[2]] = True


path_to_dataset = r"C:\Projects\mvcnn_pytorch\ModelNet40Voxelized"
centered_dataset_path = r"ModelNet40Centered"

if not os.path.exists(centered_dataset_path):
    os.makedirs(centered_dataset_path)

classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
                         'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
                         'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
                         'person','piano','plant','radio','range_hood','sink','sofa','stairs',
                         'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']


for classname in classnames:
    if not os.path.exists(os.path.join(centered_dataset_path, classname)):
        os.makedirs(os.path.join(centered_dataset_path, classname))
    print("")
    for split in ["train", "test"]:
        print("")
        i = 0
        n_files = len(os.listdir(os.path.join(path_to_dataset, classname, split)))
        for filename in os.listdir(os.path.join(path_to_dataset, classname, split)):
            if not os.path.exists(os.path.join(centered_dataset_path, classname, split)):
                os.makedirs(os.path.join(centered_dataset_path, classname, split))

            if not os.path.exists(os.path.join(centered_dataset_path, classname, split, filename[:-4]+".npy")):
                try:
                    grid = np.load(os.path.join(path_to_dataset, classname, split, filename))
                    center_z(grid)
                    center_y(grid)
                    center_x(grid)
                    np.save(os.path.join(centered_dataset_path, classname, split, filename[:-4]), grid)        
                except Exception as ex:
                    i += 1
                    print(ex)                    
                    continue        

            i += 1
            sys.stdout.write("\rWrote file {} out of {} in {}".format(i, n_files, os.path.join(centered_dataset_path, classname, split)))
            sys.stdout.flush()