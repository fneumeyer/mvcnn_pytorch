import os
import sys
import trimesh as tr
from trimesh.voxel import creation # why doesn't it work without this import? 
import numpy as np

def get_voxel_grid_from_mesh(path_to_mesh, resolution=64):
    mesh = tr.load(path_to_mesh)
    scale = 1/max(mesh.extents)
    x = y = z = 0
    for index, vertex in enumerate(mesh.vertices):       
        mesh.vertices[index] = vertex * scale # normalization
        x += mesh.vertices[index][0]
        y += mesh.vertices[index][1]
        z += mesh.vertices[index][2]

    x_avrg = x/len(mesh.vertices)
    y_avrg = y/len(mesh.vertices)
    z_avrg = z/len(mesh.vertices)

    for index, vertex in enumerate(mesh.vertices): # centering
        mesh.vertices[index][0] -= x_avrg
        mesh.vertices[index][1] -= y_avrg
        mesh.vertices[index][2] -= z_avrg

    # second argument is the voxel size - for some reason I had to devide 1 by 63 to get 64 voxels (still I got 65 voxels on one of the meshes) 
    grid = tr.voxel.creation.voxelize(mesh, 0.015873) 
    grid = grid.matrix # the actual values for the voxels as a numpy array

    new_grid = [[[False] * resolution] * resolution] * resolution
    new_grid = np.array(new_grid)

    d1, d2, d3 = grid.shape
    for i in range(d1 if d1 <= resolution else resolution):
        for j in range(d2 if d2 <= resolution else resolution):
            for k in range(d3 if d3 <= resolution else resolution):
                new_grid[i][j][k] = grid[i][j][k]

    return new_grid


path_to_dataset = r"C:\Projects\mvcnn_pytorch\ModelNet40"
voxelized_dataset_path = r"ModelNet40Voxelized"

if not os.path.exists(voxelized_dataset_path):
    os.makedirs(voxelized_dataset_path)

classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
                         'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
                         'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
                         'person','piano','plant','radio','range_hood','sink','sofa','stairs',
                         'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']

for classname in classnames:
    if not os.path.exists(os.path.join(voxelized_dataset_path, classname)):
        os.makedirs(os.path.join(voxelized_dataset_path, classname))
    print("")
    for split in ["train", "test"]:
        print("")
        i = 0
        n_files = len(os.listdir(os.path.join(path_to_dataset, classname, split)))
        for filename in os.listdir(os.path.join(path_to_dataset, classname, split)):
            if not os.path.exists(os.path.join(voxelized_dataset_path, classname, split)):
                os.makedirs(os.path.join(voxelized_dataset_path, classname, split))

            if not os.path.exists(os.path.join(voxelized_dataset_path, classname, split, filename[:-4]+".npy")):
                if not os.path.exists(os.path.join(voxelized_dataset_path, classname, split, filename[:-4]+".npy")):
                    try:
                        grid = get_voxel_grid_from_mesh(os.path.join(path_to_dataset, classname, split, filename))
                        np.save(os.path.join(voxelized_dataset_path, classname, split, filename[:-4]), grid)        
                    except Exception as ex:
                        i += 1
                        print(ex)
                        print("\File {} out of {} in {} wasn't written due to memory error!\n".format(i, n_files, os.path.join(voxelized_dataset_path, classname, split)))                        
                        continue        

            i += 1
            sys.stdout.write("\rWrote file {} out of {} in {}".format(i, n_files, os.path.join(voxelized_dataset_path, classname, split)))
            sys.stdout.flush()
