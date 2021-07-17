# %%
from pathlib import Path

import numpy as np
import k3d
from matplotlib import cm, colors
import trimesh


def visualize_occupancy(occupancy_grid, flip_axes=False):
    point_list = np.concatenate([c[:, np.newaxis] for c in np.where(occupancy_grid)], axis=1)
    visualize_pointcloud(point_list, 1, flip_axes, name='occupancy_grid')


def visualize_pointcloud(point_cloud, point_size, flip_axes=False, name='point_cloud'):
    plot = k3d.plot(name=name, grid_visible=False, grid=(-0.55, -0.55, -0.55, 0.55, 0.55, 0.55))
    if flip_axes:
        point_cloud[:, 2] = point_cloud[:, 2] * -1
        point_cloud[:, [0, 1, 2]] = point_cloud[:, [0, 2, 1]]
    plt_points = k3d.points(positions=point_cloud.astype(np.float32), point_size=point_size, color=0xd0d0d0)
    plot += plt_points
    plt_points.shader = '3d'
    plot.display()

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
    print(x_max)

    xyz = []
    for x in range(64):
        for y in range(64):
            for z in range(64):        
                if grid[x][y][z]: 
                    grid[x][y][z] = False
                    xyz.append((x,y,z))

    for point in xyz:
        grid[point[0]+min(x_avrg, 63 - x_max)][point[1]][point[2]] = True

def add_corners(grid):
    grid[0][0][0] = True
    grid[63][0][0] = True
    grid[0][63][0] = True
    grid[63][63][0] = True
    grid[0][0][63] = True
    grid[63][0][63] = True
    grid[0][63][63] = True
    grid[63][63][63] = True

# %%
grid = np.load(r"C:\Projects\mvcnn_pytorch\ModelNet40Centered\airplane\train\airplane_0029.npy")
add_corners(grid)

visualize_occupancy(grid)
# %%
