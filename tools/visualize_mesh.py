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

# %%
grid = np.load(r"C:\Projects\mvcnn_pytorch\ModelNet40Voxelized\guitar\train\guitar_0009.npy")
visualize_occupancy(grid)
# %%
