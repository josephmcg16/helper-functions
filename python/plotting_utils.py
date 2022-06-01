import numpy as np

from scipy.interpolate import griddata

import matplotlib.pyplot as plt


def grid_matrices3D(arr, n_points):
    """Return interpolated grid arrays from samples of a 3D array.
    Similar to mesh grid but for multiple arrays. Also checks for 
    NaNs when interpolated dimension is outside of the bounds.
    (This is still dodgy...)

    Args:
        arr (ndarray): Data matrix representing 3D surface
        n_points (_type_): Number of interpolated points

    Returns:
        (ndarray, ndarray, ndarray): Mesh grids for each dimension
    """
    x = arr[0]
    y = arr[1]

    if arr.shape[0] == 2:
        z = np.zeros(arr.shape[1])
    else:
        z = arr[2]

    xi = np.linspace(min(x), max(x), num=n_points)
    yi = np.linspace(min(y), max(y), num=n_points)
    x_grid, y_grid = np.meshgrid(xi, yi)
    z_grid = griddata((x, y), z, (x_grid, y_grid), method='cubic')

    inds = np.isnan(z_grid)

    z_grid_nonan = z_grid[~inds]
    n = int(np.sqrt(len(z_grid_nonan)))

    if n**2 != float(len(z_grid_nonan)):
        return x_grid, y_grid, z_grid

    z_grid = z_grid_nonan.reshape(n, n)
    x_grid = x_grid[~inds].reshape(n, n)
    y_grid = y_grid[~inds].reshape(n, n)

    return x_grid, y_grid, z_grid


def plot_trajectory(trajectory):
    """3d plot of a single trajectory in cartesian coordinates"""
    fig, axes = plt.subplots(subplot_kw={'projection': '3d'})
    axes.scatter(0, 0, 0)
    axes.plot(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2],
              marker='*', markersize=7)
    axes.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2])
    return fig, axes
