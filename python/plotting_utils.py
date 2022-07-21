import numpy as np

from scipy.interpolate import griddata

import matplotlib.pyplot as plt

import plotly.graph_objects as go


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


def hexcolor(value, max_value):
    """Returns Hex scale of RGB values for a generic value

    Example:
    `
    values = range(10)
    colors = [hexcolor(value, values.max()) for value in values]
    `

    Args:
        value (float, int): Generic value to be converted
        max_value (float, int): Maximum value of the range of values which value belongs to.

    Returns:
        str: Hex code of RGB values.
    """
    value = value / max_value * 100

    if value <= 50:
        r = 255
        g = int(255*value/50)
        b = 0
    else:
        r = int(255*(100-value)/50)
        g = 255
        b = 0

    return "#%s%s%s" % tuple([hex(c)[2:].rjust(2, "0") for c in (r, g, b)])


def plotly_trajectories_3d(data_matrix, BGCOLOR="#1E1E1E"):
    """Plot multiple trajectories of time series 3-dimensional data.

    Args:
        data_matrix (np.ndarray): Data matrix to plot. 
        Shape is (number of trajectories, number of time points, number of spatial dimensions)
        BGCOLOR (string): Hex string of the background color of plotly figure. Defaults to dark mode.

    Returns:
        fig (plotly.graph_objs._figure.Figure): Plotly graph object with the 3D plot of the data matrix.
    """
    fig = go.Figure()
    for i, traj in enumerate(data_matrix):
      fig.add_trace(
          go.Scatter3d(x=traj[:, 0], y=traj[:, 1], z=traj[:, 2],
                      mode="lines", name=f"Trajectory {i+1}")
      )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        template='plotly_dark',
        paper_bgcolor=BGCOLOR,
        legend=dict(
          yanchor="top",
          y=1,
          xanchor="left",
          x=0)
      )
    return fig
