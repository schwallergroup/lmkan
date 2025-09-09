import numpy as np
import torch


def direct_grid_function(x):
    absolute = np.abs(x)
    value = 0.5 * np.exp(-absolute)
    if x > 0:
        return 1.0 - value
    else:
        return value


def inverse_grid_function(x):
    if x <= 0.5:
        return np.log(2.0 * x)
    else:
        return -np.log(2.0 * (1.0 - x))


def get_borders_cdf_grid(n_chunks):
    chunk_size = 1.0 / n_chunks
    borders = []
    for i in range(1, n_chunks):
        level_now = i * chunk_size
        borders.append(inverse_grid_function(level_now))
    left_most = borders[0] - (borders[1] - borders[0])
    right_most = borders[-1] + (borders[-1] - borders[-2])
    return [left_most] + borders + [right_most]


@torch.compile
def compute_hessian_frobenius_norm(func: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """
    Compute the squared Frobenius norm of the Hessian of a 2D function defined on a non-uniform grid.
    The function 'func' can have extra trailing dimensions, and the Hessian is computed independently
    for all of them.

    The squared Frobenius norm of the Hessian is given by:

        ||H(f)||_F^2 = f_xx^2 + 2 f_xy^2 + f_yy^2,

    where f_xx and f_yy are the second derivatives in the x and y directions,
    and f_xy is the mixed derivative computed using a non-uniform finite difference scheme.

    Parameters:
    -----------
    func : torch.Tensor
        A tensor of shape [grid_size, grid_size, ...] where the first two dimensions
        correspond to the grid in x and y, and additional dimensions index independent
        functions.
    grid : torch.Tensor
        A 1D tensor of shape [grid_size] with strictly increasing grid coordinates.

    Returns:
    --------
    torch.Tensor:
        The squared Frobenius norm of the Hessian computed on the inner grid points, with shape
        [grid_size - 2, grid_size - 2, ...], preserving extra dimensions.
    """
    N = grid.shape[0]
    extra_dims = func.dim() - 2  # number of extra dimensions

    # Prepare view shapes for broadcasting grid spacings with extra dimensions.
    view_shape_x = (-1, 1) + (1,) * extra_dims  # for x-direction spacing: shape [N-2, 1, 1, ...]
    view_shape_y = (1, -1) + (1,) * extra_dims  # for y-direction spacing: shape [1, N-2, 1, ...]

    # Compute horizontal grid spacings (for the x-direction)
    h_left_x = grid[1:-1] - grid[:-2]   # shape: (N-2,)
    h_right_x = grid[2:] - grid[1:-1]     # shape: (N-2,)
    h_left_x = h_left_x.view(*view_shape_x)
    h_right_x = h_right_x.view(*view_shape_x)

    # Compute vertical grid spacings (for the y-direction)
    h_bottom_y = grid[1:-1] - grid[:-2]   # shape: (N-2,)
    h_top_y    = grid[2:] - grid[1:-1]     # shape: (N-2,)
    h_bottom_y = h_bottom_y.view(*view_shape_y)
    h_top_y    = h_top_y.view(*view_shape_y)

    # Slices for the inner grid points.
    # For second derivatives in x and y we use the neighbors in the same row/column.
    u_center = func[1:-1, 1:-1]  # shape: (N-2, N-2, ...)
    u_right  = func[2:  , 1:-1]  # f(x+h_right, y)
    u_left   = func[:-2 , 1:-1]  # f(x-h_left, y)
    u_top    = func[1:-1, 2:  ]  # f(x, y+h_top)
    u_bottom = func[1:-1, :-2]   # f(x, y-h_bottom)

    # Second derivative in x (f_xx)
    f_xx = 2.0 / (h_left_x + h_right_x) * (
              (u_right - u_center) / h_right_x -
              (u_center - u_left)  / h_left_x )

    # Second derivative in y (f_yy)
    f_yy = 2.0 / (h_bottom_y + h_top_y) * (
              (u_top - u_center)   / h_top_y -
              (u_center - u_bottom)/ h_bottom_y )

    # For the mixed derivative f_xy, use the four diagonal neighbors.
    # f(x+h_right, y+h_top), f(x+h_right, y-h_bottom), f(x-h_left, y+h_top), f(x-h_left, y-h_bottom)
    f_xy = (func[2:, 2:]    - func[2:, :-2] -
            func[:-2, 2:]   + func[:-2, :-2]) / (
            (h_left_x + h_right_x) * (h_bottom_y + h_top_y) )

    # Squared Frobenius norm of the Hessian: f_xx^2 + 2 f_xy^2 + f_yy^2.
    hessian_norm_sq = f_xx**2 + 2.0 * (f_xy**2) + f_yy**2
    return hessian_norm_sq

@torch.compile
def get_frobenius_regularization(func: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    hessian_norm_sq = compute_hessian_frobenius_norm(func, grid)
    return torch.mean(hessian_norm_sq, dim=(0,1)).sum()
