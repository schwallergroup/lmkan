import torch
import torch.nn as nn
from .custom_op import lmKAN_2d_op
import math
from .utilities import get_borders_cdf_grid, get_frobenius_regularization


class LMKAN2DLayer(nn.Module):
    def __init__(
        self,
        num_grids,
        input_dim,
        output_dim,
        block_size_forward=1024,
        block_size_backward=512,
        tile_size_forward=8,
        tile_size_backward=8,
        init_scale=0.1,
        backward_fast_mode=False,
    ):
        super(LMKAN2DLayer, self).__init__()

        # Ensure that input_dim is divisible by 2.
        if input_dim % 2 != 0:
            raise ValueError("input_dim must be divisible by 2")

        self.num_grids = num_grids
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.block_size_forward = block_size_forward
        self.block_size_backward = block_size_backward
        self.tile_size_forward = tile_size_forward
        self.tile_size_backward = tile_size_backward
        self.backward_fast_mode = backward_fast_mode

        # Create the func_parameter with shape:
        # [num_grids + 1, num_grids + 1, output_dim, input_dim // 2]
        self.func_parameter = nn.Parameter(
            torch.empty(
                num_grids + 1,
                num_grids + 1,
                output_dim,
                input_dim // 2,
            )
        )

        nn.init.uniform_(
            self.func_parameter,
            -init_scale / math.sqrt(input_dim),
            init_scale / math.sqrt(input_dim),
        )

        self.register_buffer(
            "borders",
            torch.tensor(
                get_borders_cdf_grid(num_grids), dtype=torch.float32
            ),
        )

    def forward(self, x):
        # x has shape [input_dim, batch_size] in batch_last mode
        return lmKAN_2d_op(
            self.func_parameter,
            x,
            self.block_size_forward,
            self.block_size_backward,
            self.tile_size_forward,
            self.tile_size_backward,
            True,
            True,  # batch_last parameter
            self.backward_fast_mode,
        )

    def get_hessian_regularization(self):
        return get_frobenius_regularization(self.func_parameter, self.borders)