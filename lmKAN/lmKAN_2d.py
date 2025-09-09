import torch
import torch.nn as nn
from .custom_op import lmKAN_2d_op
import math
from .utilities import get_borders_cdf_grid, get_frobenius_regularization

class LMKAN_2D_Layer(nn.Module):
    def __init__(self, n_chunks, input_dim, output_dim,
                 block_size_forward, block_size_backward,
                 tile_size_forward, tile_size_backward,
                 apply_scale, apply_bias, cdf_grid, apply_tanh,
                 init_scale, batch_last, backward_fast_mode):
        super(LMKAN_2D_Layer, self).__init__()

        # Ensure that input_dim is divisible by 2.
        if input_dim % 2 != 0:
            raise ValueError("input_dim must be divisible by 2")

        if not batch_last:
            raise ValueError("batch_last must be True")

        self.n_chunks = n_chunks
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.block_size_forward = block_size_forward
        self.block_size_backward = block_size_backward
        self.tile_size_forward = tile_size_forward
        self.tile_size_backward = tile_size_backward
        self.apply_scale = apply_scale
        self.cdf_grid = cdf_grid
        self.apply_tanh = apply_tanh
        self.apply_bias = apply_bias
        self.batch_last = batch_last
        self.backward_fast_mode = backward_fast_mode
        # Create the func_parameter with shape: [n_chunks + 1, n_chunks + 1, output_dim, input_dim // 2]
        self.func_parameter = nn.Parameter(
            torch.empty(n_chunks + 1, n_chunks + 1, output_dim, input_dim // 2)
        )

        nn.init.uniform_(self.func_parameter, -init_scale / math.sqrt(input_dim), init_scale / math.sqrt(input_dim))

        if apply_scale:
            # Create the scale_parameters with shape: [input_dim]
            self.scale_parameters = nn.Parameter(torch.ones(input_dim))

        if apply_bias:
            # Create the bias_parameters with shape: [input_dim]
            self.bias_parameters = nn.Parameter(torch.zeros(input_dim))

        # Parameters for the linear mapping adjusted for batch_last mode
        # W_linear: weight matrix of shape [output_dim, input_dim] (pre-transposed)
        self.W_linear = nn.Parameter(torch.empty(output_dim, input_dim))
        nn.init.kaiming_uniform_(self.W_linear, a=math.sqrt(5))
        self.bias_linear = nn.Parameter(torch.empty(output_dim))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W_linear)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias_linear, -bound, bound)

        self.register_buffer('borders', torch.tensor(get_borders_cdf_grid(n_chunks), dtype=torch.float32))

    def forward(self, x, weight_lmKAN, apply_relu_linear, relu_last = True):
        # x has shape [input_dim, batch_size] in batch_last mode

        if relu_last:
            # Linear transformation for batch_last mode (using pre-transposed weights)
            linear_out = torch.matmul(self.W_linear, x) + self.bias_linear.unsqueeze(1)

            if apply_relu_linear:
                linear_out = nn.functional.relu(linear_out)
        else:
            if apply_relu_linear:
                linear_out = nn.functional.relu(x)
            else:
                linear_out = x

            linear_out = torch.matmul(self.W_linear, linear_out) + self.bias_linear.unsqueeze(1)
        
        # Scale the input: broadcasting the scale_parameters over the batch dimension.
        if self.apply_scale:
            x = x * self.scale_parameters.unsqueeze(1)  # Broadcasting over batch dimension

        # Add the bias: broadcasting the bias_parameters over the batch dimension.
        if self.apply_bias:
            x = x + self.bias_parameters.unsqueeze(1)  # Broadcasting over batch dimension

        # Apply tanh if enabled.
        if self.apply_tanh:
            x = torch.tanh(x)

        # Compute the DKAN operation with batch_last=True
        lmKAN_out = lmKAN_2d_op(
            self.func_parameter,
            x,
            self.block_size_forward,
            self.block_size_backward,
            self.tile_size_forward,
            self.tile_size_backward,
            self.cdf_grid,
            True,  # batch_last parameter
            self.backward_fast_mode
        )

        # Combine the linear mapping and the DKAN operation,
        # scaling the DKAN part by weight_dkan.
        return (1.0) * linear_out + weight_lmKAN * lmKAN_out

    def get_frobenius_regularization(self):
        return get_frobenius_regularization(self.func_parameter, self.borders)

class PureLMKAN_2D_Layer(nn.Module):
    def __init__(
        self,
        n_chunks,
        input_dim,
        output_dim,
        block_size_forward,
        block_size_backward,
        tile_size_forward,
        tile_size_backward,
        cdf_grid,
        init_scale,
        batch_last,
        backward_fast_mode,
    ):
        super(PureLMKAN_2D_Layer, self).__init__()

        # Ensure that input_dim is divisible by 2.
        if input_dim % 2 != 0:
            raise ValueError("input_dim must be divisible by 2")

        if not batch_last:
            raise ValueError("batch_last must be True")

        self.n_chunks = n_chunks
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.block_size_forward = block_size_forward
        self.block_size_backward = block_size_backward
        self.tile_size_forward = tile_size_forward
        self.tile_size_backward = tile_size_backward
        self.cdf_grid = cdf_grid
        self.batch_last = batch_last
        self.backward_fast_mode = backward_fast_mode

        # Create the func_parameter with shape:
        # [n_chunks + 1, n_chunks + 1, output_dim, input_dim // 2]
        self.func_parameter = nn.Parameter(
            torch.empty(
                n_chunks + 1,
                n_chunks + 1,
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
                get_borders_cdf_grid(n_chunks), dtype=torch.float32
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
            self.cdf_grid,
            True,  # batch_last parameter
            self.backward_fast_mode,
        )

    def get_frobenius_regularization(self):
        return get_frobenius_regularization(self.func_parameter, self.borders)