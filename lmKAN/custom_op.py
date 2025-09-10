from torch.autograd import Function
import lmKAN_kernels  # your custom extension module


class LMKAN_2D_OP(Function):
    @staticmethod
    def forward(ctx, parameters, x, block_size_forward, block_size_backward,
                tile_size_forward, tile_size_backward, cdf_grid,
                batch_last, backward_fast_mode):
        # Save backward-related info and tensors
        ctx.block_size_backward = block_size_backward
        ctx.tile_size_backward = tile_size_backward
        ctx.cdf_grid = cdf_grid
        ctx.batch_last = batch_last
        ctx.backward_fast_mode = backward_fast_mode
        ctx.save_for_backward(parameters, x)

        # Call the forward custom function with forward-specific block and tile sizes
        output = lmKAN_kernels.forward_dkan_full_2d(
            parameters, x,
            block_size_forward,
            cdf_grid,
            tile_size_forward,
            batch_last
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        # Retrieve saved tensors and backward parameters
        parameters, x = ctx.saved_tensors
        block_size_backward = ctx.block_size_backward
        tile_size_backward = ctx.tile_size_backward
        cdf_grid = ctx.cdf_grid
        batch_last = ctx.batch_last
        backward_fast_mode = ctx.backward_fast_mode

        # Call the backward custom function with backward-specific block and tile sizes
        # This should return gradients for parameters and x
        grad_parameters, grad_x = lmKAN_kernels.backward_dkan_full_2d(
            parameters, x,
            grad_output,
            block_size_backward,
            cdf_grid,
            tile_size_backward,
            batch_last,
            fast_mode = backward_fast_mode
        )

        # Return gradients for each input of forward (9 total):
        # parameters, x, block_size_forward, block_size_backward,
        # tile_size_forward, tile_size_backward, cdf_grid,
        # batch_last, backward_fast_mode
        return grad_parameters, grad_x, None, None, None, None, None, None, None

# Convenience wrapper to use the custom autograd Function


def lmKAN_2d_op(parameters, x, block_size_forward, block_size_backward,
               tile_size_forward, tile_size_backward, cdf_grid,
               batch_last, backward_fast_mode):
    return LMKAN_2D_OP.apply(
        parameters, x, block_size_forward, block_size_backward,
        tile_size_forward, tile_size_backward, cdf_grid, batch_last, backward_fast_mode
    )
