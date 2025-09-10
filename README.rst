.. role:: bash(code)
   :language: bash


Lookup multivariate Kolmogorov-Arnold Networks
==============================================

A general-purpose drop-in replacement for high-dimensional linear mappings (linear layers) with better capacity -- inference cost ratio. 

.. image:: /figures/performance.svg
   :alt: Performance of lmKANs
   :align: center


+++++++++++++
Installation
+++++++++++++
Make sure you have numpy and CUDA-enabled PyTorch installed.

Run :bash:`pip install .`

This command will trigger in-place compilation of the CUDA kernels, so you will need the ``nvcc`` compiler (CUDA toolkit) at hand. 

+++++++++++++
Usage
+++++++++++++

.. code-block:: python

    import torch
    from lmKAN import LMKAN2DLayer

    NUM_GRIDS = 26
    BATCH_SIZE = 1024
    INPUT_DIM = 256 # must be divisible by tile_size_forward and tile_size_backward
    OUTPUT_DIM = 256 # must be divisible by tile_size_forward and tile_size_backward

    lmkan_layer = LMKAN2DLayer(
        num_grids=NUM_GRIDS,
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        tile_size_forward=8,
        tile_size_backward=4,
    ).cuda()

    x = torch.randn(INPUT_DIM, BATCH_SIZE).cuda()  # lmKANs use batch-last data layout
    out = lmkan_layer(x)  # out has shape [OUTPUT_DIM, BATCH_SIZE]

The number of trainable parameters is ``(NUM_GRIDS + 1)^2 * OUTPUT_DIM * (INPUT_DIM // 2)``. Inference FLOPs are *2x* of that of a linear layer of the same shape (same ``INPUT_DIM`` and ``OUTPUT_DIM``), **not** depending on ``NUM_GRIDS``. 

For the example above, you get as many as ``(26 + 1)^2 / 2 ~ 360x`` more trainable parameters than that of a linear layer by paying 2x more formal FLOPs and ~10x wall-clock inference time (depending on the specific GPU). 

The computational cost of backward also doesn't depend on ``NUM_GRIDS``. 

**Important** check performance caveats and recommendations on hyperparameters and fitting procedure below. 

``NUM_GRIDS`` is not unlimited. The maximal value depends on the GPU type, specifically on the amount of available shared memory, and ``tile_size``. 

For H100 GPU and tile ``tile_size=8``, the maximal values of ``NUM_GRIDS``  are 40 and 26 for forward and backward, respectively. 

For RTX 3080 (and all the other GPUs with 128KB shared memory) and tile ``tile_size=8``, the maximal values of ``NUM_GRIDS``  are 26 and 18 for forward and backward. 

We implement ``tile_size`` of 4, 8, and 16. The larger the ``tile_size`` is, the faster the layer, but the smaller the upper limit on ``NUM_GRIDS``.
For instance, on the H100 GPU, inference time is ~8x of that of a linear layer of the same shape when ``forward_tile_size=16``, and ~9.5x when ``forward_tile_size=8``.

++++++++++++++++++++++++
 Fitting recommendations
++++++++++++++++++++++++

The static percentile grids we employ suggest using batch normalization with ``affine=False`` before each of the lmKAN layers, see more details in section 3.1 of the paper. 

Given that the computational cost doesn't depend on ``NUM_GRIDS``,  it is tempting to make it large. lmKANs, however, were observed to be hard to converge when ``NUM_GRIDS`` is excessively large. Therefore, we recommend first starting with a reasonable grid resolution - ``NUM_GRIDS=8-10``, and next increase it after getting a working example. 

The Hessian regularization, which pushes all the inner functions to be smooth and slowly varying, and which we proposed in section 3.5 of the paper, can be used not only to facilitate generalizability but also to stabilize the training dynamics, and it is quite effective in doing so. 

Each lmKAN layer has the method ``def get_hessian_regularization(self):``. Sum these values across all the lmKAN layers of the model, and next add to the loss ``lambda * total_hessian_regularization``, where ``lambda`` is the regularization strength. 

We found it effective to first use very strong Hessian regularization, and next gradually relax it during the fitting. So, essentially, the model first learns low-frequency components, and next, when they are learned, enables high-frequency ones. As an example, see the following fitting schedule:


.. code-block:: python

    def get_fitting_schedule(epoch_number: int):
        HESSIAN_DECAY_EPOCHS = 250
        HESSIAN_DECAY_SCALE = 30
        COSINE_EPOCHS = 250
        BASE_LR = 1e-3
        INITIAL_HESSIAN_WEIGHT = 1.0


        if epoch_number <= HESSIAN_DECAY_EPOCHS:
            # Phase 1: constant LR, Hessian decay
            learning_rate = BASE_LR
            hessian_regularization_lambda = INITIAL_HESSIAN_WEIGHT / (10 ** ((epoch_number / HESSIAN_DECAY_SCALE)))
        else:
            # Phase 2: LR cosine decay
            offset = epoch_number - HESSIAN_DECAY_EPOCHS
            T = COSINE_EPOCHS
            learning_rate = 0.5 * BASE_LR * (1.0 + math.cos(math.pi * offset / (COSINE_EPOCHS)))
            
            hessian_regularization_lambda = 0.0

        return learning_rate, hessian_regularization_lambda


The hypers in the function above represent a good starting point for the methane dataset (Cartesian Components representation, see section 4.2). The resulting model is not so accurate, as we report, but fitting takes less than one hour, and it is already sufficient to become both FLOPs and wall-clock time Pareto optimal compared to MLPs. 

++++++++++++++++++++++++
Preprint
++++++++++++++++++++++++

Lookup multivariate Kolmogorov-Arnold Networks. 

https://arxiv.org/abs/2509.07103
