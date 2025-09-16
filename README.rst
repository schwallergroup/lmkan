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
    INPUT_DIM = 128 # must be divisible by tile_size_forward and tile_size_backward
    OUTPUT_DIM = 128 # must be divisible by tile_size_forward and tile_size_backward

    lmkan_layer = LMKAN2DLayer(
        num_grids=NUM_GRIDS,
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        tile_size_forward=8,
        tile_size_backward=4,
        block_size_forward=1024,
        block_size_backward=512,
    ).cuda()

    x = torch.randn(INPUT_DIM, BATCH_SIZE).cuda()  # lmKANs use batch-last data layout
    output = lmkan_layer(x)  # output has shape [OUTPUT_DIM, BATCH_SIZE]

The number of trainable parameters is ``(NUM_GRIDS + 1)^2 * OUTPUT_DIM * (INPUT_DIM // 2)``. Inference FLOPs are *2x* of that of a linear layer of the same shape (same ``INPUT_DIM`` and ``OUTPUT_DIM``), **not** depending on ``NUM_GRIDS``. 

For the example above, you get as many as ``(26 + 1)^2 / 2 ~ 360x`` more trainable parameters than that of a linear layer of the same shape by paying 2x more formal FLOPs and ~10x wall-clock inference time (depending on the specific GPU). Thanks to such a large number of trainable parameters, lmKANs can achieve high accuracy even with small hidden dimensions, which made them Pareto-optimal relative to MLPs in our experiments. 

**Important** check performance caveats and recommendations on hyperparameters and fitting procedure below. 

``NUM_GRIDS`` is not unlimited. The maximal value depends on the GPU type, specifically on the amount of available shared memory, and ``tile_size``. 

For H100 GPU and tile ``tile_size=8``, the maximal values of ``NUM_GRIDS``  are 40 and 26 for forward and backward, respectively. 

For RTX 3080 (and all the other GPUs with 128KB shared memory) and tile ``tile_size=8``, the maximal values of ``NUM_GRIDS``  are 26 and 18 for forward and backward. 

We implement ``tile_size`` of 4, 8, and 16. The larger the ``tile_size`` is, the faster the layer, but the smaller the upper limit on ``NUM_GRIDS``.
For instance, on the H100 GPU, inference time is ~8x of that of a linear layer of the same shape when ``forward_tile_size=16``, and ~9.5x when ``forward_tile_size=8``.

You might need to set ``block_size_forward=512`` to use ``tile_size=16`` on some GPUs.

The computational cost of backward also doesn't depend on ``NUM_GRIDS``. We have two modes - normal and fast. The latter can be invoked by providing an additional argument ``backward_fast_mode=True``. Fast mode is faster but has higher shared memory requirements (=supported maximal ``NUM_GRIDS`` is smaller).

++++++++++++++++++++++++
 Fitting recommendations
++++++++++++++++++++++++

The static percentile grids we employ suggest using batch normalization with ``affine=False`` before each of the lmKAN layers, see more details in section 3.1 of the paper. 

Given that the computational cost doesn't depend on ``NUM_GRIDS``,  it is tempting to make it large. lmKANs, however, were observed to be hard to converge when ``NUM_GRIDS`` is excessively large. Therefore, we recommend first starting with a reasonable grid resolution - ``NUM_GRIDS=8-10``, and next increase it after getting a working example. 

The Hessian regularization, which pushes all the inner functions to be smooth and slowly varying, and which we proposed in section 3.5 of the paper, can be used not only to facilitate generalizability but also to stabilize the training dynamics, and it is quite effective in doing so. 

Each lmKAN layer has the method ``def get_hessian_regularization(self):``. Sum these values across all the lmKAN layers of the model, and next add to the loss ``lambda * total_hessian_regularization``, where ``lambda`` is the regularization strength. 

We found it effective to first use very strong Hessian regularization, and next gradually relax it during the fitting. So, the model first learns low-frequency components, and next, when they are learned, enables high-frequency ones. As an example, see the following fitting schedule:


.. code-block:: python

    import math
    
    def get_fitting_schedule(epoch_number):
        HESSIAN_DECAY_EPOCHS = 250
        HESSIAN_DECAY_SCALE = 30
        COSINE_EPOCHS = 250
        BASE_LR = 1e-3
        INITIAL_HESSIAN_WEIGHT = 1.0


        if epoch_number <= HESSIAN_DECAY_EPOCHS:
            # Phase 1: constant LR, Hessian decay
            learning_rate = BASE_LR
            hessian_regularization_lambda = INITIAL_HESSIAN_WEIGHT / (10 ** (epoch_number / HESSIAN_DECAY_SCALE))
        else:
            # Phase 2: LR cosine decay
            offset = epoch_number - HESSIAN_DECAY_EPOCHS
            T = COSINE_EPOCHS
            learning_rate = 0.5 * BASE_LR * (1.0 + math.cos(math.pi * offset / (COSINE_EPOCHS)))
            
            hessian_regularization_lambda = 0.0

        return learning_rate, hessian_regularization_lambda


The hypers in the function above represent a good starting point for the methane dataset (Cartesian Components representation, see section 4.2). The resulting model is not as accurate as the tightly converged one reported in the paper, but fitting takes less than one hour, and it is already sufficient to become both FLOPs and wall-clock time Pareto optimal compared to MLPs. 

++++++++++++++++++++++++
Efficiency analysis
++++++++++++++++++++++++
| GPU: H100 SXM
| BATCH_SIZE: 1048576 (2^20)
| INPUT_DIM: 512
| OUTPUT_DIM: 512
| TILE_SIZE_FORWARD: 16
| TILE_SIZE_BACKWARD: 4
| BLOCK_SIZE_FORWARD: 512
| BLOCK_SIZE_BACKWARD: 512
| 
| MEASURED TIME: ~86ms
| 

A little math: total number of 2D functions to compute is BATCH_SIZE * (INPUT_DIM / 2) * OUTPUT_DIM. Each requires 4 float32 reads from shared memory. Each float32 is 4 bytes. Therefore, the actual shared-memory throughput is ~(1048576 * 256 * 512 * 4 * 4 / 0.086)≈ 23 TB/s. It is ~70% of the theoretical peak of 33 TB/s shared-memory bandwidth of the H100 SXM, see, for instance, here https://hazyresearch.stanford.edu/blog/2024-05-12-tk.

++++++++++++++++++++++++
Preprint
++++++++++++++++++++++++

.. code-block:: bibtex

  @article{pozdnyakov2025lookup,
    title={Lookup multivariate Kolmogorov-Arnold Networks},
    author={Pozdnyakov, Sergey and Schwaller, Philippe},
    journal={arXiv preprint arXiv:2509.07103},
    year={2025}
  }
