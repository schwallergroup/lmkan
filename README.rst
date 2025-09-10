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

    NUM_GRIDS = 30
    BATCH_SIZE = 1024
    INPUT_DIM = 256
    OUTPUT_DIM = 256

    lmkan_layer = LMKAN2DLayer(
        num_grids=NUM_GRIDS,
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        tile_size_forward=8,
        tile_size_backward=4,
    ).cuda()

    x = torch.randn(INPUT_DIM, BATCH_SIZE).cuda()  # lmKANs use batch-last data layout
    out = lmkan_layer(x)  # out has shape [OUTPUT_DIM, BATCH_SIZE]

The number of trainable parameters is ``(NUM_GRIDS + 1)^2 * OUTPUT_DIM * (INPUT_DIM // 2)``. Inference FLOPs are *2x* of that of a linear layer of the same shape (``INPUT_DIM``, ``OUTPUT_DIM``), **not** depending on ``NUM_GRIDS``. 

For the example above, you get as many as ``(30 + 1)^2 // 2 ~ 480`` more trainable parameters by paying 2x of FLOPs and 6-10x wall-clock inference time (depending on the specific GPU). 

The computational cost of backward also doesn't depend on ``NUM_GRIDS``. 

**Important** check performance caveats (including maximal supported values for ``NUM_GRIDS`` depending on GPU type and ``tile_size``) and preconditioning advice in the documentation. 


