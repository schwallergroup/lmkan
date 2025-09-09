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

This command will trigger in-place compilation of the CUDA kernels, so you will need the ``nvcc`` compiler (CUDA toolkit). 

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

Backward pass is also included.

The number of trainable parameters is :math:`(\text{NUM\_GRIDS}+1)^2 \times \text{OUTPUT\_DIM} \times \left\lfloor \text{INPUT\_DIM}/2 \right\rfloor`. Inference FLOPs are twice that of a linear layer of the same shape, independent of ``NUM_GRIDS``. So you get about :math:`(30+1)^2 / 2 \approx 480`× more trainable parameters.

.. important::
   Check performance caveats (including maximal supported values for ``NUM_GRIDS`` depending on GPU type and ``tile_size_*``) and preconditioning advice in the documentation.

