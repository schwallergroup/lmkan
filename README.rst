.. role:: bash(code)
   :language: bash


Lookup multivariate Kolmogorov-Arnold Networks
==============================================

A general-purpose drop-in replacement for high-dimensional linear mappings (aka linear layers) with better capacity -- inference cost ratio. 

.. image:: /figures/performance.svg
   :alt: Performance of lmKANs
   :align: center


+++++++++++++
Installation
+++++++++++++
Make sure you have numpy and CUDA-enabled PyTorch installed.

Run :bash:`pip install .`

This command will trigger in-place compilation of the CUDA kernels, so you will need the ``nvcc`` compiler (CUDA toolkit). 

