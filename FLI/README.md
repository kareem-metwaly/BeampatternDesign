# Unrolled PDR Algorithm


This is an implementation of the unrolling of PDR algorithm.


## contents:
1. [utils](utils): contains some utilities that are used throughout the code.
Most importantly, it contains [base_classes.py](utils/base_classes.py) where it has all the configuration classes.

2. [configs](configs): contains all examples of configurations used for datasets + training models + testing.

3. [dataset.py](datasets/base_dataset.py): contains the dataset module to load data that was saved from Matlab and prepares it for PyTorch model.


## Progress so far:
Working on implementation of the basic blocks; probably will borrow from GlideNet
