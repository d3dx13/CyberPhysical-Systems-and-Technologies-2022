Лабораторная работа №2. Введение в проектирование нейронных сетей с помощью Python


```python
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import scipy
import torch

torch.cuda.synchronize()
torch.cuda.empty_cache()

cuda = torch.device('cuda')
print(torch.cuda.get_device_properties(cuda))
```

    _CudaDeviceProperties(name='NVIDIA GeForce RTX 3080 Laptop GPU', major=8, minor=6, total_memory=8191MB, multi_processor_count=48)
    


```python
mnist_train = np.genfromtxt(f"dataset/mnist_train.csv", delimiter=',')
mnist_test = np.genfromtxt(f"dataset/mnist_test.csv", delimiter=',')
```


```python
print(mnist_train.shape)
print(mnist_test.shape)
```

    (3,)
    (3,)
    


```python
mnist_train[:,0]
```


    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    Cell In [24], line 1
    ----> 1 mnist_train[:,0]
    

    IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed

