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

    (60000, 785)
    (10000, 785)
    


```python
mnist_train[:,0]
```




    array([5., 0., 4., ..., 5., 6., 8.])


