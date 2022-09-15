---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region pycharm={"name": "#%% md\n"} -->
### 1. Импортировать библиотеки в Python.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
import numpy as np
import matplotlib.pyplot as plt
import os
import random
```

<!-- #region pycharm={"name": "#%% md\n"} -->
### 2. Загрузка и подготовка данных.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
name = random.choice(os.listdir("dataset"))
print(f"Dataset: {name}")

dataset = np.genfromtxt(f"dataset/{name}", delimiter=',')

dataset = [dataset[:, i] for i in range(dataset.shape[1])]
title = ["time", "current", "voltage"]

dataset_dict = dict(zip(title, dataset))
```

<!-- #region pycharm={"name": "#%% md\n"} -->
### 3. Нарисовать графики тока и напряжения.

Для удобства отображения отображу не весь график, а некоторый его случайный диапазон заданного размера, установив лимиты на данные.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
"""
Размер интервала
"""
time_period = 0.5
```

```python pycharm={"name": "#%%\n"}
time_interval = random.random() * (dataset_dict["time"][-1] - time_period)
time_interval = (time_interval, time_interval + time_period)

print(f"Временной интервал {time_interval}")
```

```python pycharm={"name": "#%%\n"}
plt.plot(dataset_dict["time"], dataset_dict["current"])
plt.xlim(time_interval)
plt.grid()
plt.xlabel('Время, с')
plt.ylabel('Сила Тока, А')
```

```python pycharm={"name": "#%%\n"}
plt.plot(dataset_dict["time"], dataset_dict["voltage"])
plt.xlim(time_interval)
plt.grid()
plt.xlabel('Время, с')
plt.ylabel('Напряжение, В')
```

<!-- #region pycharm={"name": "#%% md\n"} -->
### 4. Рассчитать значения параметров L и R.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}

```

```python pycharm={"name": "#%%\n"}

```

```python pycharm={"name": "#%%\n"}

```

```python pycharm={"name": "#%%\n"}

```
