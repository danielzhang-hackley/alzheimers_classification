import numpy as np

x = np.array([2, 3, 5, 15])

print(
    np.where(
        np.logical_and(x < 3, np.array([True, True, True, True]))
    )
)

print(np.where(np.logical_and(x, np.array([True, True, False, True])), x, 0))