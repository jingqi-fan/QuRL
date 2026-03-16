import os
import numpy as np

current_dir = os.getcwd()
fname = 'reentrant_2_delta2.npy'
path = os.path.join(current_dir, fname)
data = np.load(path, allow_pickle=True)
print(f"📁 File name: {fname}")
print(f"   ▶ Type: {type(data)}")
if isinstance(data, np.ndarray):
    print(f"   ▶ Shape: {data.shape}")
    print(f"   ▶ Data type: {data.dtype}")
print(f"   ▶ Content preview:\n{data}\n{'-'*60}\n")

# [6.42857143e-02 1.00000000e-06 6.42857143e-02 1.00000000e-06
 # 1.00000000e-06 1.00000000e-06]

# [[0.125      0.5        0.25       0.         0.         0.        ]
 # [0.         0.         0.         0.16666667 0.14285715 1.        ]]