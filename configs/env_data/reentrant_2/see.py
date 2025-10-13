import os
import numpy as np

current_dir = os.getcwd()
fname = 'reentrant_2_delta.npy'
path = os.path.join(current_dir, fname)
data = np.load(path, allow_pickle=True)
print(f"📁 文件名: {fname}")
print(f"   ▶ 类型: {type(data)}")
if isinstance(data, np.ndarray):
    print(f"   ▶ 形状: {data.shape}")
    print(f"   ▶ 数据类型: {data.dtype}")
print(f"   ▶ 内容预览:\n{data}\n{'-'*60}\n")
