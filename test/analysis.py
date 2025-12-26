import numpy as np

def compare_npy(path_a, path_b):
    arr_a = np.load(path_a)
    arr_b = np.load(path_b)

    # 比较形状
    if arr_a.shape != arr_b.shape:
        print("❌ 两个文件 shape 不同!")
        print(f"shape A = {arr_a.shape}, shape B = {arr_b.shape}")
        return

    # 比较内容
    if np.array_equal(arr_a, arr_b):
        print("✅ 两个 npy 文件内容完全一致!")
    else:
        diff = np.abs(arr_a - arr_b)
        print("⚠️ 两个文件内容不完全相同!")
        print(f"最大差值: {diff.max()}")
        print(f"平均差值: {diff.mean()}")

if __name__ == "__main__":
    # 修改为你的路径
    path_a = "D:/z/analysis/action_history_cmu_50_linear.npy"
    path_b = "D:/z/analysis/action_history_cmuq_50_linear.npy"

    compare_npy(path_a, path_b)
