import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# ==== 字体设置 ====
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['axes.labelsize'] = 26
plt.rcParams['legend.fontsize'] = 19
plt.rcParams['xtick.labelsize'] = 19
plt.rcParams['ytick.labelsize'] = 19

# 数据（负数表示 OOM，柱高取 abs(负数)）
data = np.array([
    [48.59, 47.53, 48.70, 48.78],
    [411.77, 413.05, 416.31, 462.27],
    [267.99, 422.23, 650.3, 1643.83],
    [78.76, 170.04, 396.22, -100]   # OOM -> 灰柱高度 200
])

x_labels = ['10', '50', '100', '200']
x = np.arange(len(x_labels))
width = 0.18

colors = ['#9ed969', '#bae99b', '#e4ffd8', '#f4d9d9']
oom_color = '#bdbdbd'

legend_labels = [
    'GPU + multi-batched',
    'GPU + multi-env',
    'CPU + multi-batched',
    'CPU + multi-env'
]

plt.figure(figsize=(10, 6))
bars = []

# 画柱子
for i in range(data.shape[0]):
    bar_group = []
    for j, value in enumerate(data[i]):
        is_oom = (value < 0)
        height = abs(value) if is_oom else value

        color = oom_color if is_oom else colors[i]
        hatch = '//' if i == 0 and (not is_oom) else None
        edge_color = '#4d4d4d' if hatch else 'white'
        line_wi = 0 if i ==0 else 0.5

        b = plt.bar(
            x[j] + i * width,
            height,
            width,
            color=color,
            edgecolor=edge_color,
            linewidth=0.5,
            hatch=hatch
        )
        bar_group.append(b[0])
    bars.append(bar_group)

# 标数值（跳过 OOM：原始值 < 0）
for i in range(data.shape[0]):
    for j, rect in enumerate(bars[i]):
        if data[i, j] < 0:
            continue
        height = rect.get_height()
        plt.text(
            rect.get_x() + rect.get_width() / 2,
            height,
            f'{height:.1f}',
            ha='center',
            va='bottom',
            fontsize=9
        )

plt.xticks(x + width * 1.5, x_labels)
plt.xlabel('queues & servers')
plt.ylabel('Running Time')

# legend（加入 OOM）
handles = []
# GPU + multi-batched（带深灰斜线）
handles.append(
    Patch(
        facecolor=colors[0],
        edgecolor='#4d4d4d',  # 斜线颜色
        hatch='//',
        label=legend_labels[0]
    )
)
# 其余正常柱子（无斜线，白边）
for i in range(1, 4):
    handles.append(
        Patch(
            facecolor=colors[i],
            edgecolor='white',
            label=legend_labels[i]
        )
    )
# OOM
handles.append(
    Patch(
        facecolor=oom_color,
        edgecolor='white',
        label='out of memory'
    )
)
plt.legend(handles=handles, loc='upper left')


plt.tight_layout()
plt.savefig('aba.png', dpi=300)
plt.savefig('aba.pdf')
plt.show()
