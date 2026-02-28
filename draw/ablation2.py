import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

# ==== 字体设置 ====
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['mathtext.fontset'] = 'dejavusans'
plt.rcParams['axes.labelsize'] = 26
plt.rcParams['legend.fontsize'] = 19
plt.rcParams['xtick.labelsize'] = 19
plt.rcParams['ytick.labelsize'] = 19

# 防止生成 type3 字体
plt.rcParams['pdf.fonttype'] = 42   # TrueType
plt.rcParams['ps.fonttype']  = 42

# 数据（负数表示 OOM，柱高取 abs(负数)）
data = np.array([
    [210.26, 211.38, 210.48, 498.46],
    [7938.35, 8120.18, 7745.60, 7980.42],
    [363.61, 509.96, 793.38, 2505.93],
    [5866.62, 7031.59, 8299.34, 12145.95]   # OOM -> 灰柱高度 200
])

x_labels = ['10', '50', '100', '200']
x = np.arange(len(x_labels))
width = 0.18
#95de54
#9ed969
#7fce37
#66b221
#74ca26
# colors = ['#e47d6b', '#e9d264', '#8ED1C6', '#6B8EC1']
# oom_color = '#6B8EC1'

colors = ['#eb9184', '#fcf2ca', '#afe4dd', '#e2f6f3']
oom_color = '#e2f6f3'
# colors = ['#eb9184', '#fcf2ca', '#afe4dd', '#9ebbf1']
# oom_color = '#9ebbf1'
#e2f6f3
legend_labels = [
    'GPU + multi-batched',
    'GPU + multi-env',
    'CPU + multi-batched',
    'CPU + multi-env'
]

plt.figure(figsize=(10, 6))

bars = []
# 画柱子（支持：OOM 负值灰色 + i=0 斜线 + i=0 linewidth=0.0）
for i in range(data.shape[0]):
    bar_group = []

    for j, value in enumerate(data[i]):
        # OOM: 用负值标记
        is_oom = (value < 0)

        # 高度：OOM 取绝对值画出来（也可以改成 0，看你想表达什么）
        height = abs(value) if is_oom else value

        # 颜色：OOM 灰色，其它正常色
        color = oom_color if is_oom else colors[i]

        # 斜线：仅 i==0 且非 OOM
        hatch = '//' if (i == 0 and not is_oom) else None

        # 边框颜色：有斜线时用深灰线色，否则白色
        edge_color = '#4d4d4d' if hatch else 'white'

        # linewidth：i==0 为 0.0，其它为 0.5
        lw = 0.0 if i == 0 else 0.5

        b = plt.bar(
            x[j] + i * width,
            height,
            width,
            color=color,
            edgecolor=edge_color,
            linewidth=lw,
            hatch=hatch,
            label=legend_labels[i] if j == 0 else None  # 避免 legend 重复
        )

        bar_group.append(b[0])

    bars.append(bar_group)


# 标数值（跳过 OOM：原始值 < 0）
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
            fontsize=10,
            fontweight='bold' if i == 0 else 'normal'   # 仅 GPU + multi-batched 加粗
        )


plt.xticks(x + width * 1.5, x_labels)
ax = plt.gca()
ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
plt.xlabel('queues & servers')
plt.ylabel('Running Time (s)')

# legend（加入 OOM）
handles = []
# GPU + multi-batched（带深灰斜线）
handles.append(
    Patch(
        facecolor=colors[0],
        edgecolor='#4d4d4d',  # 斜线颜色
        hatch='//',
        linewidth=0,
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
# plt.legend(handles=handles, loc='upper left')


plt.tight_layout()
plt.savefig('aba2.png')
plt.savefig('aba2.pdf')
plt.show()
