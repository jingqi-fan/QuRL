import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ==== 字体风格（与你主图一致：serif） ====
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['mathtext.fontset'] = 'dejavusans'
plt.rcParams['legend.fontsize'] = 19

# ==== 颜色定义 ====
colors = ['#74ca26', '#bae99b', '#e4ffd8', '#f4d9d9']
oom_color = '#bdbdbd'

legend_labels = [
    'GPU + multi-batched',
    'GPU + multi-env',
    'CPU + multi-batched',
    'CPU + multi-env'
]

# ==== 构造 handles ====
handles = []

# GPU + multi-batched（斜线）
handles.append(
    Patch(
        facecolor=colors[0],
        edgecolor='#4d4d4d',
        hatch='//',
        linewidth=0,
        label=legend_labels[0]
    )
)

# 其余三种
for i in range(1, 4):
    handles.append(
        Patch(
            facecolor=colors[i],
            edgecolor='white',
            linewidth=0.5,
            label=legend_labels[i]
        )
    )

# OOM
handles.append(
    Patch(
        facecolor=oom_color,
        edgecolor='white',
        linewidth=0.5,
        label='out of memory'
    )
)

# ==== legend-only 图 ====
fig = plt.figure(figsize=(10, 1.35))
legend = fig.legend(
    handles=handles,
    loc='center',
    ncol=5,
    frameon=True,          # 保留白底
    fancybox=False,        # 不要圆角（更干净）
    framealpha=1.0,
    borderpad=0.3,         # ⭐ 压缩内部留白
    handlelength=2.2,
    handleheight=1.0,
    columnspacing=1.3,     # ⭐ 压缩列间距
    handletextpad=0.6,     # ⭐ 图例与文字距离
    fontsize=19
)

# === 去掉边框 ===
frame = legend.get_frame()
frame.set_edgecolor('none')
frame.set_linewidth(0.0)
frame.set_facecolor('white')

plt.axis('off')

# ==== 保存（极限裁剪） ====
plt.savefig('legend_only.pdf', bbox_inches='tight', pad_inches=0.02)
plt.savefig('legend_only.png', bbox_inches='tight', pad_inches=0.02, dpi=300)
plt.show()
