import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy import stats
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'figure.facecolor': 'white',
})

df = pd.read_csv('DBPs_x.csv')

properties = ['pLC50', 'BCF', 'LogP', 'MW', 'TPSA']
labels = ['pLC50', 'BCF', 'LogP', 'MW (g/mol)', 'TPSA (Å²)']


# 基础 colormap
base = plt.get_cmap('RdBu_r')

# 蓝侧
blue_side = base(np.linspace(0.10, 0.40, 128))

# 中间浅蓝过渡
mid = base(np.linspace(0.30, 0.40, 20))

# 红侧
red_side = base(np.linspace(0.55, 0.85, 128))

# 拼接
colors_list = np.vstack([
    blue_side,
    mid,
    red_side
])

# 新 colormap
new_cmap = LinearSegmentedColormap.from_list(
    "RdBu_blue_mid",
    colors_list
)

# 取颜色
colors = [
    new_cmap(i)
    for i in np.linspace(0, 1, len(properties))
]

print("生成颜色：")
for p, c in zip(properties, colors):
    print(f"{p}: {c}")

fig = plt.figure(figsize=(22, 5.5), dpi=600)
outer = fig.add_gridspec(1, 5, wspace=0.45, left=0.04, right=0.98, top=0.92, bottom=0.12)

for i, (prop, label) in enumerate(zip(properties, labels)):
    data = df[prop].dropna().values
    color = colors[i]

    inner = outer[i].subgridspec(1, 2, width_ratios=[1, 0.4], wspace=0.05)
    ax_box = fig.add_subplot(inner[0])
    ax_hist = fig.add_subplot(inner[1], sharey=ax_box)

    # 箱线图
    bp = ax_box.boxplot(
        [data], positions=[1], widths=0.5,
        patch_artist=True, showfliers=False,
        boxprops=dict(linewidth=1.5, facecolor=color, alpha=0.3, edgecolor=color),
        medianprops=dict(linewidth=2.5, color='black', solid_capstyle='round'),
        whiskerprops=dict(linewidth=1.5, color=color, linestyle='-'),
        capprops=dict(linewidth=1.5, color=color),
    )

    # 散点
    np.random.seed(42)
    jitter = np.random.normal(1, 0.06, len(data))
    ax_box.scatter(jitter, data, alpha=0.6, s=28, color=color, edgecolors='none', zorder=3)

    ax_box.set_xlim(0.4, 1.6)
    ax_box.set_xticks([1])
    ax_box.set_xticklabels([label], fontsize=24, fontweight='bold', color='#333333')
    ax_box.tick_params(axis='x', width=2.2, length=6, color='black')
    ax_box.tick_params(axis='y', labelsize=24, labelcolor='black', color='black', width=1.8)
    for tick in ax_box.get_yticklabels():
        tick.set_fontweight('bold')
    if i == 0:
        ax_box.set_ylabel('Value', fontsize=24, fontweight='bold', color='#333333')

    ax_box.grid(axis='y', alpha=0.25, linestyle='--', linewidth=0.5, color='#999999')
    ax_box.set_axisbelow(True)

    ax_box.spines['top'].set_visible(False)
    ax_box.spines['right'].set_visible(False)
    ax_box.spines['left'].set_color('#BBBBBB')
    ax_box.spines['bottom'].set_color('#BBBBBB')

    # 直方图
    y_min, y_max = ax_box.get_ylim()
    bins = np.linspace(data.min(), data.max(), 30)
    ax_hist.hist(data, bins=bins, orientation='horizontal',
                 color=color, alpha=0.65, edgecolor='none', linewidth=0)

    try:
        kde = stats.gaussian_kde(data)
        y_kde = np.linspace(data.min(), data.max(), 200)
        x_kde = kde(y_kde)
        hist_vals, _ = np.histogram(data, bins=bins)
        scale = hist_vals.max() / x_kde.max() if x_kde.max() > 0 else 1
        ax_hist.plot(x_kde * scale, y_kde, color=color, linewidth=1.8, alpha=0.9)
    except Exception:
        pass

    ax_hist.set_ylim(y_min, y_max)
    ax_hist.tick_params(axis='x', labelsize=20, labelcolor='black', color='black', width=2.2, length=6)
    for tick in ax_hist.get_xticklabels():
        tick.set_fontweight('bold')
    plt.setp(ax_hist.get_yticklabels(), visible=False)
    ax_hist.tick_params(axis='y', length=0)

    ax_hist.spines['top'].set_visible(False)
    ax_hist.spines['right'].set_visible(False)
    ax_hist.spines['bottom'].set_color('black')
    ax_hist.spines['bottom'].set_linewidth(1.5)

    ax_box.spines['left'].set_color('black')
    ax_box.spines['bottom'].set_color('black')
    ax_box.spines['left'].set_linewidth(1.5)
    ax_box.spines['bottom'].set_linewidth(1.5)

    ax_hist.grid(axis='x', linestyle='--', linewidth=0.4, alpha=0.25, color='#999999')
    ax_hist.set_axisbelow(True)

fig.suptitle('Distribution of Molecular Properties', fontsize=17, fontweight='bold', color='#222222', y=0.98)
plt.savefig('molecular_properties_final.png', dpi=600, bbox_inches='tight', facecolor='white')
print("Done! Saved as 'molecular_properties_final.png'")