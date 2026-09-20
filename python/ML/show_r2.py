import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ===== 全局字体 =====
plt.rcParams['font.family'] = 'Arial'   # 最稳，不要再死磕 Arial
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 16

# ===== 颜色映射 =====
cmap = LinearSegmentedColormap.from_list('custom', ['#F2F2F2', '#EF949E'])

# ===== 读取数据 =====
df = pd.read_excel("RMSE结果.xlsx")
df = df.set_index(df.columns[0])
df = df.replace('`', '', regex=True)
df = df.apply(pd.to_numeric, errors='coerce')
df = df.dropna(how='all')
df = df.dropna(axis=1, how='all')

# ===== 创建画布 =====
fig, ax = plt.subplots(figsize=(10, 8))

# ===== 热力图 =====
sns.heatmap(
    df,
    annot=True,
    cmap=cmap,
    fmt=".3f",
    linewidths=3.5,
    square=True,
    annot_kws={"size": 22, "fontweight": "bold", "fontfamily": "Arial"},
    cbar_kws={"shrink": 0.9},
    ax=ax
)

# ===== 坐标轴标签 =====
ax.set_xlabel("", fontsize=20, fontweight='normal', fontname='Arial')
ax.set_ylabel("", fontsize=20, fontweight='normal', fontname='Arial')

# ===== 关键：全部用普通文本，不用 mathtext =====
xtick_labels = []
for col in df.columns:
    col_str = str(col).strip()
    if col_str == "pLC50":
        xtick_labels.append("pLC$_{50}$")   # Unicode 下标，彻底避免数学字体混入
    else:
        xtick_labels.append(col_str)

ax.set_xticklabels(xtick_labels, rotation=45, ha='right')
ax.set_yticklabels([t.get_text() for t in ax.get_yticklabels()], rotation=0)

# ===== 强制统一刻度字体 =====
for label in ax.get_xticklabels():
    label.set_fontname("Arial")
    label.set_fontsize(28)
    label.set_fontweight("bold")

for label in ax.get_yticklabels():
    label.set_fontname("Arial")
    label.set_fontsize(28)
    label.set_fontweight("bold")

# ===== colorbar 字体 =====
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=24)
for label in cbar.ax.get_yticklabels():
    label.set_fontname("Arial")

plt.tight_layout()
plt.savefig("RMSE红色.png", dpi=600, bbox_inches='tight')
plt.show()