import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# ===== 1. 读取三类数据 =====
classes_info = []
for name in ['DBPs_Chlorine-containing', 'DBPs_Bromine-containing', 'DBPs_Others']:
    df = pd.read_csv(f'{name}.csv')
    df = df.set_index(df.columns[0])
    min_val = df.min().min()
    classes_info.append((name, df, min_val))

# ===== 2. 按最低结合能排序（最低的类在上面） =====
classes_info.sort(key=lambda x: x[2])

for name, df, mv in classes_info:
    print(f"{name}: min={mv}, shape={df.shape}")

# ===== 3. 类内小分子（行）按 min 排序，拼接 =====
all_data = []
boundaries = []
cum = 0

for name, df, mv in classes_info:
    # ★ 类内小分子按最低结合能排序，最低在前 ★
    row_order = df.min(axis=1).sort_values().index
    df = df.loc[row_order]

    all_data.append(df)
    cum += len(df)
    boundaries.append(cum)

combined = pd.concat(all_data)

# ★ 蛋白（列）按全局最低结合能排序，最低在前 ★
col_order = combined.min(axis=0).sort_values().index
combined = combined[col_order]

print(f"\nCombined shape: {combined.shape}")
print(f"Boundaries: {boundaries}")
print(f"Column order: {list(col_order)}")

# ===== 4. 画图 =====
colors = ["#2b8cbe", "white", "#e41a1c"]
cmap = LinearSegmentedColormap.from_list("custom", colors)
norm = TwoSlopeNorm(vmin=-10, vcenter=-5, vmax=0)

n_rows, n_cols = combined.shape
cell_w = 0.55
cell_h = 0.22

fig_width = n_cols * cell_w + 3.0
fig_height = n_rows * cell_h + 4.0

fig, ax = plt.subplots(figsize=(fig_width, fig_height))

sns.heatmap(
    combined,
    cmap=cmap,
    norm=norm,
    linewidths=0.3,
    linecolor="white",
    annot=True,
    fmt=".2f",
    annot_kws={"size": 5},
    ax=ax,
    cbar_kws={"label": "Binding Energy (kcal/mol)", "shrink": 0.5},
)

# ===== 5. 画分类分界线 =====
for b in boundaries[:-1]:
    ax.axhline(y=b, color='black', linewidth=2.5)

# ===== 6. 添加分类标签（右侧） =====
class_names = [info[0].replace('DBPs_', '') for info in classes_info]
prev = 0
for i, b in enumerate(boundaries):
    mid = (prev + b) / 2
    ax.text(
        n_cols + 0.5, mid,
        class_names[i],
        ha='left', va='center',
        fontsize=11, fontweight='bold',
        rotation=-90
    )
    prev = b

# ===== 7. 轴标签样式 =====
ax.set_xlabel("Proteins", fontsize=13, fontweight='bold')
ax.set_ylabel("Ligands", fontsize=13, fontweight='bold')

ax.tick_params(axis="x", rotation=60, labelsize=9)
ax.tick_params(axis="y", rotation=0, labelsize=7)
for label in ax.get_xticklabels():
    label.set_ha("right")
    label.set_fontweight("bold")
for label in ax.get_yticklabels():
    label.set_fontweight("bold")

cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=11)
for label in cbar.ax.get_yticklabels():
    label.set_fontweight('bold')
cbar.ax.set_ylabel("Binding Energy (kcal/mol)", fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig("DBPs_Combined.png", dpi=600, bbox_inches="tight")
plt.close()
print("Done -> DBPs_Combined.png")