#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np

df = pd.read_csv('KEGG桑基图.csv')

rows = []
for idx, row in df.iterrows():
    genes = row['geneID'].split('/')
    for gene in genes:
        rows.append({
            'pathway': row['Description'],
            'gene': gene.strip(),
            'ratio': float(row['GeneRatio']),
            'pvalue': float(row['pvalue']),
            'neg_log10_p': -np.log10(float(row['pvalue'])),
            'count': int(row['count'])
        })

data_long = pd.DataFrame(rows)
pathways = list(df['Description'])
genes_all = []
for p in pathways:
    gs = data_long[data_long['pathway'] == p]['gene'].unique()
    for g in gs:
        if g not in genes_all:
            genes_all.append(g)
genes = genes_all

# pvalue相关
pathway_neg_log10_p = {}
for p in pathways:
    pathway_neg_log10_p[p] = data_long[data_long['pathway'] == p]['neg_log10_p'].mean()

gene_neg_log10_p = {}
for gene in genes:
    gene_neg_log10_p[gene] = data_long[data_long['gene'] == gene]['neg_log10_p'].mean()

min_nlp = data_long['neg_log10_p'].min()
max_nlp = data_long['neg_log10_p'].max()

# ====== 颜色映射：#F2F2F2 -> #EF949E，基于pvalue ======
cmap = LinearSegmentedColormap.from_list('custom', ['#F2F2F2', '#EF949E'], N=256)


def get_color(neg_log10_p):
    normalized = (neg_log10_p - min_nlp) / (max_nlp - min_nlp)
    normalized = max(0, min(1, normalized))
    c = cmap(normalized)
    return (c[0], c[1], c[2])


# 通路颜色
pathway_colors = {}
pathway_colors_light = {}
for p in pathways:
    c = get_color(pathway_neg_log10_p[p])
    pathway_colors[p] = c
    pathway_colors_light[p] = (c[0], c[1], c[2], 0.45)

# ====== 布局 ======
fig_width = 8
fig_height = 6
fig, ax = plt.subplots(figsize=(fig_width, fig_height))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

top_y = 7.5
bottom_y = 3.5
pathway_block_height = 0.30
gene_block_height = 0.30

total_connections = sum(len(data_long[data_long['pathway'] == p]['gene'].unique()) for p in pathways)
pathway_gap = 0.25
total_available_width = 8.0
x_start = 1.0

pathway_block_widths = {}
pathway_x_positions = {}
current_x = x_start
for p in pathways:
    n_conn = len(data_long[data_long['pathway'] == p]['gene'].unique())
    w = max(0.9, n_conn / total_connections * total_available_width)
    pathway_block_widths[p] = w
    pathway_x_positions[p] = current_x
    current_x += w + pathway_gap

total_pathway_width = current_x - pathway_gap - x_start

n_genes = len(genes)
gene_block_width = 0.32
gene_gap = 0.08
total_gene_width = n_genes * gene_block_width + (n_genes - 1) * gene_gap
pathway_center_x = x_start + total_pathway_width / 2
gene_start_x = pathway_center_x - total_gene_width / 2

gene_x_positions = {}
for i, gene in enumerate(genes):
    gene_x_positions[gene] = gene_start_x + i * (gene_block_width + gene_gap)

# ====== 丝带连接线 ======
for pathway in pathways:
    pathway_genes = data_long[data_long['pathway'] == pathway]['gene'].unique()
    pathway_genes_sorted = sorted(pathway_genes, key=lambda g: gene_x_positions[g])
    n_conn = len(pathway_genes_sorted)
    color = pathway_colors_light[pathway]
    p_left = pathway_x_positions[pathway]
    p_width = pathway_block_widths[pathway]

    ribbon_height = p_width / (n_conn + 1) * 0.60
    ribbon_height = min(ribbon_height, 0.18)

    for i, gene in enumerate(pathway_genes_sorted):
        x1 = p_left + (i + 1) * (p_width / (n_conn + 1))
        y1 = top_y - pathway_block_height / 2
        x2 = gene_x_positions[gene] + gene_block_width / 2
        y2 = bottom_y + gene_block_height / 2

        ctrl_y1 = y1 - (y1 - y2) * 0.35
        ctrl_y2 = y1 - (y1 - y2) * 0.65
        hw = ribbon_height / 2
        gene_hw = gene_block_width / 2 * 0.50

        verts = [
            (x1 - hw, y1), (x1 - hw, ctrl_y1), (x2 - gene_hw, ctrl_y2), (x2 - gene_hw, y2),
            (x2 + gene_hw, y2), (x2 + gene_hw, ctrl_y2), (x1 + hw, ctrl_y1), (x1 + hw, y1),
            (x1 - hw, y1),
        ]
        codes = [Path.MOVETO] + [Path.CURVE4] * 3 + [Path.LINETO] + [Path.CURVE4] * 3 + [Path.CLOSEPOLY]
        path = Path(verts, codes)
        patch = patches.PathPatch(path, facecolor=color[:3], alpha=color[3], edgecolor='none', linewidth=0)
        ax.add_patch(patch)

# ====== 通路色块 ======
for pathway in pathways:
    color = pathway_colors[pathway]
    x = pathway_x_positions[pathway]
    w = pathway_block_widths[pathway]
    rect = patches.FancyBboxPatch(
        (x, top_y - pathway_block_height / 2), w, pathway_block_height,
        boxstyle="round,pad=0.04", facecolor=color, edgecolor='none', alpha=0.95)
    ax.add_patch(rect)

# ====== 基因色块 ======
for gene in genes:
    x = gene_x_positions[gene]
    color = get_color(gene_neg_log10_p[gene])
    rect = patches.FancyBboxPatch(
        (x, bottom_y - gene_block_height / 2), gene_block_width, gene_block_height,
        boxstyle="round,pad=0.03", facecolor=color, edgecolor='none', alpha=0.95)
    ax.add_patch(rect)

# ====== 基因标签 ======
for gene in genes:
    x = gene_x_positions[gene]
    ax.text(x + gene_block_width / 2, bottom_y - gene_block_height / 2 - 0.12,
            gene, ha='center', va='top', fontsize=10, fontweight='bold', color='black', rotation=90)

# ====== 颜色条：pvalue ======
ax_cbar = fig.add_axes([0.90, 0.35, 0.018, 0.30])
norm = Normalize(vmin=int(min_nlp), vmax=int(max_nlp) + 1)
cb = ColorbarBase(ax_cbar, cmap=cmap, norm=norm, orientation='vertical')
cb.set_label('-log10(pvalue)', fontsize=10, fontweight='bold', rotation=270, labelpad=15)
cb.ax.tick_params(labelsize=8)

ax.set_xlim(0, 10.5)
ax.set_ylim(1.0, 9.5)
ax.axis('off')

# --- 版本1：带通路名 ---
pathway_texts = []
for pathway in pathways:
    x = pathway_x_positions[pathway]
    w = pathway_block_widths[pathway]
    t = ax.text(x + w / 2, top_y + pathway_block_height / 2 + 0.12,
                pathway, ha='center', va='bottom',
                fontsize=7, fontweight='bold', color='black', rotation=90)
    pathway_texts.append(t)

plt.savefig('kegg_sankey_with_labels.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('kegg_sankey_with_labels.pdf', bbox_inches='tight', facecolor='white')
print("Saved with labels")

# --- 版本2：不带通路名 ---
for t in pathway_texts:
    t.set_visible(False)
plt.savefig('kegg_sankey_no_labels.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('kegg_sankey_no_labels.pdf', bbox_inches='tight', facecolor='white')
print("Saved no labels")
print("Done!")