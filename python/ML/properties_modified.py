import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from numpy.polynomial import polynomial as P
import matplotlib.font_manager as fm

# ======================
# Arial 检查
# ======================
arial_fonts = [
    f.fname
    for f in fm.fontManager.ttflist
    if f.name == "Arial"
]

if arial_fonts:
    print("已找到 Arial:")
    for x in arial_fonts:
        print(x)
else:
    print("未找到 Arial")

# ======================
# 全局字体
# ======================
plt.rcParams['font.family'] = 'Arial'

plt.rcParams['font.sans-serif'] = [
    'Arial',
    'DejaVu Sans',
    'Liberation Sans',
    'sans-serif'
]

plt.rcParams[
    'axes.unicode_minus'
] = False

plt.rcParams[
    'font.weight'
] = 'bold'

plt.rcParams[
    'axes.labelweight'
] = 'bold'

# ======================
# 数据
# ======================
df = pd.read_csv(
    'DBPs_classified.csv'
)

class_col = (
    'Classification'
)

categories = sorted(
    df[
        class_col
    ]
    .dropna()
    .unique(),
    key=str
)

n_cat = len(
    categories
)

print(categories)

print(
    df[class_col]
    .value_counts()
    .sort_index()
)

# ======================
# 配色
# ======================
cat_colors = [

    "#2E73B8",   # 深蓝

    "#BCD8E8",   # 浅蓝

    "#D65F5F"    # 红
]

LOGP_YTOP_OFFSET = 2.0


def plot_property(
        ax,
        prop,
        panel_label=None):

    ylabel_text = (
        f'{prop} Distribution'
    )

    positions = list(
        range(
            1,
            n_cat + 1
        )
    )

    group_data = []

    group_means = []

    for i, cat in enumerate(
            categories):

        subset = (

            df[
                df[
                    class_col
                ] == cat
            ][prop]

            .dropna()

            .values
        )

        group_data.append(
            subset
        )

        group_means.append(

            np.mean(
                subset
            )

            if len(
                subset
            ) > 0

            else np.nan
        )

        color = cat_colors[
            i % len(
                cat_colors
            )
        ]

        # ======================
        # 箱线图
        # ======================
        ax.boxplot(

            [subset],

            positions=[
                positions[i]
            ],

            widths=0.45,

            patch_artist=True,

            showfliers=False,

            boxprops=dict(

                linewidth=0,

                facecolor=color,

                alpha=0.45,

                edgecolor='none'
            ),

            medianprops=dict(

                linewidth=2,

                color='black'
            ),

            whiskerprops=dict(

                linewidth=1.6,

                color=color
            ),

            capprops=dict(

                linewidth=1.6,

                color=color
            )
        )

        # ======================
        # 散点
        # ======================
        np.random.seed(
            42 + i
        )

        jitter = np.random.normal(

            0,

            0.08,

            size=len(
                subset
            )
        )

        ax.scatter(

            positions[i]
            + jitter,

            subset,

            s=35,

            alpha=0.75,

            color=color,

            edgecolors='none',

            linewidths=0,

            zorder=3
        )

    # ======================
    # 趋势线
    # ======================
    valid = [

        (p, m)

        for p, m in zip(
            positions,
            group_means
        )

        if not np.isnan(
            m
        )
    ]

    if len(valid) >= 2:

        xv, yv = zip(
            *valid
        )

        xv = np.array(
            xv
        )

        yv = np.array(
            yv
        )

        deg = min(
            2,
            len(
                xv
            ) - 1
        )

        coeffs = P.polyfit(

            xv,

            yv,

            deg
        )

        x_smooth = np.linspace(

            min(xv)-0.4,

            max(xv)+0.4,

            200
        )

        y_smooth = P.polyval(
            x_smooth,
            coeffs
        )

        ax.plot(

            x_smooth,

            y_smooth,

            color='#888888',

            linewidth=3,

            zorder=1
        )

    # ======================
    # LogP 调整
    # ======================
    if prop == 'LogP':

        all_vals = np.concatenate(

            [
                x

                for x in group_data

                if len(x) > 0
            ]

        )

        ax.set_ylim(

            bottom=np.min(
                all_vals
            ) - 0.5,

            top=np.max(
                all_vals
            )

            + LOGP_YTOP_OFFSET
        )

    # ======================
    # 显著性
    # ======================
    for i in range(
            n_cat - 1):

        d1 = group_data[i]

        d2 = group_data[
            i + 1
        ]

        if (
                len(d1) >= 3
                and
                len(d2) >= 3
        ):

            _, p_val = (
                stats.ttest_ind(

                    d1,

                    d2,

                    equal_var=False
                )
            )

            if p_val < 0.001:

                sig = "***"

            elif p_val < 0.01:

                sig = "**"

            elif p_val < 0.05:

                sig = "*"

            else:

                sig = None

            if sig:

                y_range = (

                    ax.get_ylim()[1]

                    -

                    ax.get_ylim()[0]
                )

                bar_y = (

                    ax.get_ylim()[1]

                    -

                    0.12
                    *
                    y_range

                    +

                    i
                    *
                    0.04
                    *
                    y_range
                )

                x1 = positions[i]

                x2 = positions[
                    i + 1
                ]

                ax.plot(

                    [
                        x1,
                        x1,
                        x2,
                        x2
                    ],

                    [
                        bar_y,

                        bar_y
                        +
                        0.02
                        *
                        y_range,

                        bar_y
                        +
                        0.02
                        *
                        y_range,

                        bar_y
                    ],

                    lw=1.2,

                    color='black'
                )

                ax.text(

                    (
                            x1 + x2
                    ) / 2,

                    bar_y
                    +
                    0.025
                    *
                    y_range,

                    sig,

                    ha='center',

                    fontsize=14,

                    fontweight='bold'
                )


    # ======================
    # 坐标轴
    # ======================
    ax.set_xticks(
        positions
    )

    ax.set_xticklabels(

        categories,

        fontsize=20,

        fontweight='bold'
    )

    ax.set_ylabel(

        ylabel_text,

        fontsize=20,

        fontweight='bold'
    )

    ax.tick_params(

        axis='y',

        labelsize=16
    )

    title = (
        f'Distribution of '
        f'{prop} by '
        f'{class_col}'
    )

    if panel_label:

        title = (
            f'{panel_label} '
            +
            title
        )

    ax.set_title(

        title,

        fontsize=20,

        fontweight='bold',

        pad=12
    )

    ax.grid(

        axis='y',

        alpha=0.3,

        linestyle='--',

        linewidth=0.6
    )

    ax.set_axisbelow(
        True
    )

    for spine in (
            ax.spines.values()):

        spine.set_linewidth(
            1.5
        )

    ax.set_xlim(
        0.4,
        n_cat + 0.6
    )

    # ax.set_box_aspect(1)


def save_combined_figure(

        properties,

        outname,

        panel_labels,

        width_per_plot=6,

        height=5):

    fig, axes = plt.subplots(

        1,

        len(properties),

        figsize=(

            width_per_plot
            *
            len(properties),

            height
        )
    )

    if len(
            properties
    ) == 1:

        axes = [axes]

    for ax, prop, label in zip(

            axes,

            properties,

            panel_labels):

        plot_property(
            ax,
            prop,
            label
        )

    fig.subplots_adjust(

        left=0.07,

        right=0.98,

        bottom=0.12,

        top=0.90,

        wspace=0.18
    )

    fig.savefig(

        outname,

        dpi=600,

        bbox_inches='tight'
    )

    plt.close()


save_combined_figure(

    properties=[
        'pLC50',
        'BCF',
        'LogP'
    ],

    outname=
    'pLC50_BCF_LogP.png',

    panel_labels=[
        'A',
        'B',
        'C'
    ]
)

save_combined_figure(

    properties=[
        'MW',
        'TPSA'
    ],

    outname=
    'MW_TPSA.png',

    panel_labels=[
        'A',
        'B'
    ]
)

print(
    "\n组合图已生成"
)