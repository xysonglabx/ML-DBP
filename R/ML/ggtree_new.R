rm(list = ls())

# ===============================
# 0. 加载包
# ===============================
library(ggtreeExtra)
library(ggtree)
library(treeio)
library(tidytree)
library(ggstar)
library(ggplot2)
library(ggnewscale)
library(ape)
library(dplyr)
library(RColorBrewer)
library(grid)

# ===============================
# 0.1 字体设置
# ===============================
font_family <- "Arial"

base_font_size    <- 16
title_font_size   <- 26
legend_title_size <- 16
legend_text_size  <- 14
clade_label_size  <- 5.5
tree_scale_size   <- 4.2

# ===============================
# 1. 读取数据
# ===============================
data_file <- "classification.csv"

dat_raw <- read.csv(
  data_file,
  stringsAsFactors = FALSE,
  check.names = FALSE,
  fileEncoding = "UTF-8-BOM"
)

cat("当前数据列名：\n")
print(names(dat_raw))

# ===============================
# 2. CSV列映射
# ===============================
label_col   <- names(dat_raw)[1]
phylum_col  <- names(dat_raw)[3]
type_col    <- names(dat_raw)[4]
size_col    <- names(dat_raw)[5]

cat("\n识别到的列名：\n")
cat("Label          =", label_col, "\n")
cat("Protein        =", phylum_col, "\n")
cat("Halogen_Class  =", type_col, "\n")
cat("Binding Energy   =", size_col, "\n")

# ===============================
# 3. 数据整理
# ===============================
dat1 <- dat_raw %>%
  mutate(
    RawLabel = .data[[label_col]],

    Phylum = .data[[phylum_col]],
    Phylum = ifelse(
      is.na(Phylum) | trimws(Phylum) == "",
      "Unknown",
      Phylum
    ),

    Type = .data[[type_col]],
    Type = ifelse(
      is.na(Type) | trimws(Type) == "",
      "Unknown",
      Type
    ),

    Size = suppressWarnings(
      as.numeric(.data[[size_col]])
    )
  )

# Size缺失填补
size_median <- median(dat1$Size, na.rm = TRUE)

if (!is.finite(size_median)) {
  size_median <- 0
}

dat1$Size[is.na(dat1$Size)] <- size_median

# ===============================
# 3.1 分类顺序
# ===============================
phylum_count_df <- dat1 %>%
  count(Phylum, name = "class_count") %>%
  arrange(desc(class_count), Phylum)

# ===============================
# 自定义 Protein 显示顺序
# 这个顺序同时控制：
# 1. 柱状图颜色对应关系
# 2. Protein 图注标签顺序
# 3. clade标签颜色顺序
# ===============================
phylum_levels_manual <- c(
  "ALB",
  "AR",
  "CAT",
  "CYP2E1",
  "ESR1",
  "GST",
  "GSTP1",
  "KEAP1",
  "MGMT",
  "NF",
  "NQO1",
  "PARP1",
  "SOD1",
  "TP53",
  "Thioredoxin"
)

# 只保留数据中实际存在的类别，避免报错
phylum_levels <- phylum_levels_manual[
  phylum_levels_manual %in% unique(dat1$Phylum)
]

# 如果数据中存在未写入手动列表的类别，自动补到最后
phylum_levels <- c(
  phylum_levels,
  setdiff(
    unique(dat1$Phylum),
    phylum_levels
  )
)

dat1$Phylum <- factor(
  dat1$Phylum,
  levels = phylum_levels
)

dat1$Phylum <- factor(
  dat1$Phylum,
  levels = phylum_levels
)

dat1$Type <- factor(dat1$Type)

cat("\n各 Protein 类别计数（按降序）：\n")
print(phylum_count_df)

# 排序
dat1 <- dat1 %>%
  arrange(
    Phylum,
    Type,
    desc(Size)
  )

# 重新生成ID
dat1$ID <- paste0(
  "Mol_",
  seq_len(nrow(dat1))
)

# label
dat1$TipLabel <- ifelse(
  is.na(dat1$RawLabel) |
    trimws(dat1$RawLabel) == "",
  dat1$ID,
  dat1$RawLabel
)

# ===============================
# 4. 颜色与形状
# ===============================
n_phylum <- length(phylum_levels)

# Protein 颜色顺序：蓝 -> 紫灰 -> 红
# 顺序严格对应 phylum_levels
phylum_cols_manual <- c(
  "CAT"         = "#08306B",
  "AR"          = "#08519C",
  "ALB"         = "#2171B5",
  "CYP2E1"      = "#4292C6",
  "MGMT"        = "#6BAED6",
  "GSTP1"         = "#9ECAE1",
  "ESR1"       = "#C6DBEF",
  "SOD1"       = "#B7B4C9",
  "NF"        = "#F4A6A6",
  "Thioredoxin" = "#FB6A6A",
  "NQO1"        = "#EF3B2C",
  "PARP1"       = "#CB181D",
  "TP53"        = "#A50F15",
  "KEAP1"        = "#7F0000",
  "GST" = "#67000D"
)

# 只保留当前数据中存在的 Protein
phylum_cols <- phylum_cols_manual[
  phylum_levels
]

# 如果数据里有 phylum_cols_manual 没写到的新类别，自动补色
missing_cols <- setdiff(
  phylum_levels,
  names(phylum_cols_manual)
)

if (length(missing_cols) > 0) {

  extra_cols <- colorRampPalette(
    c("#08306B", "#6BAED6", "#F4A6A6", "#A50F15")
  )(length(missing_cols))

  names(extra_cols) <- missing_cols

  phylum_cols <- c(
    phylum_cols[!is.na(phylum_cols)],
    extra_cols
  )
}

phylum_cols <- phylum_cols[
  phylum_levels
]
# ===============================
# Protein 图注显示顺序：蓝 -> 红
# 注意：只控制图注顺序，不改变蛋白对应颜色
# ===============================
protein_legend_order <- c(
  "CAT",
  "AR",
  "ALB",
  "CYP2E1",
  "MGMT",
  "GSTP1",
  "ESR1",
  "SOD1",
  "NF",
  "Thioredoxin",
  "NQO1",
  "PARP1",
  "TP53",
  "KEAP1",
  "GST"
)

# 只保留当前数据中存在的 Protein
protein_legend_order <- protein_legend_order[
  protein_legend_order %in% phylum_levels
]

# 如果有未列入的新 Protein，补到最后
protein_legend_order <- c(
  protein_legend_order,
  setdiff(phylum_levels, protein_legend_order)
)
# ===============================
# Type 自动形状
# ===============================
type_levels <- levels(dat1$Type)

shape_pool <- c(
  1,   # 圆
  15,  # 星形/方形类符号，取决于 ggstar 版本
  11   # 三角类符号
)

type_shapes <- rep(
  shape_pool,
  length.out = length(type_levels)
)

names(type_shapes) <- type_levels
# ===============================
# 5. 构树
# ===============================
X_phylum <- model.matrix(
  ~ Phylum - 1,
  data = dat1
) * 6

X_type <- model.matrix(
  ~ Type - 1,
  data = dat1
) * 1.8

X_size <- scale(dat1$Size) * 1

X <- cbind(
  X_phylum,
  X_type,
  BE = as.numeric(X_size)
)

keep_cols <- apply(
  X,
  2,
  sd,
  na.rm = TRUE
) > 0

X <- X[, keep_cols, drop = FALSE]

hc <- hclust(
  dist(X),
  method = "average"
)

tree <- as.phylo(hc)

tree <- ladderize(
  tree,
  right = FALSE
)

tree$tip.label <- dat1$ID

desired_tip_order_clockwise <- rev(dat1$ID)

tree <- rotateConstr(
  tree,
  desired_tip_order_clockwise
)

# ===============================
# 6. 压缩枝长
# ===============================
if (!is.null(tree$edge.length)) {

  pos_edge_min <- suppressWarnings(
    min(
      tree$edge.length[
        tree$edge.length > 0
      ],
      na.rm = TRUE
    )
  )

  if (!is.finite(pos_edge_min)) {
    pos_edge_min <- 1e-6
  }

  tree$edge.length[
    tree$edge.length <= 0
  ] <- pos_edge_min

  tree$edge.length <- tree$edge.length ^ 0.35

  tree$edge.length <- tree$edge.length /
    max(tree$edge.length, na.rm = TRUE) * 0.82
}

# ===============================
# 7. 构造clade
# ===============================
root_node <- Ntip(tree) + 1

clade_df <- lapply(
  levels(dat1$Phylum),
  function(g) {

    tips <- dat1$ID[
      dat1$Phylum == g
    ]

    tips <- intersect(
      tips,
      tree$tip.label
    )

    if (length(tips) >= 2) {

      node <- getMRCA(
        tree,
        tips
      )

      data.frame(
        node = node,
        Phylum = as.character(g),
        n = length(tips),
        stringsAsFactors = FALSE
      )

    } else {

      NULL
    }
  }
) %>%
  bind_rows() %>%
  filter(!is.na(node)) %>%
  filter(node != root_node)

clade_df <- clade_df %>%
  left_join(
    phylum_count_df,
    by = "Phylum"
  ) %>%
  arrange(desc(class_count))

# ===============================
# 8. heatmap / bar数据
# ===============================
size_range <- range(
  dat1$Size,
  na.rm = TRUE
)

if (
  !all(is.finite(size_range)) ||
  diff(size_range) == 0
) {

  dat1$Size01 <- 1

} else {

  dat1$Size01 <- (
    dat1$Size - size_range[1]
  ) / diff(size_range)
}

# 热图数据
dat2 <- dat1 %>%
  transmute(
    ID = ID,
    Sites = "Binding Energy",
    Abundance = Size01
  )

# 柱状图数据
# Binding Energy 通常是负值；
# 外圈柱状图朝外绘制时，柱长使用绝对值。
dat3 <- dat1 %>%
  transmute(
    ID = ID,
    HigherAbundance = abs(Size),
    Sites = Phylum
  )

dat2$Sites <- factor(
  dat2$Sites,
  levels = "Binding Energy"
)

dat3$Sites <- factor(
  dat3$Sites,
  levels = phylum_levels
)

# ===============================
# 9. 基础树
# ===============================
p <- ggtree(
  tree,
  layout = "fan",
  size = 0.10,
  open.angle = 8,
  color = "grey45"
)

base_tree_data <- p$data

max_x <- max(
  base_tree_data$x,
  na.rm = TRUE
)

# ===============================
# 10. clade高亮
# ===============================
extend_to <- max_x + 0.5

if (nrow(clade_df) > 0) {

  for (i in seq_len(nrow(clade_df))) {

    this_node <- clade_df$node[i]

    p <- p +
      geom_hilight(
        node = this_node,
        extendto = extend_to,
        alpha = 0.12,
        fill = "grey80",
        color = "grey60",
        size = 0.04
      )
  }
}

# ===============================
# 11. clade标签
# 放到对应圈外
# ===============================

#angle_fix <- function(a) {

 # ifelse(
  #  a > 90,
 #   a - 180,
 #   ifelse(
#      a < -90,
 #     a + 180,
 #     a
#    )
#  )
#}

#hjust_fix <- function(a) {

  #ifelse(
  #  a > 90 | a < -90,
 #   1,
#    0
 # )
#}

#lab_df <- clade_df %>%
  #left_join(
  #  base_tree_data %>%
   #   select(node, x, y, angle),
  #  by = "node"
 # ) %>%
 # mutate(

    # 放到最外圈外侧
   # label_x = extend_to + 0.6,

   # label_y = y,
#
  #  label_angle = angle_fix(angle),

   # label_hjust = hjust_fix(angle)
 # )

#p <- p +
 # geom_text(
 #   data = lab_df,
 #   inherit.aes = FALSE,
 #   aes(
 #     x = label_x,
 ##     y = label_y,
    #  label = Phylum,
  #    angle = label_angle,
  #    hjust = label_hjust,
 #     color = Phylum
  #  ),
#    size = clade_label_size,
 #   family = font_family,
   # fontface = "bold"
 # ) +
  #scale_color_manual(
   # values = phylum_cols,
    #guide = "none"
  #)

# ===============================
# 12. 合并tip数据
# ===============================
p$data <- left_join(
  p$data,
  dat1,
  by = c("label" = "ID")
)

tip_dat <- p$data %>%
  filter(isTip) %>%
  arrange(y)

# ===============================
# 13. 星标错落
# ===============================
ring_offsets <- c(
  0.000,
  0.018,
  0.036,
  0.054,
  0.072,
  0.090
)

tip_dat <- tip_dat %>%
  mutate(
    star_offset =
      ring_offsets[
        (row_number() - 1) %%
          length(ring_offsets) + 1
      ]
  )
# ===============================
# 13.1 内圈图标颜色：按圆周方向蓝到红渐变 + 透明度
# ===============================

# 图标透明度：
# 1 = 不透明；0.6 = 较透明；0.75 推荐
inner_icon_alpha <- 0.78

# 颜色方向：
# "clockwise"        顺时针方向：蓝 -> 红
# "counterclockwise" 逆时针方向：蓝 -> 红
inner_icon_direction <- "clockwise"

# 蓝 -> 红渐变色
inner_icon_colours <- c(
  "#08306B",
  "#08519C",
  "#2171B5",
  "#4292C6",
  "#6BAED6",
  "#9ECAE1",
  "#C6DBEF",
  "#B7B4C9",
  "#F4A6A6",
  "#FB6A6A",
  "#EF3B2C",
  "#CB181D",
  "#A50F15",
  "#7F0000",
  "#67000D"
)

tip_dat <- tip_dat %>%
  arrange(y) %>%
  mutate(
    IconColorIndex = row_number(),
    IconSize = abs(Size)
  )

if (inner_icon_direction == "counterclockwise") {
  tip_dat <- tip_dat %>%
    mutate(
      IconColorIndex = max(IconColorIndex, na.rm = TRUE) - IconColorIndex + 1
    )
}

# ===============================
# Binding Energy 绝对值图例刻度
# ===============================
energy_abs_range <- range(
  tip_dat$IconSize,
  na.rm = TRUE
)

size_breaks <- pretty(
  energy_abs_range,
  n = 4
)

size_breaks <- size_breaks[
  size_breaks >= energy_abs_range[1] &
    size_breaks <= energy_abs_range[2]
]

# 图注显示为绝对值
size_labels <- paste0("-", size_breaks)

size_breaks <- size_breaks[
  size_breaks >= min(tip_dat$IconSize, na.rm = TRUE) &
    size_breaks <= max(tip_dat$IconSize, na.rm = TRUE)
]

if (all(dat1$Size <= 0, na.rm = TRUE)) {

  size_labels <- ifelse(
    size_breaks == 0,
    "0",
    paste0("-", size_breaks)
  )

} else {

  size_labels <- as.character(size_breaks)
}

# ===============================
# 14. tip星标
# 内圈图标：蓝 -> 红渐变，并带透明度
# ===============================
p <- p +
  geom_star(
    data = tip_dat,
    inherit.aes = FALSE,
    aes(
      x = x + star_offset,
      y = y,
      fill = IconColorIndex,
      starshape = Type,
      size = IconSize
    ),
    alpha = inner_icon_alpha,
    color = grDevices::adjustcolor(
      "black",
      alpha.f = 0.35
    ),
    starstroke = 0.08
  ) +

  scale_fill_gradientn(
    colours = inner_icon_colours,
    guide = "none"
  ) +

scale_starshape_manual(
  values = type_shapes,
  name = "Halogen_Class",
  guide = guide_legend(
    order = 2,
    override.aes = list(
      size = 4   # <- 这里设置图例中图标的大小，可以调大或调小
    )
  )
)+

scale_size_continuous(
  name = "Binding Energy",
  range = c(1, 4),
  breaks = size_breaks,
  labels = size_labels,
  guide = guide_legend(
    order = 3,
    override.aes = list(
      alpha = 1,
      fill = "grey80",
      color = "black",
      starshape = type_shapes[1],
      starstroke = 0.08
    )
  )
)
# ===============================
# 15. 外圈
# ===============================

# 热图圈（缩小）
heatmap_offset <- 0.035
heatmap_width  <- 0.060

# 柱状图圈
bar_offset <- 0.045
bar_width  <- 0.22

p <- p +

  # ====================================
  # 热图（连续变量）
  # ====================================
  new_scale_fill() +

  geom_fruit(
    data = dat2,
    geom = geom_tile,
    mapping = aes(
      y = ID,
      x = Sites,
      alpha = Abundance,
      fill = Sites
    ),
    color = "grey55",
    offset = heatmap_offset,
    pwidth = heatmap_width,
    size = 0.02,
    show.legend = c(
      alpha = TRUE,
      fill = FALSE
    )
  ) +

  scale_alpha_continuous(
    name = "Abundance",
    range = c(0.25, 1),
    guide = guide_legend(
      order = 1,
      override.aes = list(
        fill = "#315A89",
        color = "grey55"
      )
    )
  ) +

  scale_fill_manual(
    values = c(
      "Binding Energy" = "#315A89"
    ),
    guide = "none"
  ) +

  # ====================================
  # 重置 fill scale
  # ====================================
  new_scale_fill() +

  # ====================================
  # 柱状图（离散变量）
  # ====================================
  geom_fruit(
    data = dat3,
    geom = geom_bar,

    mapping = aes(
      y = ID,
      x = HigherAbundance,
      fill = Sites
    ),

    offset = bar_offset,

    pwidth = bar_width,

    orientation = "y",

    stat = "identity",

    width = 0.42,

    alpha = 1,

    color = NA
  ) +

scale_fill_manual(
  values = phylum_cols,              # 保持 蛋白名-颜色 对应关系不变
  breaks = protein_legend_order,     # 只改变图注显示顺序
  limits = protein_legend_order,     # 图注按蓝 -> 红排列
  name = "Protein",
  drop = FALSE, 
 guide = guide_legend(
    order = 4
  )
)
# ===============================
# 16. 主题
# ===============================
p <- p +
  ggtitle(
    "DBPs–Protein TDbook-style Tree"
  ) +

  theme(

    text = element_text(
      family = font_family,
      size = base_font_size,
      face = "bold"
    ),

    plot.title = element_text(
      family = font_family,
      size = title_font_size,
      hjust = 0.02,
      face = "bold"
    ),

    legend.position = c(
      1.10,
      0.45
    ),

    legend.title = element_text(
      family = font_family,
      size = legend_title_size,
      face = "bold"
    ),

    legend.text = element_text(
      family = font_family,
      size = legend_text_size,
      face = "bold"
    ),

    # 给右侧标签留空间
    plot.margin = margin(
      10,
      400,
      10,
      10
    )
  )

# ===============================
# 17. 输出
# ===============================
print(p)

ggsave(
  "classification_tree0.png",
  p,
  width = 16,
  height = 11,
  dpi = 600,
  limitsize = FALSE
)

ggsave(
  "classification_tree0.pdf",
  p,
  width = 16,
  height = 11,
  device = cairo_pdf,
  limitsize = FALSE
)


