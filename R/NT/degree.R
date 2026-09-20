library(ggplot2)
library(dplyr)
library(readr)

setwd("F:/dock/1")

df <- read_csv("degree.csv")

# 重新排序
df <- df %>%
  arrange(desc(Degree)) %>%
  mutate(gene = factor(gene, levels = rev(gene)))   # 上→下排序

# 繪圖
p <- ggplot(df, aes(y = gene, yend = gene, x = 0, xend = Degree)) +
  
  # 水平線段
  geom_segment(
    linewidth = 5,
    color = "#EF949E",
    lineend = "round"
  ) +
  
  # 數值標註
  geom_text(
    aes(x = Degree + 0.2, label = Degree),
    size = 8,
    fontface = "bold",
    hjust = 0,
    color = "black"
  ) +
  
  scale_x_continuous(
    name = "Degree",
    limits = c(0, max(df$Degree) * 1.25),
    breaks = seq(0, max(df$Degree), by = 5),
    expand = expansion(mult = c(0, 0.05))
  ) +
  
  scale_y_discrete(
    name = "Target gene abbreviation"
  ) +
  
  theme_minimal(base_size = 20) +
  theme(
    text = element_text(family = "Arial"),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    
    axis.line.x = element_line(color = "black", linewidth = 0.6),
    axis.line.y = element_blank(),
    
    axis.ticks.y = element_blank(),
    
    axis.text.y = element_text(
      face = "bold",
      size = 20,
      color = "black"
    ),
    
    axis.text.x = element_text(
      face = "bold",
      size = 24,
      color = "black"
    ),
    
    axis.title.x = element_text(face = "bold", size = 28, margin = margin(t = 10)),
    axis.title.y = element_text(face = "bold", size = 28, margin = margin(r = 10)),
    
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.8),
    plot.margin = margin(30, 50, 30, 30)
  ) +
  
  coord_cartesian(clip = "off")

ggsave("gene_degree_lollipop_vertical.png",
       plot = p,
       width = 6.5,
       height = 6.5,   # 高 = 宽 × 2
       dpi = 600)

