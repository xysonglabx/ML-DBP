library(ggplot2)
library(stringr)

.libPaths(c("F:/gdc_download/download_packages", .libPaths()))
library(showtext)

font_add("Arial", regular = "C:/Windows/Fonts/arial.ttf",bold    = "C:/Windows/Fonts/arialbd.ttf")
showtext_auto()
showtext_opts(dpi = 600)

data <- read.csv("KEGG_bubble.csv", stringsAsFactors = FALSE)

data$log_pval <- -log10(data$pvalue)

data$term_wrap <- str_wrap(data$term, width = 35)
data$term_wrap <- factor(
  data$term_wrap,
  levels = data$term_wrap[order(data$Enrichment, decreasing = TRUE)]
)

p <- ggplot(
  data,
  aes(
    x = Enrichment,
    y = term_wrap,
    size = count,
    color = log_pval
  )
) +
  geom_point(alpha = 0.65) +
  
  scale_color_gradient(
    low = "white",
    high = "#F08080",
    name = NULL,
    limits = c(2, 8),
    breaks = c(3, 4, 5, 6, 7),
    labels = c("3", "4", "5", "6", "7"),
    guide = guide_colorbar(
      barwidth = unit(0.55, "cm"),
      barheight = unit(5.4, "cm"),
      ticks = TRUE,
      label.position = "right",
      label.theme = element_text(
        family = "Arial",
        face = "bold",
        size = 20,
        color = "black"
      )
    )
  ) +
  
  scale_size_continuous(
    name = "count",
    range = c(5, 16),
    breaks = c(10, 12, 14, 18, 29)
  ) +
  
  scale_x_continuous(
    limits = c(0, 12),
    breaks = seq(0, 12, by = 2),
    expand = c(0.01, 0)
  ) +
  
  labs(x = "Enrichment", y = NULL) +

  theme_bw(base_family = "Arial", base_size = 22) +
  theme(
    aspect.ratio = 1,
    text = element_text(
      family = "Arial",
      face = "bold",
      color = "black"
    ),
    
    panel.border = element_rect(
      color = "black",
      linewidth = 1.2,
      fill = NA
    ),
    
    panel.grid.major.x = element_line(
      color = "grey82",
      linetype = "dashed",
      linewidth = 0.8
    ),
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    
    axis.title.x = element_text(
      family = "Arial",
      face = "bold",
      size = 36,
      color = "black",
      margin = margin(t = 10)
    ),
    axis.text.x = element_text(
      family = "Arial",
      face = "bold",
      size = 30,
      color = "#4d4d4d"
    ),
    axis.text.y = element_blank(),
    
    axis.ticks = element_line(
      color = "black",
      linewidth = 0.9
    ),
    axis.ticks.length = unit(0.18, "cm"),
    
    legend.position = "right",
    legend.title = element_text(
      family = "Arial",
      face = "bold",
      size = 28,
      color = "black"
    ),
    legend.text = element_text(
      family = "Arial",
      face = "bold",
      size = 24,
      color = "black"
    ),
    legend.key = element_blank(),
    legend.key.size = unit(0.8, "cm"),
    legend.spacing.y = unit(0.45, "cm"),
    legend.box.spacing = unit(0.45, "cm"),
    
    plot.margin = margin(10, 18, 10, 10)
  ) +
  
  guides(
    size = guide_legend(
      title = "count",
      title.theme = element_text(
        family = "Arial",
        face = "bold",
        size = 28,
        color = "black"
      ),
      label.theme = element_text(
        family = "Arial",
        face = "bold",
        size = 24,
        color = "black"
      ),
      override.aes = list(
        color = "gray45",
        alpha = 1
      )
    )
  )

ggsave(
  "KEGG_bubble_enrichment_swapXY_final.png",
  plot = p,
  width = 7,
  height = 7,
  dpi = 600,
  bg = "white"
)

ggsave(
  "KEGG_bubble_enrichment_swapXY_final.pdf",
  plot = p,
  width = 7,
  height = 7,
  device = cairo_pdf,
  bg = "white"
)

