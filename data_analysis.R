library("dplyr")
library("ggpubr")
#library("xlsx")
library("readxl")
library("reshape2")
library("extrafont")
library("EnvStats")
library("RColorBrewer")
# font_import()        call only once
if (Sys.info()['sysname'] == "Linux") {
  # loadfonts(device = "win")
  setwd("/media/viacheslav/Dropbox/Python projects/mlc/")
  getwd() 
} else {
  extrafont::loadfonts(device = "win")
  setwd("C:/Clouds/Dropbox/Python projects/mlc")
  windowsFonts(Times = windowsFont("TT Times New Roman"))
  getwd()
}
theme_set(theme_linedraw(base_size = 52, base_family = "Times", base_line_size = 2, base_rect_size = 1.5))

# -------------------------------------------------------- Data --------------------------------------------------------
rm(list = ls())
graphics.off()
width <- 3400
height <- 1400
save_dir <- 'data_analysis/'
col_to_check <- 'age'
data <- read_excel("data/data.xlsx", sheet = 'Data')
head(data)
df_to_check <- subset(data, select = col_to_check)
head(df_to_check)
unique_vals = unique(df_to_check)
data_hist = aggregate(x = data.frame(count = df_to_check[[col_to_check]]), by = list(value = df_to_check[[col_to_check]]), FUN = length)
dir.create(save_dir, showWarnings = FALSE, mode = "0777")

if (col_to_check == "age") {
  data_hist = head(data_hist, -1)
  bar_width = 0.8
  x_lims = c(head(data_hist$value, n = 1) - 1, tail(data_hist$value, n = 1) + 2)
  y_lims = c(0, 2000)
  breaks = c(data_hist$value)
  text_angle = 90
  label_angle = 0
  vjust = 0.5
  hjust = -0.5
  x_label = tools::toTitleCase(col_to_check)
  y_breaks = seq(y_lims[1], y_lims[2], 500)
} else if (col_to_check == "sex") {
  bar_width = 0.4
  y_lims = c(0, 10000)
  breaks = c(data_hist$value)
  text_angle = 0
  label_angle = 0
  x_label = tools::toTitleCase(col_to_check)
  vjust = -1.5
  hjust = 0.5
  y_breaks = seq(y_lims[1], y_lims[2], 2000)
} else if (col_to_check == "mc_label") {
  bar_width = 0.8
  y_lims = c(0, 6000)
  breaks = c(data_hist$value)
  text_angle = 0
  label_angle = 0
  x_label = 'Class'
  vjust = -1.5
  hjust = 0.5
  y_breaks = seq(y_lims[1], y_lims[2], 2000)
} else if (col_to_check == "id") {
  bar_width = 0.8
  y_lims = c(0, 1250)
  breaks = c(data_hist$value)
  text_angle = 90
  label_angle = 90
  x_label = toupper(col_to_check)
  vjust = 0.5
  hjust = -0.5
  y_breaks = seq(y_lims[1], y_lims[2], 250)
}

# ----------------------------------------------------------- Barplot -----------------------------------------------------------
p <- ggplot(data_hist, aes(x = value, y=count)) +
  geom_bar(stat = "identity", width = bar_width, color = "black", size = 1, fill = "#FF6666") +
  geom_text(aes(label = sprintf("%.0f", round(count, 2))), vjust = vjust, hjust = hjust, family = "Times",  color = "black",
            size = 16, angle = text_angle) +
  scale_x_discrete(name = x_label) +                                                   # sex, mc_label, id
  #scale_x_continuous(name = x_label, limits = x_lims, breaks = c(data_hist$value)) +     # age
  coord_cartesian(ylim = y_lims) + 
  scale_y_continuous(name = "Frequency", breaks = y_breaks)  
p + theme(plot.margin = unit(c(0, 0, 0, 0), "pt"),
          legend.title = element_blank(),
          legend.position = "bottom",
          legend.margin = margin(0, 0, 0, 0, "pt"),
          legend.key.height = unit(7, "lines"),
          legend.key.width = unit(7, "lines"),
          legend.box.margin = margin(0, 0, 0, 0, "cm"),
          legend.box.spacing = unit(-1.5, "cm"), 
          panel.grid.minor.x = element_blank(),
          panel.grid.major.x = element_blank(),
          axis.text.x = element_text(angle = label_angle)
)
save_name = paste(save_dir, x_label, "_ordered_by_name.png", sep = "")
dev.print(png, save_name, width = width, height = height)

# ----------------------------------------------------------- Barplot -----------------------------------------------------------
p <- ggplot(data_hist, aes(x=reorder(value, -count), y=count)) +
  geom_bar(stat = "identity", width = bar_width, color = "black", size = 1, fill = "#FF6666") +
  geom_text(aes(label = sprintf("%.0f", round(count, 2))), vjust = vjust, hjust = hjust, family = "Times",  color = "black",
            size = 16, angle = text_angle) +
  scale_x_discrete(name = x_label) +                                                   # sex, mc_label, id
  #scale_x_continuous(name = x_label, limits = x_lims, breaks = c(data_hist$value)) +     # age
  coord_cartesian(ylim = y_lims) + 
  scale_y_continuous(name = "Frequency", breaks = y_breaks)  
p + theme(plot.margin = unit(c(0, 0, 0, 0), "pt"),
          legend.title = element_blank(),
          legend.position = "bottom",
          legend.margin = margin(0, 0, 0, 0, "pt"),
          legend.key.height = unit(7, "lines"),
          legend.key.width = unit(7, "lines"),
          legend.box.margin = margin(0, 0, 0, 0, "cm"),
          legend.box.spacing = unit(-1.5, "cm"), 
          panel.grid.minor.x = element_blank(),
          panel.grid.major.x = element_blank(),
          axis.text.x = element_text(angle = label_angle)
          )
save_name = paste(save_dir, x_label, "_ordered_by_value.png", sep = "")
dev.print(png, save_name, width = width, height = height)
