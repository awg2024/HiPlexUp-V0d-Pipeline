
#####################################
#### HiPlexUp AUTOMATED ANALYSIS ####
#####################################

# Images have been cropped to grey matter only (butterfly shape)
# Segmentation has been performed with Zen software (NT based, 80% confidence, 50 um^2 minimum size, watershedding xxx)
# Quantification for each cell/object includes: area, center coordinates, circularity, compactness, intensity mean and SD for each channel
# Tissue background intensities are also quantified and need to be subtracted to cell intensities



### PREP R

## Set current working directory
setwd("/Users/angusgray/Desktop/V0d-Histology/HiPlexUp-V0d-Pipeline")

## Activate packages
library(dplyr)
library(readxl)
library(rearrr)
library(ggplot2)
library(ggthemes)
library(tidyr)
library(cowplot)
library(writexl)
library(svglite)



#######################
####   LOAD DATA   ####
#######################

### Load excel file for all NT (cell intensities) (microscope measurements) (reads readxls data after segmentation)
NT <- read_xlsx("raw_xls/NT.xlsx")

### ADD ANIMAL ID

## Load excel file for animal IDs (needs readxl)
## Add animal ID columns to data file based on "Image" (it will duplicate it depending on Image ID), then move after "Image" (needs dplyr)

Animals <- read_xlsx("raw_xls/Animals.xlsx")

## Perform a left-merge, merging the animal and cell intensity datasets 
NT <- merge(NT, Animals, by="Image", all.x=TRUE)
NT <- relocate(NT, Animal, .after = Image)

## remove temporary animals spreadsheet we added in 
rm(Animals)



###########################
####   NORMALIZATION   ####
###########################

### SET NEW ORIGIN (CC 0,0) AND ROTATE -- corpus collosum as a reference point here for spatial normalisation 

## Load .xlsx file for CC coordinates and angle (new 0, 0) (needs readxl)
# Add rotation info to Markers dataframe, based on Image (needs dplyr)
## Transform X, Y coordinates to new 0,0 by subtraction of CC coordinates, and delete CC coordinates
# Rotate coordinates based on angle and new 0,0 origin using group_map to split the data by Image and separately apply the rotate_2d function, then bind and delete extra columns
#(for the angle, use either mean/unique/min to pick a single value - doesn't matter since they are all the same) (neads dplyr & rearrr)
## Visualize cells after rotation (needs ggplot2)

Rotation <- read_xlsx("Rotation&CC.xlsx") ## read an external csv script that contains x,y coordinates of the CC

NT <- merge(NT, Rotation, by="Image", all.x=TRUE)

NT$X <- (NT$X - NT$CC_X)
NT$Y <- (NT$Y - NT$CC_Y)
NT <- subset(NT, select = -c(CC_X, CC_Y))

NT <- NT %>% group_by(Image) %>% 
  group_map(
    ~ rotate_2d(data = ., degrees = mean(.[["Angle"]], na.rm = TRUE), x_col = "X", y_col = "Y", origin = c(0,0), suffix = '', overwrite = TRUE),
    .keep = TRUE
  ) %>% bind_rows()
NT <- subset(NT, select = -c(Angle, .origin, .degrees)) ## ensure cells face the same way 
rm(Rotation)

ggplot(data = NT, aes(X, Y)) +  ## plot the location of the cells, these should be overlapping instead of spread out due to this normalisation, this might be code we delete for our V0d pipeline 
  geom_point(aes(color = Image), size = 0.5) +
  guides(color = guide_legend(ncol = 2)) +
  ggtitle("All cells (after rotation)")

# ggplot(data = NT, aes(X, Y)) +
#   geom_point(aes(color = Image), size = 0.5) +
#   guides(color = guide_legend(ncol = 2)) +
#   facet_wrap(~ Animal, nrow = 3) +
#   ggtitle("All cells (after rotation)")



### SPLIT LEFT/RIGHT HEMICORDS

## Add column stating left/right depending on X coordinate, change X vlues to positives for left hemicords, rename "Image" to include hemicord info (needs dplyr)
## Visualize cells after hemicord split (needs ggplot2)

NT %>% mutate(Hemicord = case_when(X >= 0 ~ "Right", X < 0 ~ "Left")) -> NT ## seperate them based on x coordinate so greater or less than can correspond with hemicord location 
NT %>% mutate(X = case_when(Hemicord == "Left" ~ X*(-1), Hemicord == "Right" ~ X)) -> NT
NT %>% unite(Image, c(Image, Hemicord), sep = "_") -> NT

ggplot(data = NT, aes(X, Y)) + ## plotting the split 
  geom_point(aes(color = Image), size = 0.5) +
  guides(color = guide_legend(ncol = 4)) +
  ggtitle("All cells (after hemicord split)")

# ggplot(data = NT, aes(X, Y)) +
#   geom_point(aes(color = Image), size = 0.5) +
#   guides(color = guide_legend(ncol = 4)) +
#   facet_wrap(~ Animal, nrow = 3) +
#   ggtitle("All cells (after hemicord split)")



### NORMALIZE FOR GRAY MATTER WIDTH/HEIGHT, AND POSITION

## Load .xlsx file for width and height normalization
## Add columns for width & height factors and  normalize  X,Y coordinates, then visualize (needs dplyr & ggplot2)
#(this doesn't work if the MASS package is loaded because it masks the 'select' function from the dplyr package)

Size <- read_xlsx("Size_normalization.xlsx") ## size normalisation of the x,y coordinates 

NT <- merge(NT, Size %>% select(Image, Width, Height), by="Image", all.x=TRUE)

NT$X <- (NT$X * NT$Width)
NT$Y <- (NT$Y * NT$Height)

NT <- subset(NT, select = -c(Width, Height))

rm(Size)

NT <- NT %>% drop_na(X)

ggplot(data = NT, aes(X, Y)) +
  geom_point(aes(color = Image), size = 0.5) +
  guides(color = guide_legend(ncol = 4)) +
  ggtitle("All cells (after width/height normalization)")

# ggplot(data = NT, aes(X, Y)) +
#   geom_point(aes(color = Image), size = 0.5) +
#   guides(color = guide_legend(ncol = 4)) +
#   facet_wrap(~ Animal, nrow = 3) +
#   ggtitle("All cells (after width/height normalization)")


## Find min Y coordinate for each Image and subtract from each Y coordinate (move all Images up to Y=0 coordinate), then visualize (needs dplyr & ggplot2)

NT %>% group_by(Image) %>% summarise(minY = min(Y)) -> minY

NT <- merge(NT, minY, by = "Image", all.x = TRUE)

NT$Y <- (NT$Y - NT$minY)

NT <- subset(NT, select = -c(minY))

rm(minY)

ggplot(data = NT, aes(X, Y)) +
  geom_point(aes(color = Image), size = 0.5) +
  guides(color = guide_legend(ncol = 4)) +
  ggtitle("All cells (after position normalization)")

# ggplot(data = NT, aes(X, Y)) +
#   geom_point(aes(color = Image), size = 0.5) +
#   guides(color = guide_legend(ncol = 4)) +
#   facet_wrap(~ Animal, nrow = 3) +
#   ggtitle("All cells (after position normalization)")




#######################
####   PREP DATA   ####
#######################

### ADD THRESHOLD INFO

## Load excel file for thresholds (needs readxl)
## Add threshold columns to data file based on "Image" (it will duplicate the threshold value depending on Image ID) (needs dplyr)

Thresholds <- read_xlsx("Thresholds_high-low.xlsx")
Thresholds[is.na(Thresholds)] = Inf

NT_thr <- merge(NT, Thresholds %>% select(-Segmentation, -Notes), by="Image", all.x=TRUE)
rm(Thresholds)



### REMOVE DETECTED CELLS DEPENDING ON SIZE

## Delete cells that are smaller than 70 um2 (& bigger than 2500 um2?) ## what is this in reference to? 

NT_thr <- subset(NT_thr, Area_um2 >= 70)
#NT_thr <- subset(NT_thr, Area_um2 <= 2500)



### PREP DATA FOR X, Y VISUALIZATION

## Move X, Y to first columns & remove ID column
## Split "Image" column, delete ROI info, and add "Hemisection" column (needs tidyr & dplyr)
## Rearrange rows by Group and Timepoint (optional, the problem with graph order comes from facet_grid)

NT_thr <- relocate(NT_thr, X, .after = Animal)
NT_thr <- relocate(NT_thr, Y, .after = X)
NT_thr <- subset(NT_thr, select = -c(ID))

NT_thr <- separate(NT_thr, Image, c("Timepoint", "Group", "Scene", "ROI", "Hemicord"), "_")
NT_thr <- subset(NT_thr, select = -c(ROI))

NT_thr <- NT_thr %>% arrange(match(Timepoint, c("P30", "P63", "P112")), desc(Group))



## OPTIONAL: Remove P30 / Select P112

#NT_thr <- filter(NT_thr, !grepl("P30", Timepoint))
#NT_thr <- subset(NT_thr, Timepoint == "P112")




#####################################
####   IDENTIFY POSITIVE CELLS   #### CHANGE FOR V0D, for our analysis would it make sense to actually identify more than just v0d? have other ones to identify so we can meanfully train the model?
#####################################

#### CREATE SUBSETS FOR EACH CELL TYPE (based on thresholds & coordinates)

## Inhibitory

GlyT2 <- subset(NT_thr, GlyT2_Intensity.mean >= GlyT2_low)
Gad65 <- subset(NT_thr, Gad65_Intensity.mean >= Gad65_low & Gad65_Intensity.std >= Gad65_SD)
Gad67 <- subset(NT_thr, Gad67_Intensity.mean >= Gad67_low & Gad67_Intensity.std >= Gad67_SD)

Inhibitory <- subset(NT_thr, GlyT2_Intensity.mean >= GlyT2_low |
                       (Gad65_Intensity.mean >= Gad65_low & Gad65_Intensity.std >= Gad65_SD) |
                       (Gad67_Intensity.mean >= Gad67_low & Gad67_Intensity.std >= Gad67_SD))


## Excitatory

Vglut2 <- subset(NT_thr, Vglut2_Intensity.mean >= Vglut2_low)
Excitatory <- Vglut2





## Cholinergic

ChAT <- subset(NT_thr, ChAT_Intensity.mean >= ChAT_low & ChAT_Intensity.mean < ChAT_high & ChAT_Intensity.std >= ChAT_SD)
ChAT_MN <- subset(ChAT, Y <= 250)


## V1

V1 <- subset(Inhibitory, En1_Intensity.mean < En1_high & En1_Intensity.std >= En1_SD & Y <= 900)

V1_Foxp2 <- subset(V1, Foxp2_Intensity.mean >= Foxp2_low)
V1_Pou6f2 <- subset(V1, Pou6f2_Intensity.mean >= Pou6f2_low)
V1_Sp8 <- subset(V1, Sp8_Intensity.mean >= Sp8_low & Sp8_Intensity.mean < Sp8_high)
V1_Renshaw <- subset(V1, Calb1_Intensity.mean >= Calb1_low & Calb1_Intensity.mean < Calb1_high & Calb1_Intensity.std >= Calb1_SD & Y <= 250)
V1_Ia <- subset(V1, Calb2_Intensity.mean >= Calb2_low & Calb2_Intensity.std >= Calb2_SD & Y <= 500 & X >= 500)
V1_Pvalb <- subset(V1, Pvalb_Intensity.mean >= Pvalb_low)


## V2a

V2a <- subset(Excitatory, Chx10_Intensity.mean >= Chx10_low & Chx10_Intensity.mean < Chx10_high & Chx10_Intensity.std >= Chx10_SD & Y <= 900)


## Shox2

Shox2 <- subset(Excitatory, Shox2_Intensity.mean >= Shox2_low & Y <= 900)
Shox2_V2a <- subset(Shox2, Chx10_Intensity.mean >= Chx10_low & Chx10_Intensity.mean < Chx10_high & Chx10_Intensity.std >= Chx10_SD)
Shox2_nonV2a <- subset(Shox2, Chx10_Intensity.mean < Chx10_low | Chx10_Intensity.mean >= Chx10_high | Chx10_Intensity.std < Chx10_SD)


## V0

Pitx2 <- subset(NT_thr, Ptx2_Intensity.mean >= Ptx2_low & Ptx2_Intensity.mean < Ptx2_high)
V0_cg <- subset(Pitx2, Y >= 400 & Y <= 650 & X <= 200)
V0_c <- subset(V0_cg, ChAT_Intensity.mean >= ChAT_low & ChAT_Intensity.mean < ChAT_high & ChAT_Intensity.std >= ChAT_SD)
V0_g <- subset(V0_cg, Vglut2_Intensity.mean >= Vglut2_low)


## V0D


## we need to assign threshold for the lowest amount of intensity to detect this particular cell. 


## VGAT
## PAX2 
## EVX1 
## DBX1 



### LIST ALL INTERNEURON SUBPOPULATIONS

pops <- list("GlyT2" = GlyT2, "Gad65" = Gad65, "Gad67" = Gad67, "Inhibitory" = Inhibitory,
             "Excitatory" = Excitatory, "ChAT" = ChAT, "ChAT_MN" = ChAT_MN,
             "V1" = V1, "V1_Foxp2" = V1_Foxp2, "V1_Pou6f2" = V1_Pou6f2, "V1_Sp8" = V1_Sp8,
             "V1_Renshaw" = V1_Renshaw, "V1_Ia" = V1_Ia, "V1_Pvalb" = V1_Pvalb,
             "V2a" = V2a, "Shox2" = Shox2, "Shox2_V2a" = Shox2_V2a, "Shox2_nonV2a" = Shox2_nonV2a,
             "Pitx2" = Pitx2, "V0cg" = V0_cg, "V0_c" = V0_c, "V0_g" = V0_g)




################################
####   PLOT DISTRIBUTIONS   ####
################################


windowsFonts(Arial = windowsFont("Arial"))


## For loop

for (idx in 1:length(pops)) {
  
  data <- pops[[idx]]
  
  
  ggplot(subset(data, Timepoint == "P112"), aes(X, Y)) +
    xlim(0, 1100) + xlab("X (um)") +
    ylim(-100, 1300) + ylab("Y (um)") +
    geom_point(color = "black", size = 1) +
    geom_density_2d_filled(contour_var = "count", alpha=0.85, bins = 10) +
    facet_grid(. ~ factor(Group, levels=c("WT", "SOD1"))) +
    theme_few(base_family = "Arial") +
    theme(legend.position = "none",
          axis.title.y = element_text(color = "black", size = 14), axis.text.y = element_text(color = "black", size = 14),
          axis.title.x = element_text(color = "black", size = 14), axis.text.x = element_text(color = "black", size = 14),
          axis.ticks.y = element_line(color = "black"), axis.ticks.x = element_line(color = "black"),
          panel.border = element_rect(color = "black", size = 1), strip.text.x = element_text(color = "black", size = 18))
  ggsave(file=paste0("20230111_figures/P112_counts/", names(pops)[idx], "_P112_counts.svg"), plot=last_plot(), width=13.861, height=9.19)
  
  ggplot(subset(data, Timepoint == "P112"), aes(y=Y, color=Group)) +
    scale_color_manual(values = c("WT" = "#A0A0A4", "SOD1" = "#901A1E")) +
    ylim(-100, 1300) + ylab("Y (um)") + xlab("Count") +
    geom_density(size = 1, aes(x = ..count..)) +
    theme_few(base_family = "Arial") +
    theme(legend.position = "none",
          axis.title.y = element_blank(), axis.text.y = element_blank(), axis.ticks.y = element_blank(),
          axis.title.x = element_text(color = "black", size = 14), axis.text.x = element_text(color = "black", size = 14),
          axis.ticks.x = element_line(color = "black"), panel.border = element_rect(color = "black", size = 1))
  ggsave(file=paste0("20230111_figures/P112_XY/", names(pops)[idx], "_P112_Y.svg"), plot=last_plot(), width=2.97, height=8.825)
  
  ggplot(subset(data, Timepoint == "P112"), aes(x=X, color=Group)) +
    scale_color_manual(values = c("WT" = "#A0A0A4", "SOD1" = "#901A1E")) +
    xlim(0, 1100) + xlab("X (um)") + ylab("Count") +
    geom_density(size=1, aes(y = ..count..)) +
    theme_few(base_family = "Arial") +
    theme(legend.position = "none",
          axis.title.y = element_text(color = "black", size = 14), axis.text.y = element_text(color = "black", size = 14),
          axis.title.x = element_text(color = "black", size = 14), axis.text.x = element_text(color = "black", size = 14),
          axis.ticks.y = element_line(color = "black"), axis.ticks.x = element_line(color = "black"),
          panel.border = element_rect(color = "black", size = 1))
  ggsave(file=paste0("20230111_figures/P112_XY/", names(pops)[idx], "_P112_X.svg"), plot=last_plot(),  width=7.174, height=3.4)
  
  
  ggplot(subset(data, Timepoint == "P63"), aes(X, Y)) +
    xlim(0, 1100) + xlab("X (um)") +
    ylim(-100, 1300) + ylab("Y (um)") +
    geom_point(color = "black", size = 1) +
    geom_density_2d_filled(contour_var = "count", alpha=0.85, bins = 10) +
    facet_grid(. ~ factor(Group, levels=c("WT", "SOD1"))) +
    theme_few(base_family = "Arial") +
    theme(legend.position = "none",
          axis.title.y = element_text(color = "black", size = 14), axis.text.y = element_text(color = "black", size = 14),
          axis.title.x = element_text(color = "black", size = 14), axis.text.x = element_text(color = "black", size = 14),
          axis.ticks.y = element_line(color = "black"), axis.ticks.x = element_line(color = "black"),
          panel.border = element_rect(color = "black", size = 1), strip.text.x = element_text(color = "black", size = 18))
  ggsave(file=paste0("20230111_figures/P63_counts/", names(pops)[idx], "_P63_counts.svg"), plot=last_plot(), width=13.861, height=9.19)
  
  ggplot(subset(data, Timepoint == "P63"), aes(y=Y, color=Group)) +
    scale_color_manual(values = c("WT" = "#A0A0A4", "SOD1" = "#901A1E")) +
    ylim(-100, 1300) + ylab("Y (um)") + xlab("Count") +
    geom_density(size = 1, aes(x = ..count..)) +
    theme_few(base_family = "Arial") +
    theme(legend.position = "none",
          axis.title.y = element_blank(), axis.text.y = element_blank(), axis.ticks.y = element_blank(),
          axis.title.x = element_text(color = "black", size = 14), axis.text.x = element_text(color = "black", size = 14),
          axis.ticks.x = element_line(color = "black"), panel.border = element_rect(color = "black", size = 1))
  ggsave(file=paste0("20230111_figures/P63_XY/", names(pops)[idx], "_P63_Y.svg"), plot=last_plot(), width=2.97, height=8.825)
  
  ggplot(subset(data, Timepoint == "P63"), aes(x=X, color=Group)) +
    scale_color_manual(values = c("WT" = "#A0A0A4", "SOD1" = "#901A1E")) +
    xlim(0, 1100) + xlab("X (um)") + ylab("Count") +
    geom_density(size=1, aes(y = ..count..)) +
    theme_few(base_family = "Arial") +
    theme(legend.position = "none",
          axis.title.y = element_text(color = "black", size = 14), axis.text.y = element_text(color = "black", size = 14),
          axis.title.x = element_text(color = "black", size = 14), axis.text.x = element_text(color = "black", size = 14),
          axis.ticks.y = element_line(color = "black"), axis.ticks.x = element_line(color = "black"),
          panel.border = element_rect(color = "black", size = 1))
  ggsave(file=paste0("20230111_figures/P63_XY/", names(pops)[idx], "_P63_X.svg"), plot=last_plot(),  width=7.174, height=3.4)
  
  
  ggplot(subset(data, Timepoint == "P30"), aes(X, Y)) +
    xlim(0, 1100) + xlab("X (um)") +
    ylim(-100, 1300) + ylab("Y (um)") +
    geom_point(color = "black", size = 1) +
    geom_density_2d_filled(contour_var = "count", alpha=0.85, bins = 10) +
    facet_grid(. ~ factor(Group, levels=c("WT", "SOD1"))) +
    theme_few(base_family = "Arial") +
    theme(legend.position = "none",
          axis.title.y = element_text(color = "black", size = 14), axis.text.y = element_text(color = "black", size = 14),
          axis.title.x = element_text(color = "black", size = 14), axis.text.x = element_text(color = "black", size = 14),
          axis.ticks.y = element_line(color = "black"), axis.ticks.x = element_line(color = "black"),
          panel.border = element_rect(color = "black", size = 1), strip.text.x = element_text(color = "black", size = 18))
  ggsave(file=paste0("20230111_figures/P30_counts/", names(pops)[idx], "_P30_counts.svg"), plot=last_plot(), width=13.861, height=9.19)
  
  ggplot(subset(data, Timepoint == "P30"), aes(y=Y, color=Group)) +
    scale_color_manual(values = c("WT" = "#A0A0A4", "SOD1" = "#901A1E")) +
    ylim(-100, 1300) + ylab("Y (um)") + xlab("Count") +
    geom_density(size = 1, aes(x = ..count..)) +
    theme_few(base_family = "Arial") +
    theme(legend.position = "none",
          axis.title.y = element_blank(), axis.text.y = element_blank(), axis.ticks.y = element_blank(),
          axis.title.x = element_text(color = "black", size = 14), axis.text.x = element_text(color = "black", size = 14),
          axis.ticks.x = element_line(color = "black"), panel.border = element_rect(color = "black", size = 1))
  ggsave(file=paste0("20230111_figures/P30_XY/", names(pops)[idx], "_P30_Y.svg"), plot=last_plot(), width=2.97, height=8.825)
  
  ggplot(subset(data, Timepoint == "P30"), aes(x=X, color=Group)) +
    scale_color_manual(values = c("WT" = "#A0A0A4", "SOD1" = "#901A1E")) +
    xlim(0, 1100) + xlab("X (um)") + ylab("Count") +
    geom_density(size=1, aes(y = ..count..)) +
    theme_few(base_family = "Arial") +
    theme(legend.position = "none",
          axis.title.y = element_text(color = "black", size = 14), axis.text.y = element_text(color = "black", size = 14),
          axis.title.x = element_text(color = "black", size = 14), axis.text.x = element_text(color = "black", size = 14),
          axis.ticks.y = element_line(color = "black"), axis.ticks.x = element_line(color = "black"),
          panel.border = element_rect(color = "black", size = 1))
  ggsave(file=paste0("20230111_figures/P30_XY/", names(pops)[idx], "_P30_X.svg"), plot=last_plot(),  width=7.174, height=3.4)
  
}




#############################
####   QUANTIFICATIONS   ####
#############################

## General statistics

NT_thr %>% group_by(Timepoint, Group, Animal, Hemicord, Scene) %>%
  summarise(All.cells = n(),
            Area = mean(Area_um2),
  ) -> Total.cells

Total.cells <- Total.cells %>% arrange(match(Timepoint, c("P30", "P63", "P112")), desc(Group))
write_xlsx(Total.cells, "Total.cells_20221101.xlsx")


## Inhibitory, excitatory & individual markers

NT_thr %>% group_by(Timepoint, Group, Animal, Hemicord, Scene) %>%
  summarise(All.cells = n(),
            GlyT2 = sum(GlyT2_Intensity.mean >= GlyT2_low),
            Gad65 = sum(Gad65_Intensity.mean >= Gad65_low & Gad65_Intensity.std >= Gad65_SD),
            Gad67 = sum(Gad67_Intensity.mean >= Gad67_low & Gad67_Intensity.std >= Gad67_SD),
            Inhibitory = sum(GlyT2_Intensity.mean >= GlyT2_low | (Gad65_Intensity.mean >= Gad65_low & Gad65_Intensity.std >= Gad65_SD) | (Gad67_Intensity.mean >= Gad67_low & Gad67_Intensity.std >= Gad67_SD)),
            Vglut2 = sum(Vglut2_Intensity.mean >= Vglut2_low),
            Excitatory = sum(Vglut2_Intensity.mean >= Vglut2_low),
            ChAT = sum(ChAT_Intensity.mean >= ChAT_low & ChAT_Intensity.mean < ChAT_high & ChAT_Intensity.std >= ChAT_SD),
            MNs = sum(ChAT_Intensity.mean >= ChAT_low & ChAT_Intensity.mean < ChAT_high & ChAT_Intensity.std >= ChAT_SD & Y <= 250),
            En1 = sum(En1_Intensity.mean < En1_high & En1_Intensity.std >= En1_SD),
            Foxp2 = sum(Foxp2_Intensity.mean >= Foxp2_low),
            Pou6f2 = sum(Pou6f2_Intensity.mean >= Pou6f2_low),
            Sp8 = sum(Sp8_Intensity.mean >= Sp8_low & Sp8_Intensity.mean < Sp8_high),
            Calb1 = sum(Calb1_Intensity.mean >= Calb1_low & Calb1_Intensity.mean < Calb1_high & Calb1_Intensity.std >= Calb1_SD),
            Calb2 = sum(Calb2_Intensity.mean >= Calb2_low & Calb2_Intensity.std >= Calb2_SD),
            Chx10 = sum(Chx10_Intensity.mean >= Chx10_low & Chx10_Intensity.mean < Chx10_high & Chx10_Intensity.std >= Chx10_SD),
            Shox2 = sum(Shox2_Intensity.mean >= Shox2_low),
            Pitx2 = sum(Ptx2_Intensity.mean >= Ptx2_low & Ptx2_Intensity.mean < Ptx2_high),
  ) -> Quants.individual

Quants.individual <- Quants.individual %>% arrange(match(Timepoint, c("P30", "P63", "P112")), desc(Group))
write_xlsx(Quants.individual, "Quants_individual_20221103.xlsx")


## V1

V1 %>% group_by(Timepoint, Group, Animal, Hemicord, Scene) %>%
  summarise(V1 = n(),
            V1_Foxp2 = sum(Foxp2_Intensity.mean >= Foxp2_low),
            V1_Pou6f2 = sum(Pou6f2_Intensity.mean >= Pou6f2_low),
            V1_Sp8 = sum(Sp8_Intensity.mean >= Sp8_low & Sp8_Intensity.mean < Sp8_high),
            V1_Renshaw = sum(Calb1_Intensity.mean >= Calb1_low & Calb1_Intensity.mean < Calb1_high & Calb1_Intensity.std >= Calb1_SD & Y <= 250),
            V1_Ia = sum(Calb2_Intensity.mean >= Calb2_low & Calb2_Intensity.std >= Calb2_SD & Y <= 500 & X >= 500)
  ) -> Quants.V1

Quants.V1 <- Quants.V1 %>% arrange(match(Timepoint, c("P30", "P63", "P112")), desc(Group))
write_xlsx(Quants.V1, "Quants_V1_20221107.xlsx")


## V2a

V2a %>% group_by(Timepoint, Group, Animal, Hemicord, Scene) %>%
  summarise(V2a = n()) -> Quants.V2a

Quants.V2a <- Quants.V2a %>% arrange(match(Timepoint, c("P30", "P63", "P112")), desc(Group))
write_xlsx(Quants.V2a, "Quants_V2a_20221101.xlsx")


## Shox2

Shox2 %>% group_by(Timepoint, Group, Animal, Hemicord, Scene) %>%
  summarise(Shox2 = n(),
            Shox2_v2a = sum(Chx10_Intensity.mean >= Chx10_low & Chx10_Intensity.mean < Chx10_high & Chx10_Intensity.std >= Chx10_SD),
            Shox2_nonV2a = sum(Chx10_Intensity.mean < Chx10_low | Chx10_Intensity.mean >= Chx10_high | Chx10_Intensity.std < Chx10_SD)
  ) -> Quants.Shox2

Quants.Shox2 <- Quants.Shox2 %>% arrange(match(Timepoint, c("P30", "P63", "P112")), desc(Group))
write_xlsx(Quants.Shox2, "Quants_Shox2_20221101.xlsx")


## Pitx2

Pitx2 %>% group_by(Timepoint, Group, Animal, Hemicord, Scene) %>%
  summarise(Pitx2 = n(),
            V0_cg = sum(Y >= 400 & Y <= 650 & X <= 200),
            V0_c = sum(ChAT_Intensity.mean >= ChAT_low & ChAT_Intensity.mean < ChAT_high & ChAT_Intensity.std >= ChAT_SD & Y >= 400 & Y <= 650 & X <= 200),
            V0_g = sum(Vglut2_Intensity.mean >= Vglut2_low & Y >= 400 & Y <= 650 & X <= 200)
  ) -> Quants.Pitx2

Quants.Pitx2 <- Quants.Pitx2 %>% arrange(match(Timepoint, c("P30", "P63", "P112")), desc(Group))
write_xlsx(Quants.Pitx2, "Quants_Pitx2_20221101.xlsx")










