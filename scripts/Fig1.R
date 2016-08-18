## PRODUCE FIGURE 1 ##

#### Preliminaries: Load packages ####

setwd('predicting-poverty') # Set working directory to where you downloaded the replication folder
rm(list=ls())
library(magrittr)
library(foreign)
library(RColorBrewer)
library(sp)
library(lattice)
library(plyr)
library(ggplot2)
library(grid)
library(gridExtra)

#### Panels A-B: Survey availability maps ####

## Consumption surveys ##
povcal <- read.table('data/input/PovcalNet Survey Availability.txt', header = T)
for (i in 1:nrow(povcal)){
  povcal$yrs[i] <- nrow(subset(povcal, iso == povcal$iso[i] & year %in% seq(2000, 2010, by = 0.1)))
}
povcal <- unique(povcal[,c('country', 'iso', 'yrs')])
povcal$yrs[povcal$iso == 'TZA'] <- povcal$yrs[povcal$iso == 'TZA']+1 # Add a Tanzanian LSMS survey that had not yet been included in PovcalNet database

## DHS ##
dhs <- read.table('data/input/DHS Survey Availability.txt', header = T)
for (i in 1:nrow(dhs)){
  dhs$dhsyrs[i] <- nrow(subset(dhs, iso == dhs$iso[i] & year %in% seq(2000, 2010, by = 0.1) & available == T & type == 'Standard'))
}
dhs <- unique(dhs[,c('iso', 'dhsyrs')])

## Maps ##
load('data/input/AfricanCountryShapefile.RData')
africa@data$plot.order = 1:nrow(africa@data)
names(africa@data)[3] <- 'iso'
# Consumption
afr.df <- subset(povcal, iso %in% africa@data$iso)[,c('iso', 'yrs')]
afr.df <- rbind(afr.df[,c('iso', 'yrs')], data.frame(iso = setdiff(africa@data$iso, afr.df$iso), yrs = 0))
africa@data <- merge(africa@data, afr.df, by = 'iso')
africa@data$yrs <- factor(africa@data$yrs)
colors <- brewer.pal(length(levels(factor(africa@data$yrs))), 'Blues')
africa@data <- africa@data[order(africa@data$plot.order),]
m1 <- spplot(africa, 'yrs', col.regions = colors,
             main = list(label = 'Consumption/income surveys', cex = 0.9),
             colorkey = list(space = 'left', height = 0.4))
args <- m1$legend$left$args$key
legendArgs <- list(fun = draw.colorkey, args = list(key = args), corner = c(0.05,0.30))
m1 <- spplot(africa, 'yrs', col.regions = colors,
             main = list(label = 'Consumption/income surveys', cex = 0.9),
             colorkey = FALSE,
             legend = list(inside = legendArgs))
key <- draw.colorkey(m1$legend[[1]]$args$key)
# Assets
dhs <- dhs[,c('iso', 'dhsyrs')]
dhs <- subset(dhs, iso %in% africa@data$iso)
dhs <- rbind(dhs, data.frame(iso = setdiff(africa@data$iso, dhs$iso), dhsyrs = 0))
africa@data <- merge(africa@data, dhs, by = 'iso')
africa@data <- africa@data[order(africa@data$plot.order),]
africa@data$dhsyrs <- factor(africa@data$dhsyrs, levels = 0:4)
m2 <- spplot(africa, 'dhsyrs', col.regions = colors,
             main = list(label = 'Asset surveys, 2000-2010', cex = 0.9),
             colorkey = FALSE,
             legend = list(inside = legendArgs))

#### Panels C-F: Nightlights vs. Consumption plots ####

# Import household-level LSMS data for use in population density graphs
# The cluster-level data.frame is used as input into the nightlights-consumption plots
nga.hh <- read.table('data/output/LSMS/Nigeria 2013 LSMS (Household).txt', header = T) %>%
  subset(is.na(weight)==F)
nga.hh$weight <- nga.hh$weight/sum(nga.hh$weight)
nga <- read.table('data/output/LSMS/Nigeria 2013 LSMS (Cluster).txt', header = T)

tza.hh <- read.table('data/output/LSMS/Tanzania 2013 LSMS (Household).txt', header = T) %>%
  subset(is.na(weight)==F)
tza.hh$weight <- tza.hh$weight/sum(tza.hh$weight)
tza <- read.table('data/output/LSMS/Tanzania 2013 LSMS (Cluster).txt', header = T)

uga.hh <- read.table('data/output/LSMS/Uganda 2012 LSMS (Household).txt', header = T) %>%
  subset(is.na(weight)==F)
uga.hh$weight <- uga.hh$weight/sum(uga.hh$weight)
uga <- read.table('data/output/LSMS/Uganda 2012 LSMS (Cluster).txt', header = T)

mwi.hh <- read.table('data/output/LSMS/Malawi 2013 LSMS (Household).txt', header = T) %>%
  subset(is.na(weight)==F)
mwi.hh$weight <- mwi.hh$weight/sum(mwi.hh$weight)
mwi <- read.table('data/output/LSMS/Malawi 2013 LSMS (Cluster).txt', header = T)

# Paper figures also subset these cluster coordinates to areas where we have sufficient satellite imagery
# The effect on the final plot is insignificant since imagery availability affects relatively few clusters

#### Producing the final figure ####

pooled <- list(nga, tza, uga, mwi) %>%
  do.call('rbind', .) %>%
  subset(n > 1)
margin <- range(pooled$cons, na.rm = T)

# Function to combine cluster-level nightlights-consumption plots and household-level consumption density plots into one panel
twoplots <- function(df, title){
  breaks <- seq(5, 20, 5) # Choose consumption tick labels
  iso <- substr(deparse(substitute(df)), 1, 3)
  label <- 'Nightlight intensity'
  hh <- get(paste0(iso,'.hh')) # Pull household-level data.frame
  
  p1 <- ggplot(df, aes(x = cons)) +
    theme_bw() +
    geom_point(aes(y = nl), color = colors[4], alpha = 0.4) +
    geom_smooth(aes(y = nl)) +
    scale_x_continuous(trans = 'log', breaks = breaks, limits = margin, minor_breaks = NULL) +
    coord_cartesian(ylim = c(0,63)) +
    geom_vline(xintercept = 1.9, color = 'red') +
    ggtitle(title) +
    labs(y = label, x = element_blank()) +
    theme(plot.margin = unit(c(5,5,-2,5), units = 'points'),
          axis.text.x = element_blank(), axis.line = element_blank(),
          panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
          plot.title = element_text(face = 'bold', size = 16),
          axis.title = element_text(size=16))
  
  p2 <- ggplot(hh, aes(x = cons)) +
    geom_density(fill = 'darkgray', alpha = 0.8, adjust = 2, aes(weight = weight)) +
    labs(x = paste('Daily Consumption (2011 USD)'), y = 'Population') +
    theme_bw() +
    scale_y_continuous(breaks = NULL, expand = c(0.1, 0)) +
    scale_x_continuous(trans = 'log', breaks = breaks, limits = margin, minor_breaks = NULL) +
    coord_cartesian(xlim = margin) +
    geom_vline(xintercept = 1.9, color = 'red') +
    theme(plot.margin = unit(c(0,5,5,5), units = 'points'),
          panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
          axis.line.y = element_blank(), axis.text.y = element_blank(),
          axis.title = element_text(size=16))
  
  p1 <- ggplot_gtable(ggplot_build(p1))
  p2 <- ggplot_gtable(ggplot_build(p2))
  maxWidth = unit.pmax(p1$widths[2:3], p2$widths[2:3])
  p1$widths[2:3] <- maxWidth
  p2$widths <- p1$widths
  grid.arrange(p1, p2, heights = c(2/3, 1/3))
}

graphs <- list(m1, m2, twoplots(nga, 'Nigeria, 2012'), twoplots(tza, 'Tanzania, 2012'), twoplots(uga, 'Uganda, 2011'), twoplots(mwi, 'Malawi, 2013'))
pdf(file = 'figures/Figure 1.pdf', width = 8, height = 12)
grid.arrange(grobs = graphs, ncol = 2)
dev.off()
