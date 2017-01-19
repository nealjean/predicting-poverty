## PROCESS LSMS AND DHS DATA ##

#### Preliminaries: Load packages, create new folders, define aggregation functions ####
setwd('predicting-poverty') # Set working directory to where you downloaded the replication folder
rm(list=ls())
library(magrittr)
library(foreign)
library(raster)
extract <- raster::extract # Ensure the 'magrittr' package does not mask the 'raster' package's 'extract' function
library(readstata13) # One Tanzanian LSMS .dta file was saved in Stata-13 format
library(plyr)
'%&%' <- function(x,y)paste0(x,y)


dir.create('data/output/LSMS', showWarnings = F)
dir.create('data/output/DHS', showWarnings = F)

# Assign each cluster the mean nightlights values over a 10 km^2 area centered on its provided coordinates
nl <- function(df, year){
  # ls.filter identifies the nine clusters we filtered out because of LandScan data availability in our analysis
  ls.filter <- c(0.112190, -1.542321, -1.629748, -1.741995, -1.846039, -1.896059, -2.371342, -2.385341, -2.446988)
  nl <- raster(paste0('data/input/Nightlights/', year, '/', list.files(paste0('data/input/Nightlights/', year))))
  df2 <- subset(df, is.na(lat)==F & is.na(lon)==F & lat !=0 & lon != 0)
  df2 <- unique(df2[,c('lat', 'lon')])
  shape <- extent(c(range(c(df2$lon-0.5, df2$lon+0.5)),
                    range(c(df2$lat-0.5, df2$lat+0.5))))
  nl <- crop(nl, shape)
  for (i in 1:nrow(df2)){
    lat <- sort(c(df2$lat[i] - (180/pi)*(5000/6378137), df2$lat[i] + (180/pi)*(5000/6378137)))
    lon <- sort(c(df2$lon[i] - (180/pi)*(5000/6378137)/cos(df2$lat[i]), df2$lon[i] + (180/pi)*(5000/6378137)/cos(df2$lat[i])))
    ext <- extent(lon, lat)
    nl.vals <- unlist(extract(nl, ext))
    nl.vals[nl.vals==255] <- NULL
    df2$nl[i] <- mean(nl.vals, na.rm = T)
    # Add a column to indicate whether cluster was one of the nine filtered out by LandScan data availability for our study
    # This allows full replication of our data by subsetting survey data to sample == 1 as well as testing on the full survey sample
    # Ultimately, our sample differs from the full sample by just one cluster in Uganda and one in Tanzania
    df2$sample[i] <- if (round(df2$lat[i], 6) %in% ls.filter) 0 else 1
  }
  df <- merge(na.omit(df2), df, by = c('lat', 'lon'))
  return(df)
}

# Aggregate household-level data to cluster level
cluster <- function(df, dhs = F){
  # Record how many households comprise each cluster
  for (i in 1:nrow(df)){
    sub <- subset(df, lat == df$lat[i] & lon == df$lon[i])
    df$n[i] <- nrow(sub)
  }
  # Clustering for LSMS survey data
  df <- if (dhs == FALSE)
    ddply(df, .(lat, lon), summarise,
          cons = mean(cons),
          nl = mean(nl),
          n = mean(n),
          sample = min(sample))
  # Clustering for DHS survey data
  else ddply(df, .(lat, lon), summarise,
             wealthscore = mean(wealthscore),
             nl = mean(nl),
             n = mean(n),
             sample = min(sample))
  return(df)
}

#### Write LSMS Data ####

## Uganda ##
uga12.cons <- read.dta('data/input/LSMS/UGA_2011_UNPS_v01_M_STATA/UNPS 2011-12 Consumption Aggregate.dta') %$%
  data.frame(hhid = HHID, cons = welfare*118.69/(30*946.89*mean(c(66.68, 71.55))))
uga12.geo <- read.dta('data/input/LSMS/UGA_2011_UNPS_v01_M_STATA/UNPS_Geovars_1112.dta')
uga12.coords <- data.frame(hhid = uga12.geo$HHID, lat = uga12.geo$lat_mod, lon = uga12.geo$lon_mod)
uga12.rururb <- data.frame(hhid = uga12.geo$HHID, rururb = uga12.geo$urban, stringsAsFactors = F)
uga12.weight <- read.dta('data/input/LSMS/UGA_2011_UNPS_v01_M_STATA/GSEC1.dta')[,c('HHID', 'mult')]
names(uga12.weight) <- c('hhid', 'weight')
uga12.hh9 <- read.dta('data/input/LSMS/UGA_2011_UNPS_v01_M_STATA/GSEC9A.dta')
uga12.room <- data.frame(hhid = uga12.hh9$HHID, room = uga12.hh9$h9q3)
uga12.metal <- data.frame(hhid = uga12.hh9$HHID, roof = uga12.hh9$h9q4=='Iron sheets')

uga12.vars <- list(uga12.cons, uga12.rururb, uga12.coords, uga12.weight, uga12.room, uga12.metal) %>%
  Reduce(function(x,y) merge(x, y, by = 'hhid'), .) %>%
  nl(2012)

write.table(uga12.vars, 'data/output/LSMS/Uganda 2012 LSMS (Household).txt', row.names = F)
write.table(cluster(uga12.vars), 'data/output/LSMS/Uganda 2012 LSMS (Cluster).txt', row.names = F)

## Tanzania ##
tza13.cons <- read.dta('data/input/LSMS/TZA_2012_LSMS_v01_M_STATA_English_labels/ConsumptionNPS3.dta') %$%
  data.frame(hhid = y3_hhid, cons = expmR/(365*adulteq))
tza13.cons$cons <- tza13.cons$cons*112.69/(585.52*mean(c(130.72,141.01)))
tza13.geo <- read.dta13('data/input/LSMS/TZA_2012_LSMS_v01_M_STATA_English_labels/HouseholdGeovars_Y3.dta')
tza13.coords <- data.frame(hhid = tza13.geo$y3_hhid, lat = tza13.geo$lat_dd_mod, lon = tza13.geo$lon_dd_mod)
tza13.hha <- read.dta('data/input/LSMS/TZA_2012_LSMS_v01_M_STATA_English_labels/HH_SEC_A.dta')
tza13.rururb <- data.frame(hhid = tza13.hha$y3_hhid, rururb = tza13.hha$y3_rural, stringsAsFactors = F)
tza13.weight <- read.dta('data/input/LSMS/TZA_2012_LSMS_v01_M_STATA_English_labels/HH_SEC_A.dta')[,c('y3_hhid', 'y3_weight')]
names(tza13.weight) <- c('hhid', 'weight')
tza13.hhi <- read.dta('data/input/LSMS/TZA_2012_LSMS_v01_M_STATA_English_labels/HH_SEC_I.dta')
tza13.room <- na.omit(data.frame(hhid = tza13.hhi$y3_hhid, room = tza13.hhi$hh_i07_1))
tza13.metal <- data.frame(hhid = tza13.hhi$y3_hhid, metal = tza13.hhi$hh_i09=='METAL SHEETS (GCI)')

tza13.vars <- list(tza13.cons, tza13.coords, tza13.rururb, tza13.weight, tza13.room, tza13.metal) %>%
  Reduce(function(x, y) merge(x, y, by = 'hhid'), .) %>%
  nl(2013)

write.table(tza13.vars, 'data/output/LSMS/Tanzania 2013 LSMS (Household).txt', row.names = F)
write.table(cluster(tza13.vars), 'data/output/LSMS/Tanzania 2013 LSMS (Cluster).txt', row.names = F)

## Malawi ##
mwi13.cons <- read.dta('data/input/LSMS/MWI_2013_IHPS_v01_M_STATA/Round 2 (2013) Consumption Aggregate.dta') %$%
  data.frame(hhid = y2_hhid, cons = rexpagg/(365*adulteq), weight = hhweight)
mwi13.cons$cons <- mwi13.cons$cons*107.62/(116.28*166.12)
mwi13.geo <- read.dta('data/input/LSMS/MWI_2013_IHPS_v01_M_STATA/Geovariables/HouseholdGeovariables_IHPS.dta')
mwi13.coords <- data.frame(hhid = mwi13.geo$y2_hhid, lat = mwi13.geo$LAT_DD_MOD, lon = mwi13.geo$LON_DD_MOD)
mwi13.hha <- read.dta('data/input/LSMS/MWI_2013_IHPS_v01_M_STATA/Household/HH_MOD_A_FILT.dta')
mwi13.rururb <- data.frame(hhid = mwi13.hha$y2_hhid, rururb = mwi13.hha$baseline_rural, stringsAsFactors = F)
mwi13.hhf <- read.dta('data/input/LSMS/MWI_2013_IHPS_v01_M_STATA/Household/HH_MOD_F.dta')
mwi13.room <- data.frame(hhid = mwi13.hhf$y2_hhid, room = mwi13.hhf$hh_f10)
mwi13.metal <- data.frame(hhid = mwi13.hhf$y2_hhid, metal = mwi13.hhf$hh_f10=='IRON SHEETS')

mwi13.vars <- list(mwi13.cons, mwi13.coords, mwi13.rururb, mwi13.room, mwi13.metal) %>%
  Reduce(function(x, y) merge(x, y, by = 'hhid'), .) %>%
  nl(2013)

write.table(mwi13.vars, 'data/output/LSMS/Malawi 2013 LSMS (Household).txt', row.names = F)
write.table(cluster(mwi13.vars), 'data/output/LSMS/Malawi 2013 LSMS (Cluster).txt', row.names = F)

## Nigeria ##
nga13.cons <- read.dta('data/input/LSMS/DATA/cons_agg_w2.dta') %$%
  data.frame(hhid = hhid, cons = pcexp_dr_w2/365)
nga13.cons$cons <- nga13.cons$cons*110.84/(79.53*100)
nga13.geo <- read.dta('data/input/LSMS/DATA/Geodata Wave 2/NGA_HouseholdGeovars_Y2.dta')
nga13.coords <- data.frame(hhid = nga13.geo$hhid, lat = nga13.geo$LAT_DD_MOD, lon = nga13.geo$LON_DD_MOD)
nga13.rururb <- data.frame(hhid = nga13.geo$hhid, rururb = nga13.geo$sector, stringsAsFactors = F)
nga13.weight <- read.dta('data/input/LSMS/DATA/HHTrack.dta')[,c('hhid', 'wt_wave2')]
names(nga13.weight)[2] <- 'weight'
nga13.phhh8 <- read.dta('data/input/LSMS/DATA/Post Harvest Wave 2/Household/sect8_harvestw2.dta')
nga13.room <- data.frame(hhid = nga13.phhh8$hhid, room = nga13.phhh8$s8q9)
nga13.metal <- data.frame(hhid = nga13.phhh8$hhid, metal = nga13.phhh8$s8q7=='IRON SHEETS')

nga13.vars <- list(nga13.cons, nga13.coords, nga13.rururb, nga13.weight, nga13.room, nga13.metal) %>%
  Reduce(function(x, y) merge(x, y, by = 'hhid'), .) %>%
  nl(2013)

write.table(nga13.vars, 'data/output/LSMS/Nigeria 2013 LSMS (Household).txt', row.names = F)
write.table(cluster(nga13.vars), 'data/output/LSMS/Nigeria 2013 LSMS (Cluster).txt', row.names = F)

#### Write DHS Data ####
path <- function(iso){
  return(paste0('data/input/DHS/',list.files('data/input/DHS')[substr(list.files('data/input/DHS'),1,2)==iso], '/'))
}

vars <- c('001', '005', 271)
vars <- c('hhid', paste0('hv', vars))
names <- c('hhid', 'cluster', 'weight', 'wealthscore')

# Uganda 2011
uga11.dhs <- read.dta(path('UG')%&%'ughr60dt/UGHR60FL.DTA', convert.factors=NA) %>%
  subset(select = vars)
names(uga11.dhs) <- names

uga11.coords <- read.dbf(path('UG')%&%'ugge61fl/UGGE61FL.dbf')[,c('DHSCLUST', 'LATNUM', 'LONGNUM')]
names(uga11.coords) <- c('cluster', 'lat', 'lon')

uga11.dhs <- merge(uga11.dhs, uga11.coords, by = 'cluster') %>%
  nl(2011)

write.table(uga11.dhs, 'data/output/DHS/Uganda 2011 DHS (Household).txt', row.names = F)
write.table(cluster(uga11.dhs, T), 'data/output/DHS/Uganda 2011 DHS (Cluster).txt', row.names = F)

# Tanzania 2010

tza10.dhs <- read.dta(path('TZ')%&%'tzhr63dt/TZHR63FL.DTA', convert.factors = NA) %>%
  subset(select = vars)
names(tza10.dhs) <- names

tza10.coords <- read.dbf(path('TZ')%&%'tzge61fl/TZGE61FL.dbf')[,c('DHSCLUST', 'LATNUM', 'LONGNUM')]
names(tza10.coords) <- c('cluster', 'lat', 'lon')

tza10.dhs <- merge(tza10.dhs, tza10.coords, by = 'cluster') %>%
  nl(2010)

write.table(tza10.dhs, 'data/output/DHS/Tanzania 2010 DHS (Household).txt', row.names = F)
write.table(cluster(tza10.dhs, T), 'data/output/DHS/Tanzania 2010 DHS (Cluster).txt', row.names = F)

# Nigeria 2013
nga13.dhs <- read.dta(path('NG')%&%'nghr6adt/NGHR6AFL.DTA', convert.factors = NA) %>%
  subset(select = vars)
names(nga13.dhs) <- names

nga13.coords <- read.dbf(path('NG')%&%'ngge6afl/NGGE6AFL.dbf')[,c('DHSCLUST', 'LATNUM', 'LONGNUM')]
names(nga13.coords) <- c('cluster', 'lat', 'lon')

nga13.dhs <- merge(nga13.dhs, nga13.coords, by = 'cluster') %>%
  nl(2013)

write.table(nga13.dhs, 'data/output/DHS/Nigeria 2013 DHS (Household).txt', row.names = F)
write.table(cluster(nga13.dhs, T), 'data/output/DHS/Nigeria 2013 DHS (Cluster).txt', row.names = F)

# Malawi 2010
mwi10.dhs <- read.dta(path('MW')%&%'mwhr61dt/MWHR61FL.DTA', convert.factors = NA) %>%
  subset(select = vars)
names(mwi10.dhs) <- names

mwi10.coords <- read.dbf(path('MW')%&%'mwge62fl/MWGE62FL.dbf')[,c('DHSCLUST', 'LATNUM', 'LONGNUM')]
names(mwi10.coords) <- c('cluster', 'lat', 'lon')

mwi10.dhs <- merge(mwi10.dhs, mwi10.coords, by = 'cluster') %>%
  nl(2010)

write.table(mwi10.dhs, 'data/output/DHS/Malawi 2010 DHS (Household).txt', row.names = F)
write.table(cluster(mwi10.dhs, T), 'data/output/DHS/Malawi 2010 DHS (Cluster).txt', row.names = F)

# Rwanda 2010
rwa10.dhs <- read.dta(path('RW')%&%'rwhr61dt/RWHR61FL.DTA', convert.factors = NA) %>%
  subset(select = vars)
names(rwa10.dhs) <- names

rwa10.coords <- read.dbf(path('RW')%&%'rwge61fl/RWGE61FL.dbf')[,c('DHSCLUST', 'LATNUM', 'LONGNUM')]
names(rwa10.coords) <- c('cluster', 'lat', 'lon')

rwa10.dhs <- merge(rwa10.dhs, rwa10.coords, by = 'cluster') %>%
  nl(2010)

write.table(rwa10.dhs, 'data/output/DHS/Rwanda 2010 DHS (Household).txt', row.names = F)
write.table(cluster(rwa10.dhs, T), 'data/output/DHS/Rwanda 2010 DHS (Cluster).txt', row.names = F)
