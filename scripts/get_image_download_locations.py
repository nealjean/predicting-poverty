import os
import numpy as np
from osgeo import gdal, osr

'''
Generates the image locations to download for a set of cluster locations.
The image locations are taken from a 10x10km area around the cluster
location. For every country except Nigeria and Tanzania, we take 100
1x1km images. For Nigeria and Tanzania, we take 25 1kmx1km images per
cluster.
'''

def get_download_locations(countries, cluster_loc_dir, nightlight_tif):
  for country in countries:
    cluster_lats = np.load(os.path.join(cluster_loc_dir,
                                        country, 'lats.npy'))
    cluster_lons = np.load(os.path.join(cluster_loc_dir,
                                        country, 'lons.npy'))
    clusters = [(cluster_lats[i], cluster_lons[i])
                for i in xrange(cluster_lats.size)]
    top_lefts = [(lat + 0.045, lon - 0.045) for lat, lon in clusters]
    bottom_rights = [(lat - 0.045, lon + 0.045) for lat, lon in clusters]
    top_left_pixellocs = locsToPixels(nightlight_tif, top_lefts)
    bottom_right_pixellocs = locsToPixels(nightlight_tif, bottom_rights)
    output_cluster_locs = []
    output_pix = []
    for i in xrange(len(clusters)):
      top_left = top_left_pixellocs[i]
      bottom_right = bottom_right_pixellocs[i]
      for x in xrange(top_left[0], bottom_right[0]):
        for y in xrange(top_left[1], bottom_right[1]):
          if country == 'nigeria' or country == 'tanzania':
            if x % 2 == 1 or y % 2 == 1:
              continue
          output_pix.append((x, y))
          output_cluster_locs.append(clusters[i])
    output_locs = pixelsToCoords(nightlight_tif, output_pix)
    print("%d locations saved for %s" % (len(output_locs), country))
    with open(os.path.join(cluster_loc_dir, country, 'candidate_download_locs.txt'), 'w') as f:
      for loc, cluster_loc in zip(output_locs, output_cluster_locs):
        f.write("%f %f %f %f\n" % (loc[0], loc[1], cluster_loc[0], cluster_loc[1]))


def latLon2Pixel(lat, lon, ct, gt):
  (lon, lat, holder) = ct.TransformPoint(lon, lat)
  x = (lon-gt[0])/gt[1]
  y = (lat-gt[3])/gt[5]
  return (int(x), int(y))


def locsToPixels(srcAddr, latLonPairs):
  ds = gdal.Open(srcAddr)
  gt = ds.GetGeoTransform()
  srs = osr.SpatialReference()
  srs.ImportFromWkt(ds.GetProjection())
  srsLatLong = srs.CloneGeogCS()
  ct = osr.CoordinateTransformation(srsLatLong, srs)
  pixelPairs = []
  for point in latLonPairs:
    lat, lon = point
    pixelPairs.append(latLon2Pixel(lat, lon, ct, gt))
  return pixelPairs


def pixel2coord(x, y, xoff, a, b, yoff, d, e):
  xp = a * x + b * y + xoff
  yp = d * x + e * y + yoff
  return (yp, xp)


def pixelsToCoords(srcAddr, pixelPairs):
  ds = gdal.Open(srcAddr)
  xoff, a, b, yoff, d, e = ds.GetGeoTransform()
  latLonPairs = []
  for pixel in pixelPairs:
    x, y = pixel
    latLonPairs.append(pixel2coord(x + 0.5, y + 0.5, xoff, a, b, yoff, d, e))
  return latLonPairs


if __name__ == '__main__':
  nightlight_tif = "../data/input/Nightlights/2013/F182013.v4c_web.stable_lights.avg_vis.tif"

  # DHS
  print("Generating candidate image locations for DHS")
  countries = ['nigeria', 'tanzania', 'uganda', 'malawi', 'rwanda']
  cluster_loc_dir = '../data/output/DHS'
  get_download_locations(countries, cluster_loc_dir, nightlight_tif)

  # LSMS
  print("Generating candidate image locations for LSMS")
  countries = ['nigeria', 'tanzania', 'uganda', 'malawi']
  cluster_loc_dir = '../data/output/LSMS'
  get_download_locations(countries, cluster_loc_dir, nightlight_tif)
