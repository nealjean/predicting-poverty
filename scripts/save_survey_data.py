import numpy as np
import pandas as pd
import os


def retrieve_and_save(countries, fns, out_dir, names, keys, sample=True):
  for idx, country in enumerate(countries):
    df = pd.read_csv(fns[idx], sep=' ')
    if sample:
      df = df[df["sample"]==1]
    df = df[(df.lat!=0) & (df.lon!=0)]
    for name, key in zip(names, keys):
      if not os.path.exists(os.path.join(out_dir, country)):
        os.makedirs(os.path.join(out_dir, country))
      np.save(os.path.join(out_dir, country, name), df[key])
    if idx == 0:
      pooled = df.copy()
    else:
      pooled = pooled.append(df)

  for name, key in zip(names, keys):
    if not os.path.exists(os.path.join(out_dir, 'pooled')):
        os.makedirs(os.path.join(out_dir, 'pooled'))
    np.save(os.path.join(out_dir, 'pooled', name), pooled[key])


if __name__ == '__main__':
  '''
  The set of samples used in the paper was not quite the full set due to missing Landscan data (1 less cluster in Uganda LSMS, 8 less clusters in Tanzania LSMS). Set this variable to True to use the same set of household clusters, set to False to use the full set of household clusters.
  '''
  sample = True

  ############################
  ############ DHS ###########
  ############################

  countries = ['nigeria', 'tanzania', 'uganda', 'malawi', 'rwanda']
  fns = ['../data/output/DHS/Nigeria 2013 DHS (Cluster).txt',
         '../data/output/DHS/Tanzania 2010 DHS (Cluster).txt',
         '../data/output/DHS/Uganda 2011 DHS (Cluster).txt',
         '../data/output/DHS/Malawi 2010 DHS (Cluster).txt',
         '../data/output/DHS/Rwanda 2010 DHS (Cluster).txt']
  out_dir = '../data/output/DHS/'
  names = ['lats', 'lons', 'assets', 'nightlights', 'households']
  keys = ['lat', 'lon', 'wealthscore', 'nl', 'n']
  retrieve_and_save(countries, fns, out_dir, names, keys, sample=sample)

  ############################
  ############ LSMS ##########
  ############################

  countries = ['nigeria', 'tanzania', 'uganda', 'malawi']
  fns = ['../data/output/LSMS/Nigeria 2013 LSMS (Cluster).txt',
         '../data/output/LSMS/Tanzania 2013 LSMS (Cluster).txt',
         '../data/output/LSMS/Uganda 2012 LSMS (Cluster).txt',
         '../data/output/LSMS/Malawi 2013 LSMS (Cluster).txt']
  out_dir = '../data/output/LSMS/'
  names = ['lats', 'lons', 'consumptions', 'nightlights', 'households']
  keys = ['lat', 'lon', 'cons', 'nl', 'n']
  retrieve_and_save(countries, fns, out_dir, names, keys, sample=sample)
