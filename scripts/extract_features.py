import numpy as np
import math
import os
from scipy.misc import imread
# In order to import caffe, one may have to add caffe in the PYTHONPATH
import caffe


# If using GPU, set to True
GPU = False
if GPU:
  caffe.set_mode_gpu()
  caffe.set_device(0)
else:
  caffe.set_mode_cpu()


def get_features(net, locs_file):
  '''
  Runs the forward pass of the neural net on every image.
  '''
  img_cluster_locs, num_images = get_locs_info(locs_file)
  num_batches = math.ceil(num_images / 32.0)
  raw_features = []
  batch_num = 0
  with open(locs_file, 'r') as f:
    curr_batch = []
    for line in f:
      img_path = line.split()[0]
      # reads a RGB image
      input_img = imread(img_path).astype(np.float32)
      # convert to BGR
      input_img = input_img[:, :, [2, 1, 0]]
      # convert to D,H,W
      input_img = np.transpose(input_img, [2, 0, 1])
      # subtract the mean
      mean_bgr = [103.334, 107.8797, 107.4072]
      for i in xrange(0, 3):
        input_img[i, :, :] = input_img[i, :, :] - mean_bgr[i]
      curr_batch.append(input_img)

      if len(curr_batch) == 32:
        batch_num += 1
        print("Batch %d/%d for %s" % (batch_num,
                                      num_batches, locs_file))
        curr_batch = np.asarray(curr_batch)
        net.blobs['data'].data[...] = curr_batch
        net.forward()
        raw_features.append(net.blobs['conv7'].data)
        curr_batch = []
    if len(curr_batch) > 0:
      batch_num += 1
      print("Batch %d/%d for %s" % (batch_num, num_batches, locs_file))
      curr_batch = np.asarray(curr_batch)
      batch_size = curr_batch.shape[0]
      # pad end batch
      curr_batch = np.vstack((curr_batch, np.zeros((32 - batch_size, 3, 400, 400)).astype(np.float32)))
      net.blobs['data'].data[...] = curr_batch
      net.forward()
      raw_features.append(net.blobs['conv7'].data[:batch_size])
  raw_features = np.vstack(raw_features)
  # average pooling
  n, f, h, w = raw_features.shape
  features = raw_features.reshape(n, f, h*w)
  features = np.mean(features, axis=2)
  return features, img_cluster_locs


def aggregate_features(features, img_cluster_locs, clusters):
  '''
  Aggregate features by cluster by taking the mean.
  Respects the cluster ordering given by lats.npy and lons.npy.
  '''
  # average the features in the same cluster
  conv_features = []
  image_counts = []
  for cluster in clusters:
    cluster_mask = [(img_cluster == cluster) for img_cluster in img_cluster_locs]
    cluster_mask = np.asarray(cluster_mask)
    image_count = np.sum(cluster_mask)
    # if count is 0, fill with a 0 feature
    if image_count == 0:
      mean_cluster_feature = np.zeros(features.shape[1])
    else:
      mean_cluster_feature = np.mean(features[cluster_mask], axis=0)
    conv_features.append(mean_cluster_feature)
    image_counts.append(image_count)
  conv_features = np.asarray(conv_features)
  image_counts = np.asarray(image_counts)
  return conv_features, image_counts


def extract(net, countries, output_dir):
  '''
  Runs the forward pass of the CNN on every image and then
  aggregates the features by cluster by taking the mean.
  '''
  for country in countries:
    print("Extracting %s for %s" % (country, output_dir))
    locs_file = os.path.join(output_dir, country, 'downloaded_locs.txt')

    # compute conv features for every image
    features, img_cluster_locs = get_features(net, locs_file)

    # get the master cluster ordering
    cluster_lats = np.load(os.path.join(output_dir, country, 'lats.npy'))
    cluster_lons = np.load(os.path.join(output_dir, country, 'lons.npy'))
    # bit of a hack here - cluster locations can be changed when
    # writing to a file using format string
    clusters = [(float("%f" % cluster_lats[i]), float("%f" % cluster_lons[i])) for i in xrange(cluster_lats.size)]

    # aggregate features by cluster
    conv_features, image_counts = aggregate_features(features, img_cluster_locs, clusters)

    conv_features_path = os.path.join(output_dir, country, 'conv_features')
    image_counts_path = os.path.join(output_dir, country, 'image_counts')
    np.save(conv_features_path, conv_features)
    np.save(image_counts_path, image_counts)


def get_locs_info(locs_file):
  '''
  Get the cluster location for each image and compute the number of
  images.
  '''
  img_cluster_locs = []
  num_images = 0
  with open(locs_file, 'r') as f:
    for line in f:
      num_images += 1
      img_path, lat, lon, cluster_lat, cluster_lon = line.split()
      cluster_loc = (float(cluster_lat), float(cluster_lon))
      img_cluster_locs.append(cluster_loc)
  return img_cluster_locs, num_images


if __name__ == '__main__':
  model_file = '../model/predicting_poverty_deploy.prototxt'
  weights_file = '../model/predicting_poverty_trained.caffemodel'
  net = caffe.Net(model_file, weights_file, caffe.TEST)

  # DHS
  print("Extracting features for DHS")
  countries = ['nigeria', 'tanzania', 'uganda', 'malawi', 'rwanda']
  dhs_dir = '../data/output/DHS'
  extract(net, countries, dhs_dir)

  # LSMS
  print("Extracting features for LSMS")
  countries = ['nigeria', 'tanzania', 'uganda', 'malawi']
  lsms_dir = '../data/output/LSMS'
  extract(net, countries, lsms_dir)
