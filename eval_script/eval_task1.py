from __future__ import division
import os
from os.path import join, exists, basename, splitext
import re
import sys
import shapely
from shapely.geometry import Polygon
from shapely.geometry import MultiPoint
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import glob
import json
from tqdm import tqdm
import pickle

IOU_THRESH = 0.5
N_TEST = 4229


def polygon_from_str(line):
  """
  Create a shapely polygon object from gt or dt line.
  """
  polygon_points = [float(o) for o in line.split(',')[:8]]
  polygon_points = np.array(polygon_points).reshape(4, 2)
  polygon = Polygon(polygon_points).convex_hull
  return polygon


def polygon_iou(poly1, poly2):
  """
  Intersection over union between two shapely polygons.
  """
  if not poly1.intersects(poly2): # this test is fast and can accelerate calculation
    iou = 0
  else:
    try:
      inter_area = poly1.intersection(poly2).area
      union_area = poly1.area + poly2.area - inter_area
      # union_area = poly1.union(poly2).area
      iou = float(inter_area) / union_area
    except shapely.geos.TopologicalError:
      print('shapely.geos.TopologicalError occured, iou set to 0')
      iou = 0
  return iou


def det_eval(gt_dir, dt_dir, save_dir):
  """
  Evaluation detection by calculating the maximum f-measure across all thresholds.
  ARGS
    gt_dir: the directory of groundtruth files
    dt_dir: the directory of detection results files
    save_dir: the directory for saving evaluation results
  RETURN
    nothing returned
  """
  # load all groundtruths into a dict of {<image-name>: <list-of-polygons>}
  n_gt = 0
  all_gt = {}
  gt_files = glob.glob(join(gt_dir, 'image_*.txt'))
  assert(len(gt_files) == N_TEST)
  print('Number of GT files: %d' % len(gt_files))
  for gt_file in gt_files:
    with open(gt_file, 'r') as f:
      gt_lines = f.readlines()
      polygons = [polygon_from_str(o) for o in gt_lines]
      n_gt += len(polygons)
    fname = splitext(basename(gt_file))[0]
    all_gt[fname] = polygons

  # scores and match status of all dts in a single list
  all_dt_match = []
  all_dt_scores = []

  # for every detection, calculate its match to groundtruth
  dt_files = glob.glob(join(dt_dir, '*.txt'))
  print('Number of DT files: %d' % len(dt_files))
  p = re.compile(r'.*(image_\d+)\.txt')
  print('Calculating matches')
  for dt_file in tqdm(dt_files):
    # find corresponding gt file
    fname = basename(dt_file)
    key = p.match(fname).group(1)
    if key not in all_gt:
      print('Result %s not found in groundtruths! This file will be ignored')
      continue

    # calculate matches to groundtruth and append to list
    gt_polygons = all_gt[key]
    with open(dt_file, 'r') as f:
      dt_lines = f.readlines()
    dt_polygons = [polygon_from_str(o) for o in dt_lines]
    dt_match = []
    for dt_poly in dt_polygons:
      match = False
      for gt_poly in gt_polygons:
        if (polygon_iou(dt_poly, gt_poly) >= IOU_THRESH):
          match = True
          break
      dt_match.append(match)
    all_dt_match.extend(dt_match)

    # calculate scores and append to list
    dt_scores = [float(o.split(',')[8]) for o in dt_lines]
    all_dt_scores.extend(dt_scores)

  # calculate precision, recall and f-measure at all thresholds
  all_dt_match = np.array(all_dt_match, dtype=np.bool).astype(np.int)
  all_dt_scores = np.array(all_dt_scores)
  sort_idx = np.argsort(all_dt_scores)[::-1] # sort in descending order
  all_dt_match = all_dt_match[sort_idx]
  all_dt_scores = all_dt_scores[sort_idx]
  
  n_pos = np.cumsum(all_dt_match)
  n_dt = np.arange(1, len(all_dt_match)+1)
  precision = n_pos.astype(np.float) / n_dt.astype(np.float)
  recall = n_pos.astype(np.float) / float(n_gt)
  eps = 1e-9
  fmeasure = 2.0 / ((1.0 / (precision + eps)) + (1.0 / (recall + eps)))
  
  # find maximum fmeasure
  max_idx = np.argmax(fmeasure)
  
  eval_results = {
    'fmeasure': fmeasure[max_idx],
    'precision': precision[max_idx],
    'recall': recall[max_idx],
    'threshold': all_dt_scores[max_idx],
    'all_precisions': precision,
    'all_recalls': recall
  }

  # evaluation summary
  print('=================================================================')
  print('Maximum f-measure: %f' % eval_results['fmeasure'])
  print('  |-- precision:   %f' % eval_results['precision'])
  print('  |-- recall:      %f' % eval_results['recall'])
  print('  |-- threshold:   %f' % eval_results['threshold'])
  print('=================================================================')

  # save evaluation results
  if not exists(save_dir):
    os.makedirs(save_dir)
  data_save_path = join(save_dir, 'eval_data.pkl')
  with open(data_save_path, 'wb') as f:
    pickle.dump(eval_results, f)
  print('Evaluation results data written to {}'.format(data_save_path))

  # plot precision-recall curve
  vis_save_path = join(save_dir, 'pr_curve.png')
  plt.clf()
  plt.plot(recall, precision)
  plt.xlim(0, 1)
  plt.ylim(0, 1)
  plt.title('Precision-Recall Curve')
  plt.grid()
  plt.xlabel('Recall')
  plt.ylabel('Precision')
  plt.savefig(vis_save_path, dpi=200)
  print('Precision-recall curve written to {}'.format(vis_save_path))

  return


def evaluate_all_submissions(root_dir, gt_dir):
  """
  Evaluate all submissions and summarize.
  ARGS
    root_dir: root directory of submissions
  """
  # all submission directory has a 'dt_txts'

  # pairs of submission identifier and detection directory
  sub_id_sub_dir_pairs = []
  def _recursive_find_sub_dirs(curr_dir):
    for root, subdirs, files in os.walk(curr_dir):
      for subdir in subdirs:
        if basename(subdir) == 'dt_txts':
          identifier = basename(root)
          sub_id_sub_dir_pairs.append((identifier, join(root, subdir)))
        else:
          _recursive_find_sub_dirs(subdir)

  _recursive_find_sub_dirs(root_dir)

  for identifier, sub_dir in sub_id_sub_dir_pairs:
    print('Found submission %8s in directory %s' % (identifier, sub_dir))
  n_submissions = len(sub_id_sub_dir_pairs)

  # evaluate all submissions
  for i, pair in enumerate(sub_id_sub_dir_pairs):
    identifier, sub_dir = pair
    print('[%2d/%2d] Start evaluating "%s" at directory %s' % (i+1, n_submissions, identifier, sub_dir))
    det_eval(gt_dir, sub_dir, join(sub_dir, 'eval_results'))


if __name__ == '__main__':
  evaluate_all_submissions('../data/submissions', '../data/gt_txts/')
