from __future__ import division
import os
from os.path import join, exists, basename, splitext
import re
import sys
from shapely.geometry import Polygon
from shapely.geometry import MultiPoint
import numpy as np
import matplotlib.pyplot as plt
import time
import glob
import json

IOU_THRESH = 0.5
N_TEST = 4229


def polygon_from_str(line):
  """
  Create a shapely polygon object from gt or dt line.
  """
  polygon_points = [float(o) for o in line.split(',')[:8]]
  polygon_points = np.array(polygon_points).reshape(4, 2)
  polygon = Polygon(polygon_points)
  return polygon


def polygon_iou(poly1, poly2):
  """
  Intersection over union between two shapely polygons.
  """
  if not poly1.intersects(poly2):
    return 0
  else:
    inter_area = poly1.intersection(poly2).area
    union_area = poly1.union(poly2).area
    iou = float(inter_area) / union_area
    return iou


def det_eval(gt_dir, dt_dir, save_dir):
  """
  Evaluation detection by calculating the maximum f-measure across all thresholds.
  ARGS
    gt_dir: the directory of groundtruth files
    dt_dir: the directory of detection results files
    save_dir: the directory for saving evaluation results
  RETURN
  """
  # print 'start testing...'
  # print 'start time:',time.asctime(time.localtime(time.time()))
  # dt_count = 0
  # gt_count = 0
  # hit = 0

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
  dt_files = glob.glob(join(dt_dir, 'image_*.txt'))
  print('Number of DT files: %d' % len(dt_files))
  for dt_file in dt_files:
    # find corresponding gt file
    fname = splitext(basename(dt_file))[0]
    if fname not in all_gt:
      print('Result %s not found in groundtruths! This file will be ignored')
    continue

    # calculate matches to groundtruth and append to list
    gt_polygons = all_gt[fname]
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

  print('=================================================================')
  print('Maximum f-measure:       %f' % eval_results['fmeasure'])
  print('Corresponding precision: %f' % eval_results['precision'])
  print('Corresponding recall:    %f' % eval_results['recall'])
  print('Corresponding threshold: %f' % eval_results['threshold'])
  print('=================================================================')

  if not exists(save_dir):
    os.makedirs(save_dir)
  data_save_path = join(save_dir, 'eval_results_data.json')
  with open(data_save_path, 'w') as f:
    json.dump(eval_results, f)
    print('Evaluation results data written to {}'.format(data_save_path))

  return eval_results


if __name__ == '__main__':
  det_eval('../gt_txts/', '../data/test_submission/', '../data/test_submission/eval_results')
