from __future__ import division
import os
from os.path import join, exists, basename, splitext, dirname
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
  # remove possible utf-8 BOM
  if line.startswith('\xef\xbb\xbf'):
    line = line[3:]
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


def average_precision(rec, prec):
  # source: https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py#L47-L61
  # correct AP calculation
  # first append sentinel values at the end
  mrec = np.concatenate(([0.], rec, [1.]))
  mpre = np.concatenate(([0.], prec, [0.]))

  # compute the precision envelope
  for i in range(mpre.size - 1, 0, -1):
    mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

  # to calculate area under PR curve, look for points
  # where X axis (recall) changes value
  i = np.where(mrec[1:] != mrec[:-1])[0]

  # and sum (\Delta recall) * prec
  ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

  return ap


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


def find_submissions(root_dir, name):
  """
  Find pairs of submission identifiers and submission directory.
  """
  sub_id_dir_pairs = []
  def _recursive_find_sub_dirs(curr_dir):
    for root, subdirs, files in os.walk(curr_dir):
      for subdir in subdirs:
        if basename(subdir) == name:
          identifier = basename(root)
          sub_id_dir_pairs.append((identifier, root))
          break
        else:
          _recursive_find_sub_dirs(subdir)

  _recursive_find_sub_dirs(root_dir)
  return sub_id_dir_pairs


def evaluate_all_submissions(root_dir, gt_dir, skip_evaluated=False):
  """
  Evaluate all submissions and summarize.
  ARGS
    root_dir: root directory of submissions
  """
  # all submission directory has a 'dt_txts'
  sub_id_dir_pairs = find_submissions(root_dir, 'dt_txts')

  for identifier, sub_dir in sub_id_dir_pairs:
    print('Found submission %8s in directory %s' % (identifier, sub_dir))
  n_submissions = len(sub_id_dir_pairs)

  # evaluate all submissions
  for i, pair in enumerate(sub_id_dir_pairs):
    identifier, sub_dir = pair
    if skip_evaluated and exists(join(sub_dir, 'eval_results', 'eval_data.pkl')):
      print('Skip %s' % identifier)
    else:
      print('[%2d/%2d] Start evaluating "%s" at directory %s' % (i+1, n_submissions, identifier, sub_dir))
      det_eval(gt_dir, join(sub_dir, 'dt_txts'), join(sub_dir, 'eval_results'))


def summarize_evaluation(root_dir):
  sub_id_dir_pairs = find_submissions(root_dir, 'dt_txts')
  table_items = {'id': [], 'fmeasure': [], 'precision': [], 'recall': [], 'ap': []}
  pr_data = []

  # read results data
  for identifier, sub_dir in sub_id_dir_pairs:
    result_data_path = join(sub_dir, 'eval_results', 'eval_data.pkl')
    with open(result_data_path, 'rb') as f:
      eval_results = pickle.load(f)
    if 'ap' not in eval_results:
      ap = average_precision(eval_results['all_recalls'], eval_results['all_precisions'])
      eval_results['ap'] = ap
    table_items['id'].append(identifier)
    table_items['fmeasure'].append(eval_results['fmeasure'])
    table_items['precision'].append(eval_results['precision'])
    table_items['recall'].append(eval_results['recall'])
    table_items['ap'].append(eval_results['ap'])
    pr_data.append((identifier, eval_results['all_precisions'], eval_results['all_recalls']))
  for k in table_items.keys():
    table_items[k] = np.array(table_items[k])

  n_submissions = len(table_items['id'])

  def _rank(measure):
    """
    Find descending-order rankings (starts from 1) of elements in an array.
    """
    sort_idx = np.argsort(-measure)
    ranks = np.empty(len(measure), int)
    ranks[sort_idx] = np.arange(1, len(measure)+1)
    return ranks

  fm_rank = _rank(table_items['fmeasure'])
  ap_rank = _rank(table_items['ap'])
  
  # summary table
  title_fmt = '%5s | %8s | %8s | %8s | %8s | %8s | %8s'
  row_fmt = '%5s | %.6f | %8d | %.6f | %.6f | %.6f | %8d'
  print(title_fmt % ('ID', 'fmeasure', 'fm-rank', 'prec', 'rec', 'ap', 'ap-rank'))
  for i in range(n_submissions):
    print(row_fmt % (table_items['id'][i],
                     table_items['fmeasure'][i],
                     fm_rank[i],
                     table_items['precision'][i],
                     table_items['recall'][i],
                     table_items['ap'][i],
                     ap_rank[i]))

  # summary graph
  vis_save_path = join(root_dir, 'pr_summary.png')
  plt.clf()
  plt.xlim(0, 1)
  plt.ylim(0, 1)
  plt.grid()
  plt.title('Precision-Recall Curve')
  plt.xlabel('Recall')
  plt.ylabel('Precision')
  colors = ['#4c0000', '#331a1a', '#ff9180', '#e6b4ac', '#e53d00', '#a66953', '#59392d', '#ffb380', '#a65800', '#4c2900', '#e6d2ac', '#cca300', '#59502d', '#a3cc00', '#8a994d', '#eaffbf', '#00ff00', '#005900', '#00b330', '#10401d', '#608071', '#00ffaa', '#238c69', '#80ffe5', '#009ba6', '#00ccff', '#263033', '#1d5673', '#3d9df2', '#0016a6', '#99a0cc', '#5353a6', '#110040', '#494359', '#7e39e6', '#5c3366', '#2e1a33', '#bf30b6', '#cc99bb', '#7f0044', '#f20061', '#f279aa', '#a60016', '#7f4048']
  for i, pr_data_item in enumerate(pr_data):
    identifier, all_precisions, all_recalls = pr_data_item
    plt.plot(all_recalls, all_precisions, label=identifier, color=colors[i])
  plt.legend(prop={'size':6})
  plt.savefig(vis_save_path, dpi=200)
  print('PR summary saved to {}'.format(vis_save_path))


if __name__ == '__main__':
  # evaluate_all_submissions('../data/submissions', '../data/gt_txts/', skip_evaluated=True)
  summarize_evaluation('../data/submissions')
