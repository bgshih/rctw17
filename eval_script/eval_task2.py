from __future__ import division
import os
import re
import sys
from shapely.geometry import MultiPoint, Polygon
import numpy as np
from difflib import SequenceMatcher
from collections import defaultdict
import operator
import editdistance
from hanziconv import HanziConv
import csv
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

IOU_THRESH = 0.5
N_TEST = 4229

# precompile regex
gt_pattern = re.compile(r'^\s*((?:-?\d+,){7}\d+),(\d),\"(.*?)\"\s*$')
dt_pattern = re.compile(r'^\s*((?:-?\d+,){7}\d+),\"?(.*?)\"?\s*$')


def polygon_from_str(poly_str):
  """
  Create a shapely polygon object from gt or dt line.
  """
  polygon_points = [float(o) for o in poly_str.split(',')[:8]]
  polygon_points = np.array(polygon_points).reshape(4, 2)
  polygon = Polygon(polygon_points).convex_hull
  return polygon


def gt_from_line(gt_line):
  """
  Parse a line of groundtruth.
  """
  # remove possible utf-8 BOM
  if gt_line.startswith('\xef\xbb\xbf'):
    gt_line = gt_line[3:]
  gt_line = gt_line.decode('utf-8')

  matches = gt_pattern.match(gt_line)
  gt_poly_str = matches.group(1)
  gt_text = matches.group(3)

  # difficult = bool(matches.group(2))
  # if not difficult and gt_text == "###":
  #   print('Changing diffculty to true because groundtruth text is ###')
  #   difficult = True
  difficult = (gt_text == "###")
  gt_polygon = polygon_from_str(gt_poly_str)
  gt = {'polygon': gt_polygon, 'difficult': difficult, 'text': gt_text}
  return gt


def dt_from_line(dt_line):
  """
  Parse a line of detection result.
  """
  # remove possible utf-8 BOM
  if dt_line.startswith('\xef\xbb\xbf'):
    dt_line = dt_line[3:]
  dt_line = dt_line.decode('utf-8')

  matches = dt_pattern.match(dt_line)
  dt_poly_str = matches.group(1)
  dt_polygon = polygon_from_str(dt_poly_str)
  dt_text = matches.group(2)
  dt = {'polygon': dt_polygon, 'text': dt_text}
  return dt


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
      iou = float(inter_area) / union_area
    except shapely.geos.TopologicalError:
      print('shapely.geos.TopologicalError occured, iou set to 0')
      iou = 0
  return iou


def normalize_txt(st):
  """
  Normalize Chinese text strings by:
    - remove puncutations and other symbols
    - convert traditional Chinese to simplified
    - convert English chraacters to lower cases
  """
  st = ''.join(st.split(' '))
  st = re.sub("\"","",st)
  # remove any this not one of Chinese character, ascii 0-9, and ascii a-z and A-Z
  new_st = re.sub(ur'[^\u4e00-\u9fa5\u0041-\u005a\u0061-\u007a0-9]+','',st)
  # convert Traditional Chinese to Simplified Chinese
  new_st = HanziConv.toSimplified(new_st)
  # convert uppercase English letters to lowercase
  new_st = new_st.lower()
  return new_st


def text_distance(str1, str2):
  str1 = normalize_txt(str1)
  str2 = normalize_txt(str2)
  return editdistance.eval(str1, str2)


def recog_eval(gt_dir, recog_dir, visualize=False):
  test_gt_files = os.listdir(gt_dir)
  num_test = len(test_gt_files)
  gt_count = 0
  total_dist = 0

  for gt_file in tqdm(test_gt_files):
    # distance calculated on this example
    example_dist = 0

    # load groundtruth
    with open(os.path.join(gt_dir, gt_file)) as f:
      gt_lines = f.readlines()
    gts = [gt_from_line(o) for o in gt_lines]
    n_gt = len(gts)

    # load results
    dt_result_file = os.path.join(recog_dir, 'task2_' + gt_file)
    if not os.path.exists(dt_result_file):
      print('{} not found'.format(dt_result_file))
      dts = []
    else:
      with open(dt_result_file) as f:
        dt_lines = f.readlines()
      dts = [dt_from_line(o) for o in dt_lines]
    n_dt = len(dts)

    # match dt index of every gt
    gt_match = np.empty(n_gt, dtype=np.int)
    gt_match.fill(-1)
    # match gt index of every dt
    dt_match = np.empty(n_dt, dtype=np.int)
    dt_match.fill(-1)

    # find match for every GT
    for i, gt in enumerate(gts):
      max_iou = 0
      match_dt_idx = -1
      for j, dt in enumerate(dts):
        if dt_match[j] >= 0:
          # already matched to some GT
          continue
        iou = polygon_iou(gt['polygon'], dt['polygon'])
        if iou > IOU_THRESH and iou > max_iou:
          max_iou = iou
          match_dt_idx = j
      if match_dt_idx >= 0:
        gt_match[i] = match_dt_idx
        dt_match[match_dt_idx] = i

    match_tuples = []

    # calculate distances
    for i, gt in enumerate(gts):
      if gt['difficult'] == True:
        # do not accumulate distance no matter matched or not
        continue
      gt_text = gt['text']
      if gt_match[i] >= 0:
        # matched GT
        dt_text = dts[gt_match[i]]['text']
      else:
        # unmatched GT
        dt_text = u''
      dist = text_distance(gt_text, dt_text)
      example_dist += dist
      match_tuples.append((gt_text, dt_text, dist))
    
    for i, dt in enumerate(dts):
      if dt_match[i] == -1:
        # unmatched DT
        gt_text = u''
        dt_text = dts[i]['text']
        dist = text_distance(gt_text, dt_text)
        example_dist += dist
        match_tuples.append((gt_text, dt_text, dist))

    # accumulate distance
    total_dist += example_dist

    if visualize:
      print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
      print(u'GT: {}'.format(u'; '.join([o['text'] for o in gts])))
      print(u'DT: {}'.format(u'; '.join([o['text'] for o in dts])))
      for tp in match_tuples:
        gt_text, dt_text, dist = tp
        print(u'GT: "{}" matched to DT: "{}", distance = {}'.format(gt_text, dt_text, dist))
      print('Distance = %f. Total distance = %f' % (example_dist, total_dist))
      print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
      plt.clf()
      img_file = os.path.join('../data/icdar2017rctw_test', os.path.splitext(gt_file)[0] + '.jpg')
      img = mpimg.imread(img_file)
      plt.imshow(img)
      plt.show()

  average_dist = total_dist / num_test
  print('Average distance: %d / %d = %f' % (total_dist, len(test_gt_files), average_dist))
  return average_dist


def eval_all(gt_dir,test_path):
  users = os.listdir(test_path)
  user_dirs = glob.glob('/home/pxu/workspace/evaluate1/test/*/*/')
  recog_dirs = [os.path.join(u,'recog_txts') for u in user_dirs if 'recog_txts' in os.listdir(u)]
  user_dirs = [a.split('/')[-2] for a in recog_dirs]
  print user_dirs
  column = {}
  if os.path.exists('./recog_record.csv'):
    with open('./recog_record.csv', 'rb') as f:
      reader = csv.reader(f)
      headers = reader.next()    
      for h in headers:
        column[h] = []
      for row in reader:
        for h, v in zip(headers, row):
          column[h].append(v)             
    to_test = list(set(column['id'])^set(user_dirs))  
  else:
    column = {'id':[],'edit_distance':[]}
    to_test = user_dirs 
  to_test = sorted(to_test, key=lambda x:int(''.join(x.split('_'))))
  for i in to_test:
    print 'test dir:',i
    dt_dir = os.path.join(test_path,i.split('_')[0],i,'recog_txts')
    edit_dist = recog_eval(gt_dir,dt_dir)
    column['id'].append(str(i))
    column['edit_distance'].append(str(edit_dist))
    print 'column',column
    keys = ['id','edit_distance']
    with open('./recog_record.csv', 'wb') as csvfile:
      writer = csv.writer(csvfile)
      writer.writerow(keys)
      writer.writerows(zip(*[column[key] for key in keys]))
  

if __name__ == "__main__":
  # eval_all('./gt_txts/','./test/')
  recog_eval('../data/gt_txts', '../data/submissions/12/12_1/recog_txts/', visualize=False)
