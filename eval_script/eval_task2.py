from __future__ import division
import os
import re
import sys
from shapely.geometry import Polygon
from shapely.geometry import MultiPoint
import numpy as np
from difflib import SequenceMatcher
from collections import defaultdict
import operator
import editdistance
from hanziconv import HanziConv
import csv
import glob  

def polygon_from_str(polygon_points):
  """
  Create a shapely polygon object from gt or dt line.
  """
  #polygon_points = [float(o) for o in line.split(',')[:8]]
  
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

def regex(st):
  st = ''.join(st.split(' '))
  st = re.sub("\"","",st)
  st = st.decode('utf-8')
  new_st = re.sub(ur'[^\u4e00-\u9fa5\u0041-\u005a\u0061-\u007a0-9]+','',st)
  # traditional chinese to simplified chinese
  new_st = HanziConv.toSimplified(new_st)
  # new_st = new_st.encode('utf-8')
  return new_st

def ed(str1,str2):
  str1 = regex(str1.lower())  
  str2 = regex(str2.lower())
  return editdistance.eval(str1,str2)

def recog_eval(gt_dir,recog_dir):  
  print 'start testing...'
  iou_thresh = 0.5
  val_names = os.listdir(gt_dir)  
  similar_sum = 0 
  gt_count = 0
  ed_sum = 0
  for i,val_name in enumerate(val_names): 
  # for val_name in val_names[:10]:   
    # if val_name != 'image_203.txt':
    #   continue
    with open(os.path.join(gt_dir, val_name)) as f:
      gt_lines = [o.decode('utf-8-sig').encode('utf-8').strip() for o in f.readlines()]  
    gts = [g.split(',')[0:10] for g in gt_lines]  
    easy_gts = [g for g in gts if int(g[8]) == 0]
    gt_count += len(easy_gts)  
    flag_strick = [False]*len(gts)
    val_path = os.path.join(recog_dir, 'task2_' + val_name)
    if not os.path.exists(val_path):
      dt_lines = []
    else:
      with open(val_path) as f:
        dt_lines = [o.decode('utf-8-sig').encode('utf-8').strip() for o in f.readlines()]
    dts = [d.split(',') for d in dt_lines]
    dictlist = [defaultdict(int) for x in range(len(gts))]
    for index_dt, dt in enumerate(dts):    
      dt_coor = [float(d) for d in dt[0:8]]    
      ious = []
      for index_gt, gt in enumerate(gts):
        gt_coor = [float(g) for g in gt[0:8]]   
        '''   
        rectangle_1 = []
        rectangle_2 = []
        for ii in range(0,8,2):
          rectangle_1.append([gt_coor[ii],gt_coor[ii+1]])
          rectangle_2.append([dt_coor[ii],dt_coor[ii+1]])
        union_area = union(rectangle_1, rectangle_2) 
        intersect_area = intersect(rectangle_1, rectangle_2)
        iou = intersect_area/union_area
        '''
        rectangle_1 = polygon_from_str(gt_coor)
        rectangle_2 = polygon_from_str(dt_coor)
        iou = polygon_iou(rectangle_1,rectangle_2)
        ious.append(iou)
      max_iou = max(ious)
      max_index = ious.index(max_iou)
      if max_iou > iou_thresh:
      	dictlist[max_index][index_dt] = max_iou
    dt_gt = defaultdict(int)
    for index_gt_dts, gt_dts in enumerate(dictlist):
      if len(gt_dts) == 0:
      	continue
      else:
      	sorted_dts = sorted(gt_dts.items(), key=operator.itemgetter(1))
      	dt_gt[sorted_dts[0][0]] = index_gt_dts

    for index_dt, dt in enumerate(dts):
      # matched gt and dt
      if index_dt in dt_gt.keys():
      	index_gt = dt_gt[index_dt]
      	if int(gts[index_gt][8]) == 0:
        	gt_str = gts[index_gt][9]
        	dt_str = dt[8]
        	ed_sum += ed(gt_str, dt_str)
      # unmatched dt
      else:
      	dt_str = dt[8]
      	gt_str = ''
      	ed_sum += ed(dt_str, gt_str)
    # unmatched gt
    for index_gt, gt in enumerate(gts):
      if index_gt not in dt_gt.values():
        dt_str = ''
        gt_str = gt[9]
        ed_sum += ed(gt_str, dt_str)

  percent_accuracy = ed_sum/len(val_names)
  print('AED: %d / %d = %f' % (ed_sum, len(val_names), percent_accuracy))
  return percent_accuracy

def eval_all(gt_dir,test_path):
  users = os.listdir(test_path)
  user_dirs = glob.glob('/home/pxu/workspace/evaluate1/test/*/*/')
  recog_dirs = [os.path.join(u,'recog_txts') for u in user_dirs if 'recog_txts' in os.listdir(u)]
  user_dirs = [a.split('/')[-2] for a in recog_dirs]
  print user_dirs
  column = {}
  if os.path.exists('./recog_record_test.csv'):
    with open('./recog_record_test.csv', 'rb') as f:
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
    with open('./recog_record_test.csv', 'wb') as csvfile:
      writer = csv.writer(csvfile)
      writer.writerow(keys)
      writer.writerows(zip(*[column[key] for key in keys]))
  

if __name__ == "__main__":
  eval_all('./gt_txts/','./test/')
