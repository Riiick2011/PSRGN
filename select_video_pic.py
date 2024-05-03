import os
import pickle
import numpy as np
import json
import pandas as pd
from typing import Tuple
from typing import List
import pickle
from typing import Dict
def segment_tiou(target_segments, test_segments):
    """Compute intersection over union btw segments
    Parameters
    ----------
    target_segments : ndarray
        2-dim array in format [m x 2:=[init, end]]
    test_segments : ndarray
        2-dim array in format [n x 2:=[init, end]]
    Outputs
    -------
    tiou : ndarray
        2-dim array [m x n] with IOU ratio.
    Note: It assumes that target-segments are more scarce that test-segments
    """
    if target_segments.ndim != 2 or test_segments.ndim != 2:
        raise ValueError('Dimension of arguments is incorrect')

    m, n = target_segments.shape[0], test_segments.shape[0]
    t_overlap = np.empty((m, n))
    tiou = np.empty((m, n))
    for i in range(m):
        tt1 = np.maximum(target_segments[i, 0], test_segments[:, 0])
        tt2 = np.minimum(target_segments[i, 1], test_segments[:, 1])

        # Non-negative overlap score
        intersection = (tt2 - tt1).clip(0)
        union = ((test_segments[:, 1] - test_segments[:, 0]) +
                 (target_segments[i, 1] - target_segments[i, 0]) -
                 intersection)
        # Compute overlap as the ratio of the intersection
        # over union of two segments at the frame level.
        tiou[i, :] = intersection / union
        t_overlap[i, :] = intersection
    return tiou
def load_gt_seg_from_json(json_file, split=None, label='label_id', label_offset=0):
    # load json file
    with open(json_file, "r", encoding="utf8") as f:
        json_db = json.load(f)
    json_db = json_db['database']

    vids, starts, stops, labels = [], [], [], []
    for k, v in json_db.items():

        # filter based on split
        if (split is not None) and v['subset'].lower() != split:
            continue

        # video id
        vids += [k] * len(v['annotations'])
        # for each event, grab the start/end time and label
        for event in v['annotations']:
            starts += [float(event['segment'][0])]
            stops += [float(event['segment'][1])]
            if isinstance(event[label], (Tuple, List)):
                # offset the labels by label_offset
                label_id = 0
                for i, x in enumerate(event[label][::-1]):
                    label_id += label_offset**i + int(x)
            else:
                # load label_id directly
                label_id = int(event[label])
            labels += [label_id]

    # move to pd dataframe
    gt_base = pd.DataFrame({
        'video-id' : vids,
        't-start' : starts,
        't-end': stops,
        'label': labels
    })

    return gt_base
def load_pred_seg_from_json(json_file, label='label_id', label_offset=0):
    # load json file
    with open(json_file, "r", encoding="utf8") as f:
        json_db = json.load(f)
    json_db = json_db['database']

    vids, starts, stops, labels, scores = [], [], [], [], []
    for k, v, in json_db.items():
        # video id
        vids += [k] * len(v)
        # for each event
        for event in v:
            starts += [float(event['segment'][0])]
            stops += [float(event['segment'][1])]
            if isinstance(event[label], (Tuple, List)):
                # offset the labels by label_offset
                label_id = 0
                for i, x in enumerate(event[label][::-1]):
                    label_id += label_offset**i + int(x)
            else:
                # load label_id directly
                label_id = int(event[label])
            labels += [label_id]
            scores += [float(event['scores'])]

    # move to pd dataframe
    pred_base = pd.DataFrame({
        'video-id' : vids,
        't-start' : starts,
        't-end': stops,
        'label': labels,
        'score': scores
    })

    return pred_base
gt=load_gt_seg_from_json(json_file='./data/thumos14.json',split='test')
#print(gt)
with open("./test_results/Ours_eval_results.pkl","rb") as f:
    prd_ours=pickle.load(f)
pro_our= pd.DataFrame({
                'video-id' : prd_ours['video-id'],
                't-start' : prd_ours['t-start'].tolist(),
                't-end': prd_ours['t-end'].tolist(),
                'label': prd_ours['label'].tolist(),
                'score': prd_ours['score'].tolist()
            })
#print(pro_our)
with open("./test_results/Baseline_eval_results.pkl","rb") as f:
    prd_base=pickle.load(f)
pro_mean= pd.DataFrame({
                'video-id' : prd_base['video-id'],
                't-start' : prd_base['t-start'].tolist(),
                't-end': prd_base['t-end'].tolist(),
                'label': prd_base['label'].tolist(),
                'score': prd_base['score'].tolist()
            })
#print(pro_mean)

with open("./test_results/CSPG_eval_results.pkl","rb") as f:
    prd_CSPG=pickle.load(f)
pro_bert= pd.DataFrame({
                'video-id' : prd_CSPG['video-id'],
                't-start' : prd_CSPG['t-start'].tolist(),
                't-end': prd_CSPG['t-end'].tolist(),
                'label': prd_CSPG['label'].tolist(),
                'score': prd_CSPG['score'].tolist()
            })
with open("./test_results/SPGN_eval_results.pkl","rb") as f:
    prd_SPG=pickle.load(f)
pro_SPG= pd.DataFrame({
                'video-id' : prd_SPG['video-id'],
                't-start' : prd_SPG['t-start'].tolist(),
                't-end': prd_SPG['t-end'].tolist(),
                'label': prd_SPG['label'].tolist(),
                'score': prd_SPG['score'].tolist()
            })
#print(pro_mean)
'''
gt = pickle.load(open('gt_dump.pc', "rb"))
pro_our= pickle.load(open('pred_dump_our.pc', "rb"))
pro_mean=pickle.load(open('pred_dump_mean.pc', "rb"))
pro_bert=pickle.load(open('pred_dump_bert.pc', "rb"))
'''
video_lis=os.listdir('/data/zy/datasets/I3D/Flow_Test_All')
video_name='video_test_0000179'


for video_name in video_lis:
    cls=11

    gt_cls=gt.loc[gt['label']==cls]
    pro_our_cls=pro_our.loc[pro_our['label']==cls]
    pro_mean_cls=pro_mean.loc[pro_mean['label']==cls]
    pro_bert_cls=pro_bert.loc[pro_bert['label']==cls]
    pro_SPG_cls = pro_SPG.loc[pro_SPG['label'] == cls]
    gt_cls_video=gt_cls.loc[gt_cls['video-id']==video_name]

    gt_start=gt_cls_video['t-start'].values[:]

    gt_end=gt_cls_video['t-end'].values[:]
    gt_se=[]
    for j in range(len(gt_start)):
        gt_se.append([gt_start[j],gt_end[j]])
    if len(gt_se)==0:
        continue
    gt_se=np.array(gt_se)
    pro_our_cls_vid=pro_our_cls.loc[pro_our_cls['video-id']==video_name]
    pro_our_start=pro_our_cls_vid['t-start'].values[:]
    pro_our_end=pro_our_cls_vid['t-end'].values[:]
    prop_our_se=[]
    for j in range(len(pro_our_start)):
        prop_our_se.append([pro_our_start[j],pro_our_end[j]])
    prop_our_se=np.array(prop_our_se)
    pro_mean_cls_vid=pro_mean_cls.loc[pro_mean_cls['video-id']==video_name]
    pro_mean_start=pro_mean_cls_vid['t-start'].values[:]
    pro_mean_end=pro_mean_cls_vid['t-end'].values[:]
    prop_mean_se=[]
    for j in range(len(pro_mean_start)):
        prop_mean_se.append([pro_mean_start[j],pro_mean_end[j]])
#print(pro_mean_cls_vid)
    prop_mean_se=np.array(prop_mean_se)
    pro_bert_cls_vid=pro_bert_cls.loc[pro_bert_cls['video-id']==video_name]
#print(pro_bert_cls_vid)
    pro_bert_start=pro_bert_cls_vid['t-start'].values[:]
    pro_bert_end=pro_bert_cls_vid['t-end'].values[:]
    prop_bert_se=[]
    for j in range(len(pro_bert_start)):
        prop_bert_se.append([pro_bert_start[j],pro_bert_end[j]])

    prop_bert_se=np.array(prop_bert_se)
    #_______________________________________________
    pro_SPG_cls_vid = pro_SPG_cls.loc[pro_SPG_cls['video-id'] == video_name]
    # print(pro_bert_cls_vid)
    pro_SPG_start = pro_SPG_cls_vid['t-start'].values[:]
    pro_SPG_end = pro_SPG_cls_vid['t-end'].values[:]
    pro_SPG_se = []
    for j in range(len(pro_SPG_start)):
        pro_SPG_se.append([pro_SPG_start[j], pro_SPG_end[j]])

    pro_SPG_se = np.array(pro_SPG_se)
    #_______________________________________________
    iou_our=segment_tiou(gt_se,prop_our_se)
    iou_mean=segment_tiou(gt_se,prop_mean_se)
    iou_bert=segment_tiou(gt_se,prop_bert_se)
    iou_SPG = segment_tiou(gt_se, pro_SPG_se)
    iou_our_max=iou_our.max(axis=1)
    iou_our_idx=iou_our.argmax(axis=1)
    iou_mean_max=iou_mean.max(axis=1)
    iou_mean_idx=iou_mean.argmax(axis=1)
    iou_bert_max=iou_bert.max(axis=1)
    iou_bert_idx=iou_bert.argmax(axis=1)
    iou_SPG_max = iou_SPG.max(axis=1)
    iou_SPG_idx = iou_SPG.argmax(axis=1)
    for i in range(min(iou_bert_max.shape[0],iou_SPG_max.shape[0])):
        if iou_our_max[i]>iou_bert_max[i] and iou_bert_max[i]>iou_mean_max[i] and iou_our_max[i]>iou_SPG_max[i] and iou_SPG_max[i]>iou_mean_max[i]:
            print(video_name,i,gt_se[i],iou_our_max[i],prop_our_se[iou_our_idx[i]],prop_mean_se[iou_mean_idx[i]],prop_bert_se[iou_bert_idx[i]],pro_SPG_se[iou_SPG_idx[i]])

