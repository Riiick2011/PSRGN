import os
import shutil
import time
import json
import pickle
from typing import Dict

import numpy as np

import torch

from .metrics import ANETdetection


def load_results_from_pkl(filename):
    # load from pickle file
    assert os.path.isfile(filename)
    with open(filename, "rb") as f:
        results = pickle.load(f)
    return results

def load_results_from_json(filename):
    assert os.path.isfile(filename)
    with open(filename, "r") as f:
        results = json.load(f)
    # for activity net external classification scores
    if 'results' in results:
        results = results['results']
    return results

def results_to_dict(results):
    """convert result arrays into dict used by json files"""
    # video ids and allocate the dict
    vidxs = sorted(list(set(results['video-id'])))
    results_dict = {}
    for vidx in vidxs:
        results_dict[vidx] = []

    # fill in the dict
    for vidx, start, end, label, score in zip(
        results['video-id'],
        results['t-start'],
        results['t-end'],
        results['label'],
        results['score']
    ):
        results_dict[vidx].append(
            {
                "label" : int(label),
                "score" : float(score),
                "segment": [float(start), float(end)],
            }
        )
    return results_dict


def results_to_array(results, num_pred):
    # video ids and allocate the dict
    vidxs = sorted(list(set(results['video-id'])))
    results_dict = {}
    for vidx in vidxs:
        results_dict[vidx] = {
            'score'   : [],
            'segment' : [],
        }

    # fill in the dict
    for vidx, start, end, score in zip(
        results['video-id'],
        results['t-start'],
        results['t-end'],
        results['score']
    ):
        results_dict[vidx]['score'].append(float(score))
        results_dict[vidx]['segment'].append(
            [float(start), float(end)]
        )

    for vidx in vidxs:
        score = np.asarray(results_dict[vidx]['score'])
        segment = np.asarray(results_dict[vidx]['segment'])

        # the score should be already sorted, just for safety
        inds = np.argsort(score)[::-1][:num_pred]
        score, segment = score[inds], segment[inds]
        results_dict[vidx]['score'] = score
        results_dict[vidx]['segment'] = segment

    return results_dict


def postprocess_results(results,num_pred=200):

    # load results and convert to dict
    if isinstance(results, str):
        results = load_results_from_pkl(results)
    # array -> dict
    results = results_to_array(results, num_pred)

    # load external classification scores
    # dict for processed results
    processed_results = {}

    # process each video
    for vid, result in results.items():
        vid_ap=[]
        pred_score, pred_segment = result['score'], result['segment']
        for i in range(len(pred_score)):
            tmp_proposal = {}
            tmp_proposal["score"] = pred_score[i]
            tmp_proposal["segment"] = [pred_segment[i][0],pred_segment[i][1]]
            vid_ap.append(tmp_proposal)
        processed_results[vid]=vid_ap
    return processed_results
