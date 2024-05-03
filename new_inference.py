# python imports
import argparse
import os
import glob
import time
import json
from pprint import pprint

# torch imports
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
from eval import evaluation_proposal


# our code
from libs.core import load_config
from libs.modelings.models import PtTransformer_gcn_1,PtTransformer_gcn_loss
from libs.datasets.Datasets import THUMOS14Dataset,BufferList
from libs.utils import ANETdetection, batched_nms
from libs.utils.postprocessing_single import postprocess_results
import pickle
def trivial_batch_collator(batch):
    """
        A batch collator that does nothing
    """
    return batch
def inference(
        video_list,
        rois,
        scores, cls_scores,
        test_cfg
    ):
    results = []

    # 1: gather video meta information
    vid_idxs = [x['video_id'] for x in video_list]
    vid_fps = [x['fps'] for x in video_list]
    vid_lens = [x['duration'] for x in video_list]
    vid_ft_stride = [x['feat_stride'] for x in video_list]
    vid_ft_nframes = [x['feat_num_frames'] for x in video_list]

    # 2: inference on each single video and gather the results
    # upto this point, all results use timestamps defined on feature grids
    for idx, (vidx, fps, vlen, stride, nframes) in enumerate(
                zip(vid_idxs, vid_fps, vid_lens, vid_ft_stride, vid_ft_nframes)
    ):
        # gather per-video outputs
        rois_per_vid = rois[idx,:,1:].contiguous()
        scores_per_vid = scores[idx].contiguous()
        cls_scores_per_vid=cls_scores[idx].contiguous()
        ids=torch.nonzero(scores_per_vid>0)
        rois_per_vid=rois_per_vid[ids].squeeze(1)
        scores_per_vid=scores_per_vid[ids]
        # inference on a single video (should always be the case)
        results_per_vid = {'segments': rois_per_vid,
               'scores': scores_per_vid}
        # pass through video meta info
        results_per_vid['video_id'] = vidx
        results_per_vid['fps'] = fps
        results_per_vid['duration'] = vlen
        results_per_vid['feat_stride'] = stride
        results_per_vid['feat_num_frames'] = nframes
        results.append(results_per_vid)

        # step 3: postprocssing
    results = postprocessing(results,test_cfg)

    return results

def inference_single_video(
        points,
        fpn_masks,
        out_cls_logits,
        out_offsets,test_cfg
    ):
    # points F (list) [T_i, 4]
    # fpn_masks, out_*: F (List) [T_i, C]
    segs_all = []
    scores_all = []
    cls_idxs_all = []

    # loop over fpn levels
    for cls_i, offsets_i, pts_i, mask_i in zip(
            out_cls_logits, out_offsets, points, fpn_masks
    ):
        # sigmoid normalization for output logits
        pred_prob = (cls_i.sigmoid() * mask_i.unsqueeze(-1)).flatten()
        pts_i=pts_i.to(offsets_i.device)
        # Apply filtering to make NMS faster following detectron2
        # 1. Keep seg with confidence score > a threshold
        keep_idxs1 = (pred_prob > test_cfg["pre_nms_thresh"])
        pred_prob = pred_prob[keep_idxs1]
        topk_idxs = keep_idxs1.nonzero(as_tuple=True)[0]

        # 2. Keep top k top scoring boxes only
        num_topk = min(test_cfg["pre_nms_topk"], topk_idxs.size(0))
        pred_prob, idxs = pred_prob.sort(descending=True)
        pred_prob = pred_prob[:num_topk].clone()
        topk_idxs = topk_idxs[idxs[:num_topk]].clone()
        # 3. gather predicted offsets
        offsets = offsets_i[topk_idxs]
        pts = pts_i[topk_idxs]

        # 4. compute predicted segments (denorm by stride for output offsets)
        seg_left = pts[:, 0] - offsets[:, 0] * pts[:, 3]
        seg_right = pts[:, 0] + offsets[:, 1] * pts[:, 3]
        pred_segs = torch.stack((seg_left, seg_right), -1)

        # 5. Keep seg with duration > a threshold (relative to feature grids)
        seg_areas = seg_right - seg_left
        keep_idxs2 = seg_areas > test_cfg["duration_thresh"]

        # *_all : N (filtered # of segments) x 2 / 1
        segs_all.append(pred_segs[keep_idxs2])
        scores_all.append(pred_prob[keep_idxs2])
    # cat along the FPN levels (F N_i, C)
    segs_all, scores_all = [
        torch.cat(x) for x in [segs_all, scores_all]
    ]
    results = {'segments': segs_all,
               'scores': scores_all}

    return results


def postprocessing(results,test_cfg):
    # input : list of dictionary items
    # (1) push to CPU; (2) NMS; (3) convert to actual time stamps
    processed_results = []
    for results_per_vid in results:
        # unpack the meta info
        vidx = results_per_vid['video_id']
        fps = results_per_vid['fps']
        vlen = results_per_vid['duration']
        stride = results_per_vid['feat_stride']
        nframes = results_per_vid['feat_num_frames']
        # 1: unpack the results and move to CPU
        segs = results_per_vid['segments'].detach().cpu()
        scores = results_per_vid['scores'].detach().cpu()
        '''
        if test_cfg["nms_method"] != 'none':
            # 2: batched nms (only implemented on CPU)
            segs, scores = batched_nms(
                segs, scores,
                test_cfg["iou_threshold"],
                test_cfg["min_score"],
                test_cfg["max_seg_num"],
                use_soft_nms=(test_cfg["nms_method"] == 'soft'),
                multiclass=False,
                sigma=test_cfg["nms_sigma"],
                voting_thresh=test_cfg["voting_thresh"]
            )
        '''
        # 3: convert from feature grids to seconds
        if segs.shape[0] > 0:
            segs = (segs * stride + 0.5 * nframes) / fps
            # truncate all boundaries within [0, duration]
            segs[segs <= 0.0] *= 0.0
            segs[segs >= vlen] = segs[segs >= vlen] * 0.0 + vlen
        # 4: repack the results
        processed_results.append(
            {'video_id': vidx,
             'segments': segs,
             'scores': scores}
        )

    return processed_results

def generat_points(max_seq_len,fpn_levels,scale_factor,regression_range):
    points_list = []
    # initial points
    initial_points = torch.arange(0, max_seq_len, 1.0)

    # loop over all points at each pyramid level
    for l in range(fpn_levels):
        stride = scale_factor ** l
        reg_range = torch.as_tensor(
            regression_range[l], dtype=torch.float)
        fpn_stride = torch.as_tensor(stride, dtype=torch.float)
        points = initial_points[::stride][:, None]
        # add offset if necessary (not in our current model)
        # pad the time stamp with additional regression range / stride
        reg_range = reg_range[None].repeat(points.shape[0], 1)
        fpn_stride = fpn_stride[None].repeat(points.shape[0], 1)
        # size: T x 4 (ts, reg_range, stride)
        points_list.append(torch.cat((points, reg_range, fpn_stride), dim=1))
    buffer_points = BufferList(points_list)
    feat_lens = []
    pts_list = []
    for i in range(fpn_levels):
        feat_lens.append(int(max_seq_len / scale_factor ** i))
    for feat_len, buffer_pts in zip(feat_lens, buffer_points):
        assert feat_len <= buffer_pts.shape[0], "Reached max buffer length for point generator"
        pts = buffer_pts[:feat_len, :]
        pts_list.append(pts)
    return pts_list

################################################################################
def main(args):
    """0. load config"""
    # sanity check
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")
    assert len(cfg['val_split']) > 0, "Test set must be specified!"
    model_cfg = cfg["model"]
    train_cfg = cfg["train_cfg"]
    data_cfg = cfg["dataset"]
    test_cfg=cfg["test_cfg"]
    if args.sigma>0:
        train_cfg["nms_sigma"]=args.sigma
    if args.iou_threshold>0:
        train_cfg["iou_threshold"]=args.iou_threshold
    model = PtTransformer_gcn_loss(model_configs=model_cfg, train_cfg=train_cfg).cuda()
    ckpt_file = args.ckpt
    if args.topk > 0:
        cfg['model']['test_cfg']['max_seg_num'] = args.topk
    val_dataset=THUMOS14Dataset(is_training=False, split=cfg["val_split"], data_cfg=data_cfg, model_cfg=model_cfg,
                    train_cfg=train_cfg)
    val_loader = torch.utils.data.DataLoader(
       val_dataset,
        batch_size=1, shuffle=False,
        collate_fn=trivial_batch_collator,
        num_workers=cfg["loader"]["num_workers"], pin_memory=True)
    checkpoint = torch.load(ckpt_file)
    model.load_state_dict(checkpoint['model'])
    # set up evaluator
    #print("\nStart testing model {:s} ...".format(cfg['model_name']))
    start = time.time()
    model.eval()
    # dict for results (for our evaluation code)
    results = {
        'video-id': [],
        't-start': [],
        't-end': [],
        'label': [],
        'score': []
    }
    for iter_idx, video_list in enumerate(val_loader,0):
        # forward the model (wo. grad)
        with torch.no_grad():
            input_feat=video_list[0]['feats']
            max_len = input_feat.size()[1]
            # input length < self.max_seq_len, pad to max_seq_len
            if max_len <= data_cfg["max_seq_len"]:
                max_len = data_cfg["max_seq_len"]
            else:
                # pad the input to the next divisible size
                stride = model.max_div_factor
                max_len = (max_len + (stride - 1)) // stride * stride
            padding_size = [0, max_len - input_feat.size()[1]]
            batched_input = F.pad(
                input_feat, padding_size, value=0.0).unsqueeze(0)
            batched_mask = torch.arange(max_len) < input_feat.size()[1]
            batched_mask=batched_mask.unsqueeze(0)
            fpn_masks,out_cls_logits, out_offsets,rois,scorces,rois_mask,cls_log,com_log,_,_,_,_= model([batched_input,batched_mask],False)
            cls_log=F.softmax(cls_log,dim=2)
            output = inference(
                video_list, rois,scorces,cls_log,test_cfg
            )
            num_vids = len(output)
            for vid_idx in range(num_vids):
                if output[vid_idx]['segments'].shape[0] > 0:
                    results['video-id'].extend(
                        [output[vid_idx]['video_id']] *
                        output[vid_idx]['segments'].shape[0]
                    )
                    results['t-start'].append(output[vid_idx]['segments'][:, 0])
                    results['t-end'].append(output[vid_idx]['segments'][:, 1])
                    results['score'].append(output[vid_idx]['scores'])

            # upack the results into ANet format
    results['t-start'] = torch.cat(results['t-start']).numpy()
    results['t-end'] = torch.cat(results['t-end']).numpy()
    results['score'] = torch.cat(results['score']).numpy()
    end = time.time()
    print("All done! Total time: {:0.2f} sec".format(end - start))
    results = postprocess_results(results)
    output_dict = {"version": "THUMOS14", "results": results, "external_data": {}}

    with open('./outputs/detection_result.json', "w") as out:
        json.dump(output_dict, out)
    a=evaluation_proposal()
    return

################################################################################
if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
      description='Train a point-based transformer for action localization')
    parser.add_argument('config', type=str, metavar='DIR',
                        help='path to a config file')
    parser.add_argument('ckpt', type=str, metavar='DIR',
                        help='path to a checkpoint')
    parser.add_argument('-t', '--topk', default=-1, type=int,
                        help='max number of output actions (default: -1)')
    parser.add_argument('--saveonly', action='store_true',
                        help='Only save the ouputs without evaluation (e.g., for test set)')
    parser.add_argument('-s', '--sigma', default=-1.0, type=float,
                        help='the parameter of Softnms')
    parser.add_argument('-i', '--iou_threshold', default=-1.0, type=float,
                        help='the parameter of Softnms')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        help='print frequency (default: 10 iterations)')
    args = parser.parse_args()
    main(args)
