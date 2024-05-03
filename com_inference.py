# python imports
import argparse
import os
import glob
import time
import json
from pprint import pprint
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
# torch imports
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
from eval import evaluation_proposal


# our code
from libs.core import load_config
from libs.modelings.model_baseline import Ours_model as final_model
from libs.datasets.Datasets import THUMOS14Dataset,BufferList
from libs.utils import ANETdetection, batched_nms,postprocess_results
import pickle
def trivial_batch_collator(batch):
    """
        A batch collator that does nothing
    """
    return batch
def inference(
        video_list,
        rois,
        scores, cls_scores,com_scores,
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
        com_scores_per_vid = com_scores[idx].contiguous()
        final_scores_per_vid=cls_scores_per_vid*scores_per_vid[:,None]
        final_scores_per_vid=final_scores_per_vid[:,1:]#*torch.exp(com_scores_per_vid)
        final_scores_per_vid=final_scores_per_vid.flatten()
        #keep_idxs1 = (final_scores_per_vid > test_cfg["pre_cls_nms_thresh"])
        keep_idxs1 = (final_scores_per_vid > 0.0)
        #print(test_cfg["pre_cls_nms_thresh"], "pre_cls_nms_thresh")
        final_scores_per_vid = final_scores_per_vid[keep_idxs1]
        topk_idxs = keep_idxs1.nonzero(as_tuple=True)[0]
        #print(topk_idxs.size(0),test_cfg["pre_cls_nms_thresh"])

        # 2. Keep top k top scoring boxes only
        #num_topk = min(test_cfg["pre_cls_nms_topk"], topk_idxs.size(0))
        num_topk =topk_idxs.size(0)
        #print(test_cfg["pre_cls_nms_topk"], "pre_cls_nms_topk")
        final_scores_per_vid, idxs = final_scores_per_vid.sort(descending=True)
        final_scores_per_vid = final_scores_per_vid[:num_topk].clone()
        topk_idxs = topk_idxs[idxs[:num_topk]].clone()

        # fix a warning in pytorch 1.9
        roi_idxs = torch.div(
            topk_idxs, test_cfg["num_classes"], rounding_mode='floor'
        )
        cls_idxs = torch.fmod(topk_idxs, test_cfg["num_classes"])
        rois_per_vid=rois_per_vid[roi_idxs].squeeze(1)

        # inference on a single video (should always be the case)
        results_per_vid = {'segments': rois_per_vid,
               'scores': final_scores_per_vid,
                'labels':cls_idxs}
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
        labels = results_per_vid['labels'].detach().cpu()
        '''
        print(test_cfg["cls_iou_threshold"], "cls_iou_threshold")
        print(test_cfg['cls_min_score'], "cls_min_score")
        print(test_cfg["cls_max_seg_num"], "cls_max_seg_num")
        print(test_cfg["cls_nms_sigma"], "cls_nms_sigma")
        print(test_cfg["voting_thresh"], "cls_oting_thresh")
        '''
        if test_cfg["nms_method"] != 'none':
            # 2: batched nms (only implemented on CPU)
            segs, scores, labels = batched_nms(
                segs, scores,
                test_cfg["cls_iou_threshold"],
                test_cfg["cls_min_score"],
                test_cfg["cls_max_seg_num"],
                use_soft_nms=(test_cfg["nms_method"] == 'soft'),
                multiclass=True,
                cls_idxs= labels,
                sigma=test_cfg["cls_nms_sigma"],
                voting_thresh=test_cfg["voting_thresh"]
            )
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
             'scores': scores,
             'labels': labels}
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
    if args.min_scores>-1:
        train_cfg["min_score"]=args.min_scores
    if args.num_generation>0:
        train_cfg["test_max_seg_num"]=args.num_generation
    if args.sigma>0:
        train_cfg["nms_sigma"]=args.sigma
    if args.iou_threshold>0:
        train_cfg["iou_threshold"]=args.iou_threshold
    if args.cls_sigma > 0:
            test_cfg["cls_nms_sigma"] = args.cls_sigma
    if args.cls_iou_threshold > 0:
            test_cfg["cls_iou_threshold"] = args.cls_iou_threshold
    model = final_model(model_configs=model_cfg, train_cfg=train_cfg).cuda()
    ckpt_file = args.ckpt

    result_num=ckpt_file.split('/')[-1].split('_')[1]

    if args.topk > 0:
        cfg['model']['test_cfg']['cls_max_seg_num'] = args.topk
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
    det_eval, output_file = None, None
    if not args.saveonly:
        val_db_vars = val_dataset.get_attributes()
        det_eval = ANETdetection(
            val_dataset.json_file,
            val_dataset.split[0],
            tiou_thresholds=val_db_vars['tiou_thresholds']
        )
    else:
        output_file = os.path.join(os.path.split(ckpt_file)[0], 'eval_results.pkl')
    output_file = os.path.join('./test_results', 'eval_results.pkl')
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
            #print(com_log)

            cls_log=F.softmax(cls_log,dim=2)
            #if video_list[0]["video_id"] == 'video_test_0000131':
                #print(cls_log[0][0])
            output = inference(
                video_list, rois,scorces,cls_log,com_log,test_cfg
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
                    results['label'].append(output[vid_idx]['labels'])
                    results['score'].append(output[vid_idx]['scores'])

            # upack the results into ANet format
    results['t-start'] = torch.cat(results['t-start']).numpy()
    results['t-end'] = torch.cat(results['t-end']).numpy()
    results['label'] = torch.cat(results['label']).numpy()
    results['score'] = torch.cat(results['score']).numpy()
    end = time.time()
    print("All done! Total time: {:0.2f} sec".format(end - start))
    f1 = open('./test_results/flow_results.txt', 'a+')
    f2 = open('./test_results/mid_results.txt', 'a+')
    if det_eval is not None:
        if (test_cfg['ext_score_file'] is not None) and isinstance(test_cfg['ext_score_file'], str):
            results = postprocess_results(results, test_cfg['ext_score_file'])
        # call the evaluator
        iou_map, mAP = det_eval.evaluate(results, verbose=True)
        f1.write(str(result_num) + ' ' + str(args.sigma) + ' ' + str(iou_map[2]) + '\n')
        f2.write(str(result_num) + ' ' + str(args.sigma) + ' ' + str(iou_map[2]) + '\n')
        with open(output_file, "wb") as f:
            pickle.dump(results, f)
    else:
        # dump to a pickle file that can be directly used for evaluation
        with open(output_file, "wb") as f:
            pickle.dump(results, f)
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
    parser.add_argument('-cls', '--cls_sigma', default=-1.0, type=float,
                        help='the parameter of Softnms')
    parser.add_argument('-i', '--iou_threshold', default=-1.0, type=float,
                        help='the parameter of Softnms')
    parser.add_argument('-min', '--min_scores', default=-1.0, type=float,
                        help='the parameter of Softnms')
    parser.add_argument('-numg', '--num_generation', default=-1.0, type=int,
                        help='the parameter of Softnms')
    parser.add_argument('-clsi', '--cls_iou_threshold', default=-1.0, type=float,
                        help='the parameter of Softnms')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        help='print frequency (default: 10 iterations)')
    args = parser.parse_args()
    main(args)
