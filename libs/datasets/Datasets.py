from torch.utils.data.dataloader import default_collate
import numpy as np
import torch
from libs.datasets.losses import ctr_giou_loss_1d,ctr_diou_loss_1d,sigmoid_focal_loss,CompletenessLoss,ClassWiseRegressionLoss
import torch.nn as nn
import os
import json
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
from libs.datasets.loc_generators import PointGenerator

from .data_utils import truncate_feats

class BufferList(nn.Module):
    """
    Similar to nn.ParameterList, but for buffers

    Taken from https://github.com/facebookresearch/detectron2/blob/master/detectron2/modeling/anchor_generator.py
    """

    def __init__(self, buffers):
        super().__init__()
        for i, buffer in enumerate(buffers):
            # Use non-persistent buffer so the values are not saved in checkpoint
            self.register_buffer(str(i), buffer, persistent=False)

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.values())
class THUMOS14Dataset(Dataset):
    def __init__(
        self,
        is_training,
        split,           # split, a tuple/list allowing concat of subsets
        data_cfg,
        model_cfg,
        train_cfg
    ):
        # file path
        assert os.path.exists(data_cfg["feat_folder"]) and os.path.exists(data_cfg["json_file"])
        assert isinstance(split, tuple) or isinstance(split, list)
        assert data_cfg["crop_ratio"] == None or len(data_cfg["crop_ratio"]) == 2
        self.feat_folder = data_cfg["feat_folder"]
        if data_cfg["file_prefix"] is not None:
            self.file_prefix = data_cfg["file_prefix"]
        else:
            self.file_prefix = ''
        self.file_ext = data_cfg["file_ext"]
        self.is_training = is_training
        if self.is_training:
            self.json_file = data_cfg["json_file"]
        else:
            self.json_file = data_cfg["test_json_file"]
        # split / training mode
        self.split = split
        # features meta info
        self.feat_stride = data_cfg["feat_stride"]
        self.num_frames = data_cfg["num_frames"]
        self.input_dim = data_cfg["input_dim"]
        self.default_fps = data_cfg["default_fps"]
        self.downsample_rate = data_cfg["downsample_rate"]
        self.max_seq_len = data_cfg["max_seq_len"]
        self.trunc_thresh = data_cfg["trunc_thresh"]
        self.label_dict = None
        self.use_offset = False
        self.crop_ratio = data_cfg["crop_ratio"]
        self.fpn_levels=model_cfg["backbone_arch"][-1]+1
        self.scale_factor=model_cfg["scale_factor"]
        self.regression_range=model_cfg["regression_range"]
        self.train_center_sample = train_cfg['center_sample']
        assert self.train_center_sample in ['radius', 'none']
        self.train_center_sample_radius = train_cfg['center_sample_radius']
        # load database and select the subset
        dict_db, label_dict = self._load_json_db(self.json_file)
        self.data_list = dict_db
        self.label_dict = label_dict
        self.points=self._generat_points()

        # dataset specific attributes
        self.db_attributes = {
            'dataset_name': 'thumos-14',
            'tiou_thresholds': np.linspace(0.3, 0.7, 5),
            # we will mask out cliff diving
            'empty_label_ids': [4],
        }
    def _generat_points(self):
        points_list = []
        # initial points
        initial_points = torch.arange(0, self.max_seq_len, 1.0)

        # loop over all points at each pyramid level
        for l in range(self.fpn_levels):
            stride = self.scale_factor ** l
            reg_range = torch.as_tensor(
                self.regression_range[l], dtype=torch.float)
            fpn_stride = torch.as_tensor(stride, dtype=torch.float)
            points = initial_points[::stride][:, None]
            # add offset if necessary (not in our current model)
            if self.use_offset:
                points += 0.5 * stride
            # pad the time stamp with additional regression range / stride
            reg_range = reg_range[None].repeat(points.shape[0], 1)
            fpn_stride = fpn_stride[None].repeat(points.shape[0], 1)
            # size: T x 4 (ts, reg_range, stride)
            points_list.append(torch.cat((points, reg_range, fpn_stride), dim=1))
        self.buffer_points=BufferList(points_list)
        feat_lens=[]
        pts_list = []
        for i in range(self.fpn_levels):
            feat_lens.append(int(self.max_seq_len/self.scale_factor ** i))
        for feat_len, buffer_pts in zip(feat_lens, self.buffer_points):
            assert feat_len <= buffer_pts.shape[0], "Reached max buffer length for point generator"
            pts = buffer_pts[:feat_len, :]
            pts_list.append(pts)
        return pts_list

    def get_attributes(self):
        return self.db_attributes

    def _load_json_db(self, json_file):
        # load database and select the subset
        with open(json_file, 'r') as fid:
            json_data = json.load(fid)
        json_db = json_data['database']

        # if label_dict is not available
        if self.label_dict is None:
            label_dict = {}
            for key, value in json_db.items():
                for act in value['annotations']:
                    label_dict[act['label']] = act['label_id']+1
                        #if key=='video_validation_0000170':
                            #print(label_dict[act['label']])

        # fill in the db (immutable afterwards)
        dict_db = tuple()
        for key, value in json_db.items():
            # skip the video if not in the split
            if value['subset'].lower() not in self.split:
                continue
            # or does not have the feature file
            if '-ad' in key:
                feat_file = os.path.join(self.feat_folder,
                                     self.file_prefix + key[:-3] + self.file_ext)
            else:
                feat_file = os.path.join(self.feat_folder,
                                         self.file_prefix + key + self.file_ext)
            if not os.path.exists(feat_file):
                continue

            # get fps if available
            if self.default_fps is not None:
                fps = self.default_fps
            elif 'fps' in value:
                fps = value['fps']
            else:
                assert False, "Unknown video FPS."

            # get video duration if available
            if 'duration' in value:
                duration = value['duration']
            else:
                duration = 1e8

            # get annotations if available
            if ('annotations' in value) and (len(value['annotations']) > 0):
                # a fun fact of THUMOS: cliffdiving (4) is a subset of diving (7)
                # we remove all cliffdiving from training and output 0 at inferenece
                # as our model can't assign two labels to the same segment
                segments, labels = [], []
                for act in value['annotations']:
                    segments.append(act['segment'])
                    if '-ad' in key:
                        labels.append([8])
                    else:
                        labels.append([label_dict[act['label']]])

                segments = np.asarray(segments, dtype=np.float32)
                labels = np.squeeze(np.asarray(labels, dtype=np.int64), axis=1)
            else:
                segments = None
                labels = None
            dict_db += ({'id': key,
                         'fps' : fps,
                         'duration' : duration,
                         'segments' : segments,
                         'labels' : labels
            }, )

        return dict_db, label_dict

    def label_points_single_video(self, points, gt_segment, gt_label):
        # concat_points : F T x 4 (t, regressoin range, stride)
        # gt_segment : N (#Events) x 2
        # gt_label : N (#Events) x 1
        concat_points = torch.cat(points, dim=0)
        num_pts = concat_points.shape[0]
        num_gts = gt_segment.shape[0]
        # corner case where current sample does not have actions
        if num_gts == 0:
            cls_targets = gt_label.new_full((num_pts,), 0.0)
            reg_targets = gt_segment.new_zeros((num_pts, 2))
            return cls_targets, reg_targets

        # compute the lengths of all segments -> F T x N
        lens = gt_segment[:, 1] - gt_segment[:, 0]
        lens = lens[None, :].repeat(num_pts, 1)

        # compute the distance of every point to each segment boundary
        # auto broadcasting for all reg target-> F T x N x2
        gt_segs = gt_segment[None].expand(num_pts, num_gts, 2)
        left = concat_points[:, 0, None] - gt_segs[:, :, 0]
        right = gt_segs[:, :, 1] - concat_points[:, 0, None]
        reg_targets = torch.stack((left, right), dim=-1)

        if self.train_center_sample == 'radius':
            # center of all segments F T x N
            center_pts = 0.5 * (gt_segs[:, :, 0] + gt_segs[:, :, 1])
            # center sampling based on stride radius
            # compute the new boundaries:
            # concat_points[:, 3] stores the stride
            t_mins = \
                center_pts - concat_points[:, 3, None] * self.train_center_sample_radius
            t_maxs = \
                center_pts + concat_points[:, 3, None] * self.train_center_sample_radius
            # prevent t_mins / maxs from over-running the action boundary
            # left: torch.maximum(t_mins, gt_segs[:, :, 0])
            # right: torch.minimum(t_maxs, gt_segs[:, :, 1])
            # F T x N (distance to the new boundary)
            cb_dist_left = concat_points[:, 0, None] \
                           - torch.maximum(t_mins, gt_segs[:, :, 0])
            cb_dist_right = torch.minimum(t_maxs, gt_segs[:, :, 1]) \
                            - concat_points[:, 0, None]
            # F T x N x 2
            center_seg = torch.stack(
                (cb_dist_left, cb_dist_right), -1)
            # F T x N
            inside_gt_seg_mask = center_seg.min(-1)[0] > 0
        else:
            # inside an gt action
            inside_gt_seg_mask = reg_targets.min(-1)[0] > 0

        # limit the regression range for each location
        max_regress_distance = reg_targets.max(-1)[0]
        # F T x N
        inside_regress_range = (
                (max_regress_distance >= concat_points[:, 1, None])
                & (max_regress_distance <= concat_points[:, 2, None])
        )

        # if there are still more than one actions for one moment
        # pick the one with the shortest duration (easiest to regress)
        lens.masked_fill_(inside_gt_seg_mask == 0, float('inf'))
        lens.masked_fill_(inside_regress_range == 0, float('inf'))
        # F T
        min_len, min_len_inds = lens.min(dim=1)

        # cls_targets: F T; reg_targets F T x 2
        cls_targets = gt_label[min_len_inds]
        # set unmatched points as BG
        cls_targets.masked_fill_(min_len == float('inf'), float(0.0))

        reg_targets = reg_targets[range(num_pts), min_len_inds]

        # normalization based on stride
        reg_targets /= concat_points[:, 3, None]

        return cls_targets, reg_targets

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # directly return a (truncated) data point (so it is very fast!)
        # auto batching will be disabled in the subsequent dataloader
        # instead the model will need to decide how to batch / preporcess the data
        video_item = self.data_list[idx]

        # load features
        if '-ad' in video_item['id']:
            filename = os.path.join(self.feat_folder,
                                self.file_prefix + video_item['id'][:-3] + self.file_ext)
        else:
            filename = os.path.join(self.feat_folder,
                                    self.file_prefix + video_item['id'] + self.file_ext)
        feats = np.load(filename).astype(np.float32)

        # deal with downsampling (= increased feat stride)
        feats = feats[::self.downsample_rate, :]
        feat_stride = self.feat_stride * self.downsample_rate
        # T x C -> C x T
        feats = torch.from_numpy(np.ascontiguousarray(feats.transpose()))

        # convert time stamp (in second) into temporal feature grids
        # ok to have small negative values here
        if video_item['segments'] is not None:
            segments = torch.from_numpy(
                (video_item['segments'] * video_item['fps'] - 0.5 * self.num_frames) / feat_stride
            )
            labels = torch.from_numpy(video_item['labels'])
        else:
            segments, labels = None, None
        # return a data dict
        data_dict = {'video_id'        : video_item['id'],
                     'feats'           : feats,      # C x T
                     'segments'        : segments,   # N x 2
                     'labels'          : labels,     # N
                     'fps'             : video_item['fps'],
                     'duration'        : video_item['duration'],
                     'feat_stride'     : feat_stride,
                     'feat_num_frames' : self.num_frames}

        # truncate the features during training
        if self.is_training and (segments is not None):
            data_dict = truncate_feats(
                data_dict, self.max_seq_len, self.trunc_thresh, self.crop_ratio
            )
        gt_cls_labels, gt_offsets = self.label_points_single_video(
            self.points, data_dict['segments'], data_dict['labels'])


        if self.is_training:
            return data_dict['feats'],gt_cls_labels,gt_offsets,data_dict['segments'],data_dict['labels']
        else:
            return data_dict

def mt_collate_fn(batch):
    max_seq_len=2304
    max_gt = 0
    for b in batch:
        if b[3].shape[0] > max_gt:
            max_gt = b[3].shape[0]

    "Pads data and puts it into a tensor of same dimensions"
    batch.sort(key=lambda data:len(data[0][0]),reverse=True)
    max_len = batch[0][0].shape[1]
    assert max_len <= max_seq_len, "Input length must be smaller than max_seq_len during training"
    max_len=max_seq_len
    new_batch = []
    for b in batch:
        f = np.zeros((b[0].shape[0],max_len), np.float32)
        m = np.zeros((max_len), np.float32)
        f[:,:b[0].shape[1]] = b[0]
        m[:b[0].shape[1]] = 1
        f_t=torch.ones(max_gt,b[3].shape[1])
        f_t[:,0]=10000.0
        f_t[:,1]=20000.0
        m_t=torch.zeros(max_gt)
        l_t=torch.ones(max_gt)*-1
        f_t[:b[3].shape[0]]=b[3]
        m_t[:b[3].shape[0]] = 1
        l_t[:b[3].shape[0]] = b[4]
        new_batch.append([[[f, torch.from_numpy(m),f_t,m_t]], [[b[1], b[2],f_t,l_t,m_t]]])
    return default_collate(new_batch)

class FormerLoss(nn.Module):
    def __init__(self,train_cfg,data_cfg):
        super(FormerLoss,self).__init__()
        self.train_label_smoothing=train_cfg["label_smoothing"]
        self.loss_normalizer = train_cfg['init_loss_norm']
        self.loss_normalizer_momentum = 0.9
        self.train_loss_weight = train_cfg['loss_weight']
        self.cls_criterion = torch.nn.CrossEntropyLoss()
        self.fg_iou=0.7
        self.bg_iou=0.01
        self.sample_ratio=1
    def losses(
        self, fpn_masks,
        out_cls_logits, out_offsets,
        gt_cls, gt_offsets
    ):
        # fpn_masks, out_*: F (List) [B, T_i, C]
        # gt_* : B (list) [F T, C]
        # fpn_masks -> (B, FT)
        valid_mask = torch.cat(fpn_masks, dim=1)

        # 1. classification loss
        # stack the list -> (B, FT) -> (# Valid, )
        pos_mask = (gt_cls >= 0) & (gt_cls != 0.0) & valid_mask

        # cat the predicted offsets -> (B, FT, 2 (xC)) -> # (#Pos, 2 (xC))
        pred_offsets = torch.cat(out_offsets, dim=1)[pos_mask]
        gt_offsets = gt_offsets[pos_mask]

        # update the loss normalizer
        num_pos = pos_mask.sum().item()
        self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + (
            1 - self.loss_normalizer_momentum
        ) * max(num_pos, 1)

        # #cls + 1 (background)
        gt_target = gt_cls[valid_mask]
        gt_target=gt_target>0
        gt_target = gt_target.to(out_cls_logits[0].dtype)
        #print(gt_target[gt_target>0])
        # focal loss
        cls_loss = sigmoid_focal_loss(
            torch.cat(out_cls_logits, dim=1)[valid_mask],
            gt_target,
            reduction='sum'
        )
        cls_loss /= self.loss_normalizer

        # 2. regression using IoU/GIoU loss (defined on positive samples)
        if num_pos == 0:
            reg_loss = 0 * pred_offsets.sum()
        else:
            # giou loss defined on positive samples
            reg_loss = ctr_diou_loss_1d(
                pred_offsets,
                gt_offsets,
                reduction='sum'
            )
            reg_loss /= self.loss_normalizer

        if self.train_loss_weight > 0:
            loss_weight = self.train_loss_weight
        else:
            loss_weight = cls_loss.detach() / max(reg_loss.item(), 0.01)

        # return a dict of losses
        final_loss = cls_loss + reg_loss * loss_weight
        return final_loss
    def _get_proposal_traget(self,gt_segments,rois,gt_label,gt_mask):
        rois=rois[:,1:]
        gt_segments=gt_segments.unsqueeze(0).repeat(rois.size()[0],1,1)
        rois=rois.unsqueeze(1).repeat(1,gt_segments.size()[1],1)
        min_left=torch.min(gt_segments[:,:,0],rois[:,:,0])
        max_left = torch.max(gt_segments[:, :, 0], rois[:, :, 0])
        min_right = torch.min(gt_segments[ :, :, 1], rois[ :, :, 1])
        max_right = torch.max(gt_segments[ :, :, 1], rois[ :, :, 1])
        ious=(min_right-max_left)/(max_right-min_left)
        ious,iou_idx=ious.max(dim=1)
        iou_labels=gt_label[iou_idx]
        return ious,iou_labels


    def _classify_proposal_loss(self,rois,scores,roi_mask,gt_segments,gt_labels,gt_mask,cls_log):
        bacth_size=rois.size()[0]
        labels_all=[]
        cls_all=[]
        for j in range(bacth_size):
            ious,iou_labels = self._get_proposal_traget(gt_segments[j], rois[j], gt_labels[j], gt_mask[j])
            pos_pro = (ious > self.fg_iou)
            iou_labels=iou_labels*pos_pro
            pos_idx = pos_pro.nonzero(as_tuple=True)[0]
            bg_pro = (ious < self.bg_iou) * (scores[j] > 0)
            bg_idxs = bg_pro.nonzero(as_tuple=True)[0]
            bg_idxs=bg_idxs[:self.sample_ratio*pos_idx.size()[0]]
            idx=torch.cat((pos_idx,bg_idxs),dim=0)
            labels=iou_labels[idx]
            sub_cls=cls_log[j][idx,:]
            labels_all.append(labels)
            cls_all.append(sub_cls)
        labels_all=torch.cat(labels_all)
        labels_all=labels_all.long()
        cls_all=torch.cat(cls_all)
        act_loss = self.cls_criterion(cls_all, labels_all)

        return act_loss

    def forward(self,outputs,targets):
        gt_cls_labels=targets[0]
        gt_offsets=targets[1]
        gt_segments=targets[2]
        segments_label=targets[3]
        segments_mask = targets[4]


        fpn_masks, out_cls_logits, out_offsets,out_rois,out_scores,out_roimask,cls_log=outputs[0],outputs[1],outputs[2],outputs[3],outputs[4],outputs[5],outputs[6]
        cls_loss=self._classify_proposal_loss(rois=out_rois,scores=out_scores,roi_mask=out_roimask,gt_segments=gt_segments,gt_labels=segments_label,gt_mask=segments_mask,cls_log=cls_log)

        losses = self.losses(
            fpn_masks,
            out_cls_logits, out_offsets,
            gt_cls_labels, gt_offsets
        )
        all_loss=losses+cls_loss

        return all_loss



class FormerLoss_com(nn.Module):
    def __init__(self,train_cfg,data_cfg):
        super(FormerLoss_com,self).__init__()
        self.train_label_smoothing=train_cfg["label_smoothing"]
        self.loss_normalizer = train_cfg['init_loss_norm']
        self.loss_normalizer_momentum = 0.9
        self.train_loss_weight = train_cfg['loss_weight']
        self.cls_criterion = torch.nn.CrossEntropyLoss()
        self.com_criterion = CompletenessLoss()
        self.fg_iou=0.7
        self.bg_iou=0.01
        self.com_iou=0.3
        self.sample_ratio=6
    def losses(
        self, fpn_masks,
        out_cls_logits, out_offsets,
        gt_cls, gt_offsets
    ):
        # fpn_masks, out_*: F (List) [B, T_i, C]
        # gt_* : B (list) [F T, C]
        # fpn_masks -> (B, FT)
        valid_mask = torch.cat(fpn_masks, dim=1)

        # 1. classification loss
        # stack the list -> (B, FT) -> (# Valid, )
        pos_mask = (gt_cls >= 0) & (gt_cls != 0.0) & valid_mask

        # cat the predicted offsets -> (B, FT, 2 (xC)) -> # (#Pos, 2 (xC))
        pred_offsets = torch.cat(out_offsets, dim=1)[pos_mask]
        gt_offsets = gt_offsets[pos_mask]

        # update the loss normalizer
        num_pos = pos_mask.sum().item()
        self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + (
            1 - self.loss_normalizer_momentum
        ) * max(num_pos, 1)

        # #cls + 1 (background)
        gt_target = gt_cls[valid_mask]
        gt_target=gt_target>0
        gt_target = gt_target.to(out_cls_logits[0].dtype)
        #print(gt_target[gt_target>0])
        # focal loss
        cls_loss = sigmoid_focal_loss(
            torch.cat(out_cls_logits, dim=1)[valid_mask],
            gt_target,
            reduction='sum'
        )
        cls_loss /= self.loss_normalizer

        # 2. regression using IoU/GIoU loss (defined on positive samples)
        if num_pos == 0:
            reg_loss = 0 * pred_offsets.sum()
        else:
            # giou loss defined on positive samples
            reg_loss = ctr_giou_loss_1d(
                pred_offsets,
                gt_offsets,
                reduction='sum'
            )
            reg_loss /= self.loss_normalizer

        if self.train_loss_weight > 0:
            loss_weight = self.train_loss_weight
        else:
            loss_weight = cls_loss.detach() / max(reg_loss.item(), 0.01)

        # return a dict of losses
        final_loss = cls_loss + reg_loss * loss_weight
        return final_loss
    def _get_proposal_traget(self,gt_segments,rois,gt_label,gt_mask):
        rois=rois[:,1:]
        gt_segments=gt_segments.unsqueeze(0).repeat(rois.size()[0],1,1)
        rois=rois.unsqueeze(1).repeat(1,gt_segments.size()[1],1)
        min_left=torch.min(gt_segments[:,:,0],rois[:,:,0])
        #print(min_left.size())
        max_left = torch.max(gt_segments[:, :, 0], rois[:, :, 0])
        min_right = torch.min(gt_segments[ :, :, 1], rois[ :, :, 1])
        max_right = torch.max(gt_segments[ :, :, 1], rois[ :, :, 1])
        ious=(min_right-max_left)/(max_right-min_left)
        #print(ious.size())
        ious,iou_idx=ious.max(dim=1)
        iou_labels=gt_label[iou_idx]
        return ious,iou_labels


    def _classify_proposal_loss(self,rois,scores,gt_segments,gt_labels,gt_mask,cls_log,com_log):
        bacth_size=rois.size()[0]
        labels_all=[]
        cls_all=[]
        com_labels_all = []
        com_all = []
        for j in range(bacth_size):
            ious,iou_labels = self._get_proposal_traget(gt_segments[j], rois[j], gt_labels[j], gt_mask[j])
            pos_pro = (ious > self.fg_iou)
            iou_labels=iou_labels*pos_pro
            pos_idx = pos_pro.nonzero(as_tuple=True)[0]
            bg_pro = (ious < self.bg_iou) * (scores[j] > 0)
            com_pro= (ious < self.com_iou) * (scores[j] > 0)
            bg_idxs = bg_pro.nonzero(as_tuple=True)[0]
            com_idxs=com_pro.nonzero(as_tuple=True)[0]
            com_idxs=com_idxs[:max(1,self.sample_ratio*pos_idx.size()[0])]
            bg_idxs = bg_idxs[: pos_idx.size()[0]]
            idx=torch.cat((pos_idx,bg_idxs),dim=0)
            idx_com=torch.cat((pos_idx,com_idxs),dim=0)
            labels=iou_labels[idx]
            sub_cls=cls_log[j][idx,:]
            labels_com = iou_labels[idx_com]
            sub_com = com_log[j][idx_com, :]
            labels_all.append(labels)
            cls_all.append(sub_cls)
            com_labels_all.append(labels_com)
            com_all.append(sub_com)
        labels_all=torch.cat(labels_all)
        labels_all=labels_all.long()
        cls_all=torch.cat(cls_all)
        com_labels_all = torch.cat(com_labels_all)
        com_labels_all = com_labels_all.long()
        comp_group_size=com_labels_all.size()[0]
        ehem_num=(com_labels_all.nonzero(as_tuple=True)[0]).size()[0]
        #print(comp_group_size,ehem_num)
        com_all = torch.cat(com_all)
        act_loss = self.cls_criterion(cls_all, labels_all)

        comp_loss = self.com_criterion(com_all,com_labels_all, ehem_num,comp_group_size)
        loss=act_loss+0.5*comp_loss
        return loss

    def forward(self,outputs,targets):
        gt_cls_labels=targets[0]
        gt_offsets=targets[1]
        gt_segments=targets[2]
        segments_label=targets[3]
        segments_mask = targets[4]
        fpn_masks, out_cls_logits, out_offsets,out_rois,out_scores,out_roimask,cls_log,com_log=outputs[0],outputs[1],outputs[2],outputs[3],outputs[4],outputs[5],outputs[6],outputs[7]
        cls_loss=self._classify_proposal_loss(rois=out_rois,scores=out_scores,gt_segments=gt_segments,gt_labels=segments_label,gt_mask=segments_mask,cls_log=cls_log,com_log=com_log)

        losses = self.losses(
            fpn_masks,
            out_cls_logits, out_offsets,
            gt_cls_labels, gt_offsets
        )
        all_loss=losses+cls_loss

        return all_loss

class FormerLoss_reg(nn.Module):
    def __init__(self,train_cfg,data_cfg):
        super(FormerLoss_reg,self).__init__()
        self.train_label_smoothing=train_cfg["label_smoothing"]
        self.loss_normalizer = train_cfg['init_loss_norm']
        self.loss_normalizer_momentum = 0.9
        self.train_loss_weight = train_cfg['loss_weight']
        self.cls_criterion = torch.nn.CrossEntropyLoss()
        self.num_classes=data_cfg["num_classes"]
        self.com_criterion = CompletenessLoss()
        self.reg_off=ClassWiseRegressionLoss()
        self.fg_iou=0.7
        self.bg_iou=0.01
        self.com_iou=0.3
        self.sample_ratio=6
    def losses(
        self, fpn_masks,
        out_cls_logits, out_offsets,
        gt_cls, gt_offsets
    ):
        # fpn_masks, out_*: F (List) [B, T_i, C]
        # gt_* : B (list) [F T, C]
        # fpn_masks -> (B, FT)
        valid_mask = torch.cat(fpn_masks, dim=1)

        # 1. classification loss
        # stack the list -> (B, FT) -> (# Valid, )
        pos_mask = (gt_cls >= 0) & (gt_cls != 0.0) & valid_mask

        # cat the predicted offsets -> (B, FT, 2 (xC)) -> # (#Pos, 2 (xC))
        pred_offsets = torch.cat(out_offsets, dim=1)[pos_mask]
        gt_offsets = gt_offsets[pos_mask]

        # update the loss normalizer
        num_pos = pos_mask.sum().item()
        self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + (
            1 - self.loss_normalizer_momentum
        ) * max(num_pos, 1)

        # #cls + 1 (background)
        gt_target = gt_cls[valid_mask]
        gt_target=gt_target>0
        gt_target = gt_target.to(out_cls_logits[0].dtype)
        #print(gt_target[gt_target>0])
        # focal loss
        cls_loss = sigmoid_focal_loss(
            torch.cat(out_cls_logits, dim=1)[valid_mask],
            gt_target,
            reduction='sum'
        )
        cls_loss /= self.loss_normalizer

        # 2. regression using IoU/GIoU loss (defined on positive samples)
        if num_pos == 0:
            reg_loss = 0 * pred_offsets.sum()
        else:
            # giou loss defined on positive samples
            reg_loss = ctr_giou_loss_1d(
                pred_offsets,
                gt_offsets,
                reduction='sum'
            )
            reg_loss /= self.loss_normalizer

        if self.train_loss_weight > 0:
            loss_weight = self.train_loss_weight
        else:
            loss_weight = cls_loss.detach() / max(reg_loss.item(), 0.01)

        # return a dict of losses
        final_loss = cls_loss + reg_loss * loss_weight
        return final_loss
    def _get_proposal_traget(self,gt_segments_org,rois,gt_label,gt_mask):
        rois_org = rois[:, 1:]
        ids = gt_mask > 0
        gt_segments_org = gt_segments_org[ids]
        gt_label = gt_label[ids]
        gt_segments = gt_segments_org.unsqueeze(0).repeat(rois.size()[0], 1, 1)
        rois = rois_org.unsqueeze(1).repeat(1, gt_segments.size()[1], 1)
        min_left = torch.min(gt_segments[:, :, 0], rois[:, :, 0])
        max_left = torch.max(gt_segments[:, :, 0], rois[:, :, 0])
        min_right = torch.min(gt_segments[:, :, 1], rois[:, :, 1])
        max_right = torch.max(gt_segments[:, :, 1], rois[:, :, 1])
        ious = (min_right - max_left) / (max_right - min_left)
        ious, iou_idx = ious.max(dim=1)
        iou_labels = gt_label[iou_idx]
        gt_sub = gt_segments_org[iou_idx]
        gt_center = (gt_sub[:, 0] + gt_sub[:, 1]) / 2.0
        gt_size = gt_sub[:, 1] - gt_sub[:, 0]
        roi_center = (rois_org[:, 1] + rois_org[:, 0]) / 2.0
        roi_size = (rois_org[:, 1] - rois_org[:, 0])
        loc_reg = (gt_center - roi_center) / roi_size
        size_reg = torch.log(gt_size / roi_size)
        roi_offsets = torch.stack((loc_reg, size_reg), dim=1)
        return ious, iou_labels, roi_offsets


    def _classify_proposal_loss(self,rois,scores,gt_segments,gt_labels,gt_mask,cls_log,com_log,res_log):
        bacth_size=rois.size()[0]
        labels_all=[]
        cls_all=[]
        com_labels_all = []
        com_all = []
        offsets_all = []
        offsets_label_all=[]
        offsets_log_all=[]
        for j in range(bacth_size):
            ious,iou_labels,roi_offsets = self._get_proposal_traget(gt_segments[j], rois[j], gt_labels[j], gt_mask[j])
            pos_pro = (ious > self.fg_iou)
            iou_labels=iou_labels*pos_pro
            pos_idx = pos_pro.nonzero(as_tuple=True)[0]
            bg_pro = (ious < self.bg_iou) * (scores[j] > 0)
            com_pro= (ious < self.com_iou) * (scores[j] > 0)
            bg_idxs = bg_pro.nonzero(as_tuple=True)[0]
            com_idxs=com_pro.nonzero(as_tuple=True)[0]
            com_idxs=com_idxs[:max(1,self.sample_ratio*pos_idx.size()[0])]
            bg_idxs = bg_idxs[: pos_idx.size()[0]]
            idx=torch.cat((pos_idx,bg_idxs),dim=0)
            idx_com=torch.cat((pos_idx,com_idxs),dim=0)
            labels=iou_labels[idx]
            off_label=iou_labels[pos_idx]
            offsets_label_all.append(off_label)
            sub_cls=cls_log[j][idx,:]
            sub_off = roi_offsets[pos_idx]
            offsets_all.append(sub_off)
            sub_off_log=res_log[j][pos_idx]
            offsets_log_all.append(sub_off_log)
            labels_com = iou_labels[idx_com]
            sub_com = com_log[j][idx_com, :]
            labels_all.append(labels)
            cls_all.append(sub_cls)
            com_labels_all.append(labels_com)
            com_all.append(sub_com)
        labels_all=torch.cat(labels_all)
        labels_all=labels_all.long()
        cls_all=torch.cat(cls_all)
        com_labels_all = torch.cat(com_labels_all)
        offsets_all = torch.cat(offsets_all)
        offsets_label_all=torch.cat(offsets_label_all)
        offsets_log_all=torch.cat(offsets_log_all)
        offsets_label_all=offsets_label_all.long()
        com_labels_all = com_labels_all.long()
        comp_group_size=com_labels_all.size()[0]
        ehem_num=(com_labels_all.nonzero(as_tuple=True)[0]).size()[0]
        #print(comp_group_size,ehem_num)
        com_all = torch.cat(com_all)
        act_loss = self.cls_criterion(cls_all, labels_all)

        comp_loss = self.com_criterion(com_all,com_labels_all, ehem_num,comp_group_size)
        offsets_log_all=offsets_log_all.view(-1,self.num_classes,2)
        reg_loss=self.reg_off(offsets_log_all,offsets_label_all,offsets_all)
        #print(offsets_log_all.size(),offsets_label_all.size(),offsets_all.size())
        loss=0.5*act_loss+comp_loss+20*reg_loss
        #print(act_loss,comp_loss,reg_loss)
        return loss

    def forward(self,outputs,targets):
        gt_cls_labels=targets[0]
        gt_offsets=targets[1]
        gt_segments=targets[2]
        segments_label=targets[3]
        segments_mask = targets[4]
        fpn_masks, out_cls_logits, out_offsets,out_rois,out_scores,out_roimask,cls_log,com_log,reg_log=outputs[0],outputs[1],outputs[2],outputs[3],outputs[4],outputs[5],outputs[6],outputs[7],outputs[8]
        cls_loss=self._classify_proposal_loss(rois=out_rois,scores=out_scores,gt_segments=gt_segments,gt_labels=segments_label,gt_mask=segments_mask,cls_log=cls_log,com_log=com_log,res_log=reg_log)

        losses = self.losses(
            fpn_masks,
            out_cls_logits, out_offsets,
            gt_cls_labels, gt_offsets
        )
        all_loss=losses+cls_loss
        #print(losses)

        return all_loss


def feature_measure(gt,node_feat,weight_type ='counsine',alte=3):
    chinnal=gt.size()[-1]
    gt=gt.view(-1,chinnal)
    node_feat=node_feat.view(-1,chinnal)
    epsilon = 1e-8
    if weight_type == 'counsine':
        kernel_weigths =1.0- (torch.cosine_similarity(gt, node_feat, dim=1)+1.0)/2.0

    else:
        if weight_type == 'euclidean':
            x = gt - node_feat
            x = x * x
            x = torch.sum(x, dim=1)
        elif weight_type == 'lance':
            x1 = torch.abs(gt - node_feat)
            x2 = torch.abs(gt + node_feat)
            x = x1 / (x2 + epsilon)
            x = torch.mean(x, dim=1)
        elif weight_type == 'Manhattan':
            x = torch.abs(gt - node_feat)
            x = torch.sum(x, dim=1)
        elif weight_type == 'chebyshev':
            x = torch.abs(gt - node_feat)
            x = torch.max(x, dim=1)[0]
        elif weight_type == 'Minkowski':
            x = torch.norm(gt - node_feat, p=alte, dim=1, keepdim=False)
        kernel_weigths = 1.0-torch.exp_(x.mul_(-0.5))
    return kernel_weigths
def score_loss(sorce,label):
    m=0.9
    loss1=sorce*label/label.sum()+(1-label)*F.relu(m-sorce)/(1-label).sum()
    loss=loss1.sum()
    return loss
def whole_scorce_loss(act_scorce1,com_scorce1,label_act,label_comp):
    alta=0.5
    label_act=(label_act>0).float()
    act_loss1=score_loss(act_scorce1,label_act)
    com_loss1 = score_loss(com_scorce1, label_comp)
    loss=act_loss1+alta*com_loss1
    return loss
def whole_scorce_loss_1(act_scorce1,com_scorce1,label_act,label_comp):
    alta=0.5

    loss=act_scorce1.sum()+alta*com_scorce1.sum()
    return loss

class FormerLoss_metirc(nn.Module):
    def __init__(self,train_cfg,data_cfg):
        super(FormerLoss_metirc,self).__init__()
        self.train_label_smoothing=train_cfg["label_smoothing"]
        self.loss_normalizer = train_cfg['init_loss_norm']
        self.loss_normalizer_momentum = 0.9
        self.train_loss_weight = train_cfg['loss_weight']
        self.cls_criterion = torch.nn.CrossEntropyLoss()
        self.num_classes=data_cfg["num_classes"]
        self.com_criterion = CompletenessLoss()
        self.fg_iou=0.7
        self.bg_iou=0.01
        self.com_iou=0.3
        self.sample_ratio=6
    def losses(
        self, fpn_masks,
        out_cls_logits, out_offsets,
        gt_cls, gt_offsets
    ):
        # fpn_masks, out_*: F (List) [B, T_i, C]
        # gt_* : B (list) [F T, C]
        # fpn_masks -> (B, FT)
        valid_mask = torch.cat(fpn_masks, dim=1)

        # 1. classification loss
        # stack the list -> (B, FT) -> (# Valid, )
        pos_mask = (gt_cls >= 0) & (gt_cls != 0.0) & valid_mask

        # cat the predicted offsets -> (B, FT, 2 (xC)) -> # (#Pos, 2 (xC))
        pred_offsets = torch.cat(out_offsets, dim=1)[pos_mask]
        gt_offsets = gt_offsets[pos_mask]

        # update the loss normalizer
        num_pos = pos_mask.sum().item()
        self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + (
            1 - self.loss_normalizer_momentum
        ) * max(num_pos, 1)

        # #cls + 1 (background)
        gt_target = gt_cls[valid_mask]
        gt_target=gt_target>0
        gt_target = gt_target.to(out_cls_logits[0].dtype)
        #print(gt_target[gt_target>0])
        # focal loss
        cls_loss = sigmoid_focal_loss(
            torch.cat(out_cls_logits, dim=1)[valid_mask],
            gt_target,
            reduction='sum'
        )
        cls_loss /= self.loss_normalizer

        # 2. regression using IoU/GIoU loss (defined on positive samples)
        if num_pos == 0:
            reg_loss = 0 * pred_offsets.sum()
        else:
            # giou loss defined on positive samples
            reg_loss = ctr_giou_loss_1d(
                pred_offsets,
                gt_offsets,
                reduction='sum'
            )
            reg_loss /= self.loss_normalizer

        if self.train_loss_weight > 0:
            loss_weight = self.train_loss_weight
        else:
            loss_weight = cls_loss.detach() / max(reg_loss.item(), 0.01)

        # return a dict of losses
        final_loss = cls_loss + reg_loss * loss_weight
        return final_loss
    def _get_proposal_traget(self,gt_segments_org,rois,gt_label,gt_mask,cls_gt,cls_node,com_gt,com_node):
        rois_org = rois[:, 1:]
        ids = gt_mask > 0
        gt_segments_org = gt_segments_org[ids]
        #print(gt_segments_org.size(),cls_gt.size())
        gt_label = gt_label[ids]
        gt_segments = gt_segments_org.unsqueeze(0).repeat(rois.size()[0], 1, 1)
        rois = rois_org.unsqueeze(1).repeat(1, gt_segments.size()[1], 1)
        min_left = torch.min(gt_segments[:, :, 0], rois[:, :, 0])
        max_left = torch.max(gt_segments[:, :, 0], rois[:, :, 0])
        min_right = torch.min(gt_segments[:, :, 1], rois[:, :, 1])
        max_right = torch.max(gt_segments[:, :, 1], rois[:, :, 1])
        ious = (min_right - max_left) / (max_right - min_left)
        ious, iou_idx = ious.max(dim=1)

        cls_gt_ious=cls_gt[iou_idx]
        cls_distance=feature_measure(cls_gt_ious,cls_node)
        com_gt_ious=com_gt[iou_idx]
        com_distance=feature_measure(com_gt_ious,com_node)

        iou_labels = gt_label[iou_idx]
        return ious, iou_labels, cls_distance,com_distance


    def _classify_proposal_loss(self,rois,scores,gt_segments,gt_labels,gt_mask,cls_log,com_log,cls_gt,cls_node,com_gt,com_node):
        bacth_size=rois.size()[0]
        labels_all=[]
        cls_all=[]
        com_labels_all = []
        com_all = []
        cls_distance_all=[]
        com_distance_all=[]
        for j in range(bacth_size):
            ious,iou_labels,cls_distance,com_distance = self._get_proposal_traget(gt_segments[j], rois[j], gt_labels[j], gt_mask[j],cls_gt[j],cls_node[j],com_gt[j],com_node[j])
            #pos_pro = (ious > self.fg_iou)
            '''
            ious, iou_labels = self._get_proposal_traget(gt_segments[j], rois[j],
                                                                                     gt_labels[j], gt_mask[j],
                                                                                     cls_gt[j], cls_node[j], com_gt[j],
                                                                                     com_node[j])
            '''
            pos_pro = (ious > self.fg_iou)
            iou_labels=iou_labels*pos_pro
            pos_idx = pos_pro.nonzero(as_tuple=True)[0]
            bg_pro = (ious < self.bg_iou) * (scores[j] > 0)*(ious>0)
            com_pro= (ious < self.com_iou) * (scores[j] > 0)*(ious>0)
            bg_idxs = bg_pro.nonzero(as_tuple=True)[0]
            com_idxs=com_pro.nonzero(as_tuple=True)[0]
            com_idxs=com_idxs[:max(1,self.sample_ratio*pos_idx.size()[0])]
            bg_idxs = bg_idxs[: pos_idx.size()[0]]
            idx=torch.cat((pos_idx,bg_idxs),dim=0)
            idx_com=torch.cat((pos_idx,com_idxs),dim=0)
            labels=iou_labels[idx]
            sub_cls_dis=cls_distance[pos_idx]
            cls_distance_all.append(sub_cls_dis)
            sub_cls=cls_log[j][idx,:]
            labels_com = iou_labels[idx_com]
            sub_com_dis=com_distance[pos_idx]
            com_distance_all.append(sub_com_dis)
            sub_com = com_log[j][idx_com, :]
            labels_all.append(labels)
            cls_all.append(sub_cls)
            com_labels_all.append(labels_com)
            com_all.append(sub_com)
        labels_all=torch.cat(labels_all)
        labels_all=labels_all.long()
        cls_all=torch.cat(cls_all)
        com_labels_all = torch.cat(com_labels_all)
        com_labels_all = com_labels_all.long()
        cls_distance_all=torch.cat(cls_distance_all)
        com_distance_all=torch.cat(com_distance_all)
        dis_loss=whole_scorce_loss_1(act_scorce1=cls_distance_all,com_scorce1=com_distance_all,label_act=labels_all,label_comp=com_labels_all)
        comp_group_size=com_labels_all.size()[0]
        ehem_num=(com_labels_all.nonzero(as_tuple=True)[0]).size()[0]

        com_all = torch.cat(com_all)
        act_loss = self.cls_criterion(cls_all, labels_all)

        comp_loss = self.com_criterion(com_all,com_labels_all, ehem_num,comp_group_size)
        loss=act_loss+0.5*comp_loss#+1*dis_loss

        #print(act_loss,comp_loss,dis_loss)
        return loss

    def forward(self,outputs,targets):
        gt_cls_labels=targets[0]
        gt_offsets=targets[1]
        gt_segments=targets[2]
        segments_label=targets[3]
        segments_mask = targets[4]
        fpn_masks, out_cls_logits, out_offsets,out_rois,out_scores,out_roimask,cls_log,com_log,cls_gt,cls_node,com_gt,com_node=outputs[0],outputs[1],outputs[2],outputs[3],outputs[4],outputs[5],outputs[6],outputs[7],outputs[8],outputs[9],outputs[10],outputs[11]
        cls_loss=self._classify_proposal_loss(rois=out_rois,scores=out_scores,gt_segments=gt_segments,gt_labels=segments_label,gt_mask=segments_mask,cls_log=cls_log,com_log=com_log,cls_gt=cls_gt,cls_node=cls_node,com_gt=com_gt,com_node=com_node)

        losses = self.losses(
            fpn_masks,
            out_cls_logits, out_offsets,
            gt_cls_labels, gt_offsets
        )
        all_loss=losses+cls_loss
        #print(losses)

        return all_loss
