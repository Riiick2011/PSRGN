import math
import torch
import torch.nn.functional as F
from torch import nn
from libs.utils.weight_init import trunc_normal_
from libs.modelings.Backbone import ConvBackbone, ConvTransformerBackbone
from libs.modelings.Neck import FPNIdentity, FPN1D
from libs.modelings.Block import MaskedConv1D, LayerNorm, Scale
from libs.modelings.loc_generators import PointGenerator
from libs.utils.creator_tool import ProposalCreator
from torchvision.ops import RoIPool, RoIAlign
from libs.modelings.test_layer_attention_spatial_metric import Graph_module_net_0, Graph_module_net_0_loss, \
    Graph_module_net_0_loss_type_2, Graph_module_net_0_loss_type, Graph_module_net_0_loss_2
import numpy as np
from libs.datasets.Datasets import BufferList


class PtTransformerClsHead(nn.Module):
    """
    1D Conv heads for classification
    """

    def __init__(
            self,
            input_dim,
            feat_dim,
            num_classes,
            prior_prob=0.01,
            num_layers=3,
            kernel_size=3,
            act_layer=nn.ReLU,
            with_ln=False,
            empty_cls=[]
    ):
        super().__init__()
        self.act = act_layer()

        # build the head
        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()
        for idx in range(num_layers - 1):
            if idx == 0:
                in_dim = input_dim
                out_dim = feat_dim
            else:
                in_dim = feat_dim
                out_dim = feat_dim
            self.head.append(
                MaskedConv1D(
                    in_dim, out_dim, kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                    bias=(not with_ln)
                )
            )
            if with_ln:
                self.norm.append(
                    LayerNorm(out_dim)
                )
            else:
                self.norm.append(nn.Identity())

        # classifier
        self.cls_head = MaskedConv1D(
            feat_dim, num_classes, kernel_size,
            stride=1, padding=kernel_size // 2
        )

        # use prior in model initialization to improve stability
        # this will overwrite other weight init
        bias_value = -(math.log((1 - prior_prob) / prior_prob))
        torch.nn.init.constant_(self.cls_head.conv.bias, bias_value)

        # a quick fix to empty categories:
        # the weights assocaited with these categories will remain unchanged
        # we set their bias to a large negative value to prevent their outputs
        if len(empty_cls) > 0:
            bias_value = -(math.log((1 - 1e-6) / 1e-6))
            for idx in empty_cls:
                torch.nn.init.constant_(self.cls_head.conv.bias[idx], bias_value)

    def forward(self, fpn_feats, fpn_masks):
        assert len(fpn_feats) == len(fpn_masks)

        # apply the classifier for each pyramid level
        out_logits = tuple()
        for _, (cur_feat, cur_mask) in enumerate(zip(fpn_feats, fpn_masks)):
            cur_out = cur_feat
            for idx in range(len(self.head)):
                cur_out, _ = self.head[idx](cur_out, cur_mask)
                cur_out = self.act(self.norm[idx](cur_out))
            cur_logits, _ = self.cls_head(cur_out, cur_mask)
            out_logits += (cur_logits,)

        # fpn_masks remains the same
        return out_logits


class MLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(MLP, self).__init__()

        self.gc1 = nn.Linear(nfeat, nhid)
        self.gc2 = nn.Linear(nhid, nclass)
        self.dropout = dropout

    def forward(self, x):
        x = F.relu(self.gc1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x)
        # x = F.relu(self.gc2(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)
        return x


class PtTransformerRegHead(nn.Module):
    """
    Shared 1D Conv heads for regression
    Simlar logic as PtTransformerClsHead with separated implementation for clarity
    """

    def __init__(
            self,
            input_dim,
            feat_dim,
            fpn_levels,
            num_layers=3,
            kernel_size=3,
            act_layer=nn.ReLU,
            with_ln=False
    ):
        super().__init__()
        self.fpn_levels = fpn_levels
        self.act = act_layer()

        # build the conv head
        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()
        for idx in range(num_layers - 1):
            if idx == 0:
                in_dim = input_dim
                out_dim = feat_dim
            else:
                in_dim = feat_dim
                out_dim = feat_dim
            self.head.append(
                MaskedConv1D(
                    in_dim, out_dim, kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                    bias=(not with_ln)
                )
            )
            if with_ln:
                self.norm.append(
                    LayerNorm(out_dim)
                )
            else:
                self.norm.append(nn.Identity())

        self.scale = nn.ModuleList()
        for idx in range(fpn_levels):
            self.scale.append(Scale())

        # segment regression
        self.offset_head = MaskedConv1D(
            feat_dim, 2, kernel_size,
            stride=1, padding=kernel_size // 2
        )

    def forward(self, fpn_feats, fpn_masks):
        assert len(fpn_feats) == len(fpn_masks)
        assert len(fpn_feats) == self.fpn_levels

        # apply the classifier for each pyramid level
        out_offsets = tuple()
        for l, (cur_feat, cur_mask) in enumerate(zip(fpn_feats, fpn_masks)):
            cur_out = cur_feat
            for idx in range(len(self.head)):
                cur_out, _ = self.head[idx](cur_out, cur_mask)
                cur_out = self.act(self.norm[idx](cur_out))
            cur_offsets, _ = self.offset_head(cur_out, cur_mask)
            out_offsets += (F.relu(self.scale[l](cur_offsets)),)

        # fpn_masks remains the same
        return out_offsets



class PtTransformer_gcn_loss_type(nn.Module):
    """
        Transformer based model for single stage action localization
    """

    def __init__(
            self,
            model_configs,
            train_cfg,
    ):
        super().__init__()
        # re-distribute params to backbone / neck / head
        self.fpn_strides = [model_configs["scale_factor"] ** i for i in range(model_configs["backbone_arch"][-1] + 1)]
        self.reg_range = model_configs["regression_range"]
        assert len(self.fpn_strides) == len(self.reg_range)
        self.scale_factor = model_configs["scale_factor"]
        # check the feature pyramid and local attention window size
        self.max_seq_len = model_configs["max_seq_len"]
        if isinstance(model_configs["n_mha_win_size"], int):
            self.mha_win_size = [model_configs["n_mha_win_size"]] * len(self.fpn_strides)
        else:
            assert len(model_configs["n_mha_win_size"]) == len(self.fpn_strides)
            self.mha_win_size = model_configs["n_mha_win_size"]
        max_div_factor = 1
        for l, (s, w) in enumerate(zip(self.fpn_strides, self.mha_win_size)):
            stride = s * (w // 2) * 2 if w > 1 else s
            assert model_configs[
                       "max_seq_len"] % stride == 0, "max_seq_len must be divisible by fpn stride and window size"
            if max_div_factor < stride:
                max_div_factor = stride
        self.max_div_factor = max_div_factor

        # training time config
        self.train_cls_prior_prob = train_cfg['cls_prior_prob']
        self.train_dropout = train_cfg['dropout']
        self.use_offset = False
        self.train_droppath = train_cfg['droppath']
        self.fpn_levels = model_configs["backbone_arch"][-1] + 1
        self.regression_range = model_configs["regression_range"]
        self.num_classes = model_configs["num_classes"]
        # we will need a better way to dispatch the params to backbones / necks
        # backbone network: conv + transformer
        assert model_configs["backbone_type"] in ['convTransformer', 'conv']
        if model_configs["backbone_type"] == 'convTransformer':
            self.backbone = ConvTransformerBackbone(
                n_in=model_configs["input_dim"],
                n_embd=model_configs["embd_dim"],
                n_head=model_configs["n_head"],
                n_embd_ks=model_configs["embd_kernel_size"],
                max_len=model_configs["max_seq_len"],
                arch=model_configs["backbone_arch"],
                mha_win_size=self.mha_win_size,
                scale_factor=model_configs["scale_factor"],
                with_ln=model_configs["embd_with_ln"],
                attn_pdrop=0.0,
                proj_pdrop=self.train_dropout,
                path_pdrop=self.train_droppath,
                use_abs_pe=model_configs["use_abs_pe"],
                use_rel_pe=model_configs["use_rel_pe"]
            )
        else:
            self.backbone = ConvBackbone(n_in=model_configs["input_dim"], n_embd=model_configs["embd_dim"],
                                         n_embd_ks=model_configs["embd_kernel_size"],
                                         arch=model_configs["backbone_arch"],
                                         scale_factor=model_configs["scale_factor"],
                                         with_ln=model_configs["embd_with_ln"])
        # fpn network: convs
        assert model_configs["fpn_type"] in ['fpn', 'identity']
        if model_configs["fpn_type"] == 'fpn':
            self.neck = FPN1D(
                in_channels=[model_configs["embd_dim"]] * (model_configs["backbone_arch"][-1] + 1),
                out_channel=model_configs["fpn_dim"],
                scale_factor=model_configs["scale_factor"],
                with_ln=model_configs["fpn_with_ln"]
            )
        else:
            self.neck = FPNIdentity(
                in_channels=[model_configs["embd_dim"]] * (model_configs["backbone_arch"][-1] + 1),
                out_channel=model_configs["fpn_dim"],
                scale_factor=model_configs["scale_factor"],
                with_ln=model_configs["fpn_with_ln"]
            )
        self.point_generator = PointGenerator(
            max_seq_len=model_configs["max_seq_len"] * model_configs["max_buffer_len_factor"],
            fpn_levels=len(self.fpn_strides),
            scale_factor=model_configs["scale_factor"],
            regression_range=self.reg_range)
        # classfication and regerssion heads
        self.cls_head = PtTransformerClsHead(
            model_configs["fpn_dim"], model_configs["head_dim"], 1,
            kernel_size=model_configs["head_kernel_size"],
            prior_prob=self.train_cls_prior_prob,
            with_ln=model_configs["head_with_ln"],
            empty_cls=train_cfg['head_empty_cls']
        )
        self.reg_head = PtTransformerRegHead(
            model_configs["fpn_dim"], model_configs["head_dim"], len(self.fpn_strides),
            kernel_size=model_configs["head_kernel_size"],
            with_ln=model_configs["head_with_ln"]
        )
        '''
        for p in self.parameters():
            p.requires_grad=False
        '''

        self.pool = RoIPool(1, 1)
        self.proposal_creator = ProposalCreator(parent_model=self, nms_thresh=train_cfg["iou_threshold"],
                                                n_train_pre_nms=train_cfg["pre_train_nms_topk"],
                                                n_train_post_nms=train_cfg["train_max_seg_num"],
                                                n_test_pre_nms=train_cfg["pre_test_nms_topk"],
                                                n_test_post_nms=train_cfg["test_max_seg_num"],
                                                pre_nms_thresh=train_cfg["pre_nms_thresh"],
                                                min_size=train_cfg["duration_thresh"], sigma=train_cfg["nms_sigma"],
                                                min_score=train_cfg["min_score"])
        self.type = 2
        self.childs_num = 4

        self.classify = nn.Linear(model_configs["fpn_dim"] * 2, self.num_classes + 1)
        self.classify1 = Graph_module_net_0_loss_type(model_configs["fpn_dim"], model_configs["fpn_dim"] * 2,
                                                      model_configs["fpn_dim"], self.type, self.childs_num, dropout=0.7)
        self.relu = nn.ReLU()
        '''
        self.compeletness = nn.Sequential(
            MLP(model_configs["fpn_dim"] * 3, model_configs["fpn_dim"] * 6, model_configs["fpn_dim"] * 3, dropout=0.7),
            nn.ReLU(),
            nn.Linear(model_configs["fpn_dim"] * 3, self.num_classes)
        )

        self.classify = nn.Sequential(
            MLP(model_configs["fpn_dim"], model_configs["fpn_dim"] * 2, model_configs["fpn_dim"] , dropout=0.7),
            nn.ReLU(),
            nn.Linear(model_configs["fpn_dim"] , self.num_classes+1)
        )
        self.relu = nn.ReLU()
        '''
        self.compeletness = nn.Linear(model_configs["fpn_dim"] * 6, self.num_classes)
        self.compeletness1 = Graph_module_net_0_loss_type(model_configs["fpn_dim"] * 3, model_configs["fpn_dim"] * 6,
                                                          model_configs["fpn_dim"] * 3, self.type, self.childs_num,
                                                          dropout=0.7)

    @property
    def device(self):
        # a hacky way to get the device type
        # will throw an error if parameters are on different devices
        return list(set(p.device for p in self.parameters()))[0]

    def forward(self, inputs, training=True):
        # batch the video list into feats (B, C, T) and masks (B, 1, T)
        if training:
            batched_inputs, batched_masks, gt_segments, gt_masks = inputs[0], inputs[1], inputs[2], inputs[3]
        else:
            batched_inputs, batched_masks = inputs[0], inputs[1]
            gt_segments = torch.ones(batched_inputs.size()[0], 100, 2)
            gt_masks = torch.ones(batched_inputs.size()[0], 100)
        batch = batched_inputs.size()[0]
        batched_inputs = batched_inputs.cuda()
        batched_masks = batched_masks.unsqueeze(1).cuda()

        # forward the network (backbone -> neck -> heads)
        feats, masks = self.backbone(batched_inputs, batched_masks)
        fpn_feats, fpn_masks = self.neck(feats, masks)
        # out_cls: List[B, #cls + 1, T_i]
        out_cls_logits = self.cls_head(fpn_feats, fpn_masks)
        # out_offset: List[B, 2, T_i]
        out_offsets = self.reg_head(fpn_feats, fpn_masks)
        points = self.point_generator(fpn_feats)
        # permute the outputs
        rois = []
        scorces = []
        rois_mask = []
        for i in range(batch):
            # out_cls: F List[B, #cls, T_i] -> F List[B, T_i, #cls]
            out_cls_logits_i = [x.permute(0, 2, 1)[i, :, :] for x in out_cls_logits]
            # out_offset: F List[B, 2 (xC), T_i] -> F List[B, T_i, 2 (xC)]
            out_offsets_i = [x.permute(0, 2, 1)[i, :, :] for x in out_offsets]
            # fpn_masks: F list[B, 1, T_i] -> F List[B, T_i]
            fpn_masks_i = [x.squeeze(1)[i, :] for x in fpn_masks]
            gt_segment = gt_segments[i]
            gt_mask = gt_masks[i]
            mask_idx = gt_mask > 0
            gt_segment = gt_segment[mask_idx]
            with torch.no_grad():
                rois_i, score_i, roi_mask_i = self.proposal_creator(out_offsets=out_offsets_i,
                                                                    out_cls_logits=out_cls_logits_i,
                                                                    fpn_masks=fpn_masks_i, points=points,
                                                                    gt_rois=gt_segment)
            batch_index = i * torch.ones((rois_i.size()[0],)).unsqueeze(1)
            new_rois_i = torch.cat((batch_index, rois_i), dim=1)
            rois.append(new_rois_i)
            scorces.append(score_i)
            rois_mask.append(roi_mask_i)
        rois = torch.stack(rois, dim=0)
        rois_b, rois_n, _ = rois.size()
        gcn_mask1, gcn_mask2 = self._sample_nodes(rois)
        scorces = torch.stack(scorces, dim=0)
        rois_mask = torch.stack(rois_mask, dim=0)
        out_cls_logits = [x.permute(0, 2, 1) for x in out_cls_logits]
        # out_offset: F List[B, 2 (xC), T_i] -> F List[B, T_i, 2 (xC)]
        out_offsets = [x.permute(0, 2, 1) for x in out_offsets]
        # fpn_masks: F list[B, 1, T_i] -> F List[B, T_i]
        fpn_masks = [x.squeeze(1) for x in fpn_masks]
        rois_mask = rois_mask.long()
        rois_mask = F.one_hot(
            rois_mask, num_classes=len(fpn_masks)
        )
        rois_mask = rois_mask.float().view(-1, 1, len(fpn_masks))
        rois_pool_feats = []
        com_rois_feats = []

        for j in range(len(fpn_feats)):
            new_rois = rois / self.fpn_strides[j]
            extend_rois = new_rois.new_zeros(new_rois.size()[0], new_rois.size()[1], 5)
            extend_rois[:, :, 0] = new_rois[:, :, 0]
            extend_rois[:, :, 1] = new_rois[:, :, 1]
            # extend_rois[:, :, 2] = 0
            extend_rois[:, :, 3] = new_rois[:, :, 2]
            # extend_rois[:, :, 4] = 0
            roi_feat = (fpn_feats[j] * fpn_masks[j][:, None, :]).unsqueeze(2)
            left_extend_rois = extend_rois.new_zeros(new_rois.size()[0], new_rois.size()[1], 5)
            right_extend_rois = extend_rois.new_zeros(new_rois.size()[0], new_rois.size()[1], 5)
            left_extend_rois[:, :, 0] = extend_rois[:, :, 0]
            left_extend_rois[:, :, 1] = extend_rois[:, :, 1] - (extend_rois[:, :, 3] - extend_rois[:, :, 1]) * 0.5
            left_extend_rois[:, :, 3] = extend_rois[:, :, 1]
            left_extend_rois[:, :, 1] = left_extend_rois[:, :, 1].clamp(min=0.0, max=self.max_seq_len)
            right_extend_rois[:, :, 0] = extend_rois[:, :, 0]
            right_extend_rois[:, :, 1] = extend_rois[:, :, 3]
            right_extend_rois[:, :, 3] = extend_rois[:, :, 3] + (extend_rois[:, :, 3] - extend_rois[:, :, 1]) * 0.5
            right_extend_rois[:, :, 1] = right_extend_rois[:, :, 1].clamp(min=0.0, max=self.max_seq_len)
            extend_rois = extend_rois.view(-1, 5).cuda()
            left_extend_rois = left_extend_rois.view(-1, 5).cuda()
            right_extend_rois = right_extend_rois.view(-1, 5).cuda()
            sub_pool_feats = self.pool(roi_feat, extend_rois)
            left_pool_feats = self.pool(roi_feat, left_extend_rois)
            left_pool_feats = left_pool_feats.view(rois_b, rois_n, -1)
            right_pool_feats = self.pool(roi_feat, right_extend_rois)
            right_pool_feats = right_pool_feats.view(rois_b, rois_n, -1)
            sub_pool_feats = sub_pool_feats.view(rois_b, rois_n, -1)
            com_pool_feats = torch.cat((left_pool_feats, sub_pool_feats, right_pool_feats), dim=2)
            rois_pool_feats.append(sub_pool_feats)
            com_rois_feats.append(com_pool_feats)
        rois_pool_feats = torch.stack(rois_pool_feats, dim=2).view(rois_b * rois_n, len(fpn_feats), -1)
        com_rois_feats = torch.stack(com_rois_feats, dim=2).view(rois_b * rois_n, len(fpn_feats), -1)
        final_pool_feats = torch.bmm(rois_mask, rois_pool_feats).squeeze(1)
        scorces_mask = (scorces > 0.5).float()
        final_com_feats = torch.bmm(rois_mask, com_rois_feats).squeeze(1)
        # cls_log = self.classify(final_pool_feats).view(rois_b, rois_n, -1)

        final_pool_feats1 = final_pool_feats.view(rois_b, rois_n, -1)
        gt_pool_feats = final_pool_feats1[:, :gt_segments.size()[1], :]

        cls_log1, cls_gt, cls_node = self.classify1(final_pool_feats1, gcn_mask1, gcn_mask2, scorces_mask,
                                                    gt_pool_feats, rois)
        cls_log1 = self.relu(cls_log1)

        cls_log = torch.cat((final_pool_feats1, cls_log1), dim=2)
        cls_log = cls_log.view(rois_b * rois_n, -1)
        cls_log = self.classify(cls_log).view(rois_b, rois_n, -1)
        # com_log = self.compeletness(final_com_feats).view(rois_b, rois_n, -1)

        final_com_feats1 = final_com_feats.view(rois_b, rois_n, -1)
        gt_com_feats = final_com_feats1[:, :gt_segments.size()[1], :]

        com_log1, com_gt, com_node = self.compeletness1(final_com_feats1, gcn_mask1, gcn_mask2, scorces_mask,
                                                        gt_com_feats, rois)
        com_log1 = self.relu(com_log1)

        com_log = torch.cat((final_com_feats1, com_log1), dim=2)
        com_log = com_log.view(rois_b * rois_n, -1)
        com_log = self.compeletness(com_log).view(rois_b, rois_n, -1)

        # print(cls_log.size())

        # print(new_rois[0][0],2,fpn_feats[j].size()[2])
        return fpn_masks, out_cls_logits, out_offsets, rois.cuda(), scorces.cuda(), rois_mask, cls_log, com_log, cls_gt, cls_node, com_gt, com_node

        # return loss during training

    def _sample_nodes(self, rois):
        batch, num, _ = rois.size()
        new_rois = rois.new_zeros((rois.size()[0], rois.size()[1], 2))
        new_rois[:, :, 0] = rois[:, :, 1] - (rois[:, :, 2] - rois[:, :, 1]) * 0.5
        new_rois[:, :, 1] = rois[:, :, 2] + (rois[:, :, 2] - rois[:, :, 1]) * 0.5
        new_rois = new_rois.clamp(min=0.0, max=self.max_seq_len)
        # print(new_rois.size())
        left_rois = new_rois.unsqueeze(1).repeat(1, new_rois.size()[1], 1, 1)
        rigth_rois = new_rois.unsqueeze(2).repeat(1, 1, new_rois.size()[1], 1)
        min_left = torch.min(left_rois[:, :, :, 0], rigth_rois[:, :, :, 0])
        max_left = torch.max(left_rois[:, :, :, 0], rigth_rois[:, :, :, 0])
        min_right = torch.min(left_rois[:, :, :, 1], rigth_rois[:, :, :, 1])
        max_right = torch.max(left_rois[:, :, :, 1], rigth_rois[:, :, :, 1])
        ious = (min_right - max_left) / (max_right - min_left)
        ious_mask = (ious > 0).float()
        left_rois_center = (left_rois[:, :, :, 0] + left_rois[:, :, :, 1]) / 2.0
        rigth_rois_center = (rigth_rois[:, :, :, 0] + rigth_rois[:, :, :, 1]) / 2.0
        type_idxs = left_rois_center - rigth_rois_center
        ious_mask1 = ious_mask * (type_idxs >= 0).float()
        ious_mask2 = ious_mask * (type_idxs < 0).float()
        return ious_mask1, ious_mask2


class PtTransformer_gcn_loss_type_cat_all(nn.Module):
    """
        Transformer based model for single stage action localization
    """

    def __init__(
            self,
            model_configs,
            train_cfg,
    ):
        super().__init__()
        # re-distribute params to backbone / neck / head
        self.fpn_strides = [model_configs["scale_factor"] ** i for i in range(model_configs["backbone_arch"][-1] + 1)]
        self.reg_range = model_configs["regression_range"]
        assert len(self.fpn_strides) == len(self.reg_range)
        self.scale_factor = model_configs["scale_factor"]
        # check the feature pyramid and local attention window size
        self.max_seq_len = model_configs["max_seq_len"]
        if isinstance(model_configs["n_mha_win_size"], int):
            self.mha_win_size = [model_configs["n_mha_win_size"]] * len(self.fpn_strides)
        else:
            assert len(model_configs["n_mha_win_size"]) == len(self.fpn_strides)
            self.mha_win_size = model_configs["n_mha_win_size"]
        max_div_factor = 1
        for l, (s, w) in enumerate(zip(self.fpn_strides, self.mha_win_size)):
            stride = s * (w // 2) * 2 if w > 1 else s
            assert model_configs[
                       "max_seq_len"] % stride == 0, "max_seq_len must be divisible by fpn stride and window size"
            if max_div_factor < stride:
                max_div_factor = stride
        self.max_div_factor = max_div_factor

        # training time config
        self.train_cls_prior_prob = train_cfg['cls_prior_prob']
        self.train_dropout = train_cfg['dropout']
        self.use_offset = False
        self.train_droppath = train_cfg['droppath']
        self.fpn_levels = model_configs["backbone_arch"][-1] + 1
        self.regression_range = model_configs["regression_range"]
        self.num_classes = model_configs["num_classes"]
        # we will need a better way to dispatch the params to backbones / necks
        # backbone network: conv + transformer
        assert model_configs["backbone_type"] in ['convTransformer', 'conv']
        if model_configs["backbone_type"] == 'convTransformer':
            self.backbone = ConvTransformerBackbone(
                n_in=model_configs["input_dim"],
                n_embd=model_configs["embd_dim"],
                n_head=model_configs["n_head"],
                n_embd_ks=model_configs["embd_kernel_size"],
                max_len=model_configs["max_seq_len"],
                arch=model_configs["backbone_arch"],
                mha_win_size=self.mha_win_size,
                scale_factor=model_configs["scale_factor"],
                with_ln=model_configs["embd_with_ln"],
                attn_pdrop=0.0,
                proj_pdrop=self.train_dropout,
                path_pdrop=self.train_droppath,
                use_abs_pe=model_configs["use_abs_pe"],
                use_rel_pe=model_configs["use_rel_pe"]
            )
        else:
            self.backbone = ConvBackbone(n_in=model_configs["input_dim"], n_embd=model_configs["embd_dim"],
                                         n_embd_ks=model_configs["embd_kernel_size"],
                                         arch=model_configs["backbone_arch"],
                                         scale_factor=model_configs["scale_factor"],
                                         with_ln=model_configs["embd_with_ln"])
        # fpn network: convs
        assert model_configs["fpn_type"] in ['fpn', 'identity']
        if model_configs["fpn_type"] == 'fpn':
            self.neck = FPN1D(
                in_channels=[model_configs["embd_dim"]] * (model_configs["backbone_arch"][-1] + 1),
                out_channel=model_configs["fpn_dim"],
                scale_factor=model_configs["scale_factor"],
                with_ln=model_configs["fpn_with_ln"]
            )
        else:
            self.neck = FPNIdentity(
                in_channels=[model_configs["embd_dim"]] * (model_configs["backbone_arch"][-1] + 1),
                out_channel=model_configs["fpn_dim"],
                scale_factor=model_configs["scale_factor"],
                with_ln=model_configs["fpn_with_ln"]
            )
        self.point_generator = PointGenerator(
            max_seq_len=model_configs["max_seq_len"] * model_configs["max_buffer_len_factor"],
            fpn_levels=len(self.fpn_strides),
            scale_factor=model_configs["scale_factor"],
            regression_range=self.reg_range)
        # classfication and regerssion heads
        self.cls_head = PtTransformerClsHead(
            model_configs["fpn_dim"], model_configs["head_dim"], 1,
            kernel_size=model_configs["head_kernel_size"],
            prior_prob=self.train_cls_prior_prob,
            with_ln=model_configs["head_with_ln"],
            empty_cls=train_cfg['head_empty_cls']
        )
        self.reg_head = PtTransformerRegHead(
            model_configs["fpn_dim"], model_configs["head_dim"], len(self.fpn_strides),
            kernel_size=model_configs["head_kernel_size"],
            with_ln=model_configs["head_with_ln"]
        )
        '''
        for p in self.parameters():
            p.requires_grad=False
        '''

        self.pool = RoIPool(1, 1)
        self.proposal_creator = ProposalCreator(parent_model=self, nms_thresh=train_cfg["iou_threshold"],
                                                n_train_pre_nms=train_cfg["pre_train_nms_topk"],
                                                n_train_post_nms=train_cfg["train_max_seg_num"],
                                                n_test_pre_nms=train_cfg["pre_test_nms_topk"],
                                                n_test_post_nms=train_cfg["test_max_seg_num"],
                                                pre_nms_thresh=train_cfg["pre_nms_thresh"],
                                                min_size=train_cfg["duration_thresh"], sigma=train_cfg["nms_sigma"],
                                                min_score=train_cfg["min_score"])
        self.type = 2
        self.childs_num = 4

        self.classify = nn.Linear(model_configs["fpn_dim"] * 2, self.num_classes + 1)
        self.classify1 = Graph_module_net_0_loss_type(model_configs["fpn_dim"], model_configs["fpn_dim"] * 2,
                                                      model_configs["fpn_dim"], self.type, self.childs_num, dropout=0.7)
        self.relu = nn.ReLU()
        self.compeletness = nn.Linear(model_configs["fpn_dim"] * 6, self.num_classes)
        self.compeletness1 = Graph_module_net_0_loss_type(model_configs["fpn_dim"] * 3, model_configs["fpn_dim"] * 6,
                                                          model_configs["fpn_dim"] * 3, self.type, self.childs_num,
                                                          dropout=0.7)
        self.class_pool=nn.Sequential(nn.Linear(model_configs["fpn_dim"]*len(self.fpn_strides),model_configs["fpn_dim"]),
                                      nn.ReLU())
        self.com_pool = nn.Sequential(
            nn.Linear(model_configs["fpn_dim"] * len(self.fpn_strides)*3, model_configs["fpn_dim"]*3),
            nn.ReLU())

    @property
    def device(self):
        # a hacky way to get the device type
        # will throw an error if parameters are on different devices
        return list(set(p.device for p in self.parameters()))[0]

    def forward(self, inputs, training=True):
        # batch the video list into feats (B, C, T) and masks (B, 1, T)
        if training:
            batched_inputs, batched_masks, gt_segments, gt_masks = inputs[0], inputs[1], inputs[2], inputs[3]
        else:
            batched_inputs, batched_masks = inputs[0], inputs[1]
            gt_segments = torch.ones(batched_inputs.size()[0], 100, 2)
            gt_masks = torch.ones(batched_inputs.size()[0], 100)
        batch = batched_inputs.size()[0]
        batched_inputs = batched_inputs.cuda()
        batched_masks = batched_masks.unsqueeze(1).cuda()

        # forward the network (backbone -> neck -> heads)
        feats, masks = self.backbone(batched_inputs, batched_masks)
        fpn_feats, fpn_masks = self.neck(feats, masks)
        # out_cls: List[B, #cls + 1, T_i]
        out_cls_logits = self.cls_head(fpn_feats, fpn_masks)
        # out_offset: List[B, 2, T_i]
        out_offsets = self.reg_head(fpn_feats, fpn_masks)
        points = self.point_generator(fpn_feats)
        # permute the outputs
        rois = []
        scorces = []
        rois_mask = []
        for i in range(batch):
            # out_cls: F List[B, #cls, T_i] -> F List[B, T_i, #cls]
            out_cls_logits_i = [x.permute(0, 2, 1)[i, :, :] for x in out_cls_logits]
            # out_offset: F List[B, 2 (xC), T_i] -> F List[B, T_i, 2 (xC)]
            out_offsets_i = [x.permute(0, 2, 1)[i, :, :] for x in out_offsets]
            # fpn_masks: F list[B, 1, T_i] -> F List[B, T_i]
            fpn_masks_i = [x.squeeze(1)[i, :] for x in fpn_masks]
            gt_segment = gt_segments[i]
            gt_mask = gt_masks[i]
            mask_idx = gt_mask > 0
            gt_segment = gt_segment[mask_idx]
            with torch.no_grad():
                rois_i, score_i, roi_mask_i = self.proposal_creator(out_offsets=out_offsets_i,
                                                                    out_cls_logits=out_cls_logits_i,
                                                                    fpn_masks=fpn_masks_i, points=points,
                                                                    gt_rois=gt_segment)
            batch_index = i * torch.ones((rois_i.size()[0],)).unsqueeze(1)
            new_rois_i = torch.cat((batch_index, rois_i), dim=1)
            rois.append(new_rois_i)
            scorces.append(score_i)
            rois_mask.append(roi_mask_i)
        rois = torch.stack(rois, dim=0)
        rois_b, rois_n, _ = rois.size()
        gcn_mask1, gcn_mask2 = self._sample_nodes(rois)
        scorces = torch.stack(scorces, dim=0)
        rois_mask = torch.stack(rois_mask, dim=0)
        out_cls_logits = [x.permute(0, 2, 1) for x in out_cls_logits]
        # out_offset: F List[B, 2 (xC), T_i] -> F List[B, T_i, 2 (xC)]
        out_offsets = [x.permute(0, 2, 1) for x in out_offsets]
        # fpn_masks: F list[B, 1, T_i] -> F List[B, T_i]
        fpn_masks = [x.squeeze(1) for x in fpn_masks]
        rois_mask = rois_mask.long()
        rois_mask = F.one_hot(
            rois_mask, num_classes=len(fpn_masks)
        )
        rois_mask = rois_mask.float().view(-1, 1, len(fpn_masks))
        rois_pool_feats = []
        com_rois_feats = []

        for j in range(len(fpn_feats)):
            new_rois = rois / self.fpn_strides[j]
            extend_rois = new_rois.new_zeros(new_rois.size()[0], new_rois.size()[1], 5)
            extend_rois[:, :, 0] = new_rois[:, :, 0]
            extend_rois[:, :, 1] = new_rois[:, :, 1]
            # extend_rois[:, :, 2] = 0
            extend_rois[:, :, 3] = new_rois[:, :, 2]
            # extend_rois[:, :, 4] = 0
            roi_feat = (fpn_feats[j] * fpn_masks[j][:, None, :]).unsqueeze(2)
            left_extend_rois = extend_rois.new_zeros(new_rois.size()[0], new_rois.size()[1], 5)
            right_extend_rois = extend_rois.new_zeros(new_rois.size()[0], new_rois.size()[1], 5)
            left_extend_rois[:, :, 0] = extend_rois[:, :, 0]
            left_extend_rois[:, :, 1] = extend_rois[:, :, 1] - (extend_rois[:, :, 3] - extend_rois[:, :, 1]) * 0.5
            left_extend_rois[:, :, 3] = extend_rois[:, :, 1]
            left_extend_rois[:, :, 1] = left_extend_rois[:, :, 1].clamp(min=0.0, max=self.max_seq_len)
            right_extend_rois[:, :, 0] = extend_rois[:, :, 0]
            right_extend_rois[:, :, 1] = extend_rois[:, :, 3]
            right_extend_rois[:, :, 3] = extend_rois[:, :, 3] + (extend_rois[:, :, 3] - extend_rois[:, :, 1]) * 0.5
            right_extend_rois[:, :, 1] = right_extend_rois[:, :, 1].clamp(min=0.0, max=self.max_seq_len)
            extend_rois = extend_rois.view(-1, 5).cuda()
            left_extend_rois = left_extend_rois.view(-1, 5).cuda()
            right_extend_rois = right_extend_rois.view(-1, 5).cuda()
            sub_pool_feats = self.pool(roi_feat, extend_rois)
            left_pool_feats = self.pool(roi_feat, left_extend_rois)
            left_pool_feats = left_pool_feats.view(rois_b, rois_n, -1)
            right_pool_feats = self.pool(roi_feat, right_extend_rois)
            right_pool_feats = right_pool_feats.view(rois_b, rois_n, -1)
            sub_pool_feats = sub_pool_feats.view(rois_b, rois_n, -1)
            com_pool_feats = torch.cat((left_pool_feats, sub_pool_feats, right_pool_feats), dim=2)
            rois_pool_feats.append(sub_pool_feats)
            com_rois_feats.append(com_pool_feats)
        rois_pool_feats = torch.stack(rois_pool_feats, dim=2).view(rois_b * rois_n, -1)
        com_rois_feats = torch.stack(com_rois_feats, dim=2).view(rois_b * rois_n, -1)
        #final_pool_feats = torch.bmm(rois_mask, rois_pool_feats).squeeze(1)
        final_pool_feats =self.class_pool(rois_pool_feats)
        scorces_mask = (scorces > 0.5).float()
        #final_com_feats = torch.bmm(rois_mask, com_rois_feats).squeeze(1)
        # cls_log = self.classify(final_pool_feats).view(rois_b, rois_n, -1)
        final_com_feats =self.com_pool(com_rois_feats)
        final_pool_feats1 = final_pool_feats.view(rois_b, rois_n, -1)
        gt_pool_feats = final_pool_feats1[:, :gt_segments.size()[1], :]

        cls_log1, cls_gt, cls_node = self.classify1(final_pool_feats1, gcn_mask1, gcn_mask2, scorces_mask,
                                                    gt_pool_feats, rois)
        cls_log1 = self.relu(cls_log1)

        cls_log = torch.cat((final_pool_feats1, cls_log1), dim=2)
        cls_log = cls_log.view(rois_b * rois_n, -1)
        cls_log = self.classify(cls_log).view(rois_b, rois_n, -1)
        # com_log = self.compeletness(final_com_feats).view(rois_b, rois_n, -1)

        final_com_feats1 = final_com_feats.view(rois_b, rois_n, -1)
        gt_com_feats = final_com_feats1[:, :gt_segments.size()[1], :]

        com_log1, com_gt, com_node = self.compeletness1(final_com_feats1, gcn_mask1, gcn_mask2, scorces_mask,
                                                        gt_com_feats, rois)
        com_log1 = self.relu(com_log1)

        com_log = torch.cat((final_com_feats1, com_log1), dim=2)
        com_log = com_log.view(rois_b * rois_n, -1)
        com_log = self.compeletness(com_log).view(rois_b, rois_n, -1)

        # print(cls_log.size())

        # print(new_rois[0][0],2,fpn_feats[j].size()[2])
        return fpn_masks, out_cls_logits, out_offsets, rois.cuda(), scorces.cuda(), rois_mask, cls_log, com_log, cls_gt, cls_node, com_gt, com_node

        # return loss during training

    def _sample_nodes(self, rois):
        batch, num, _ = rois.size()
        new_rois = rois.new_zeros((rois.size()[0], rois.size()[1], 2))
        new_rois[:, :, 0] = rois[:, :, 1] - (rois[:, :, 2] - rois[:, :, 1]) * 0.5
        new_rois[:, :, 1] = rois[:, :, 2] + (rois[:, :, 2] - rois[:, :, 1]) * 0.5
        new_rois = new_rois.clamp(min=0.0, max=self.max_seq_len)
        # print(new_rois.size())
        left_rois = new_rois.unsqueeze(1).repeat(1, new_rois.size()[1], 1, 1)
        rigth_rois = new_rois.unsqueeze(2).repeat(1, 1, new_rois.size()[1], 1)
        min_left = torch.min(left_rois[:, :, :, 0], rigth_rois[:, :, :, 0])
        max_left = torch.max(left_rois[:, :, :, 0], rigth_rois[:, :, :, 0])
        min_right = torch.min(left_rois[:, :, :, 1], rigth_rois[:, :, :, 1])
        max_right = torch.max(left_rois[:, :, :, 1], rigth_rois[:, :, :, 1])
        ious = (min_right - max_left) / (max_right - min_left)
        ious_mask = (ious > 0).float()
        left_rois_center = (left_rois[:, :, :, 0] + left_rois[:, :, :, 1]) / 2.0
        rigth_rois_center = (rigth_rois[:, :, :, 0] + rigth_rois[:, :, :, 1]) / 2.0
        type_idxs = left_rois_center - rigth_rois_center
        ious_mask1 = ious_mask * (type_idxs >= 0).float()
        ious_mask2 = ious_mask * (type_idxs < 0).float()
        return ious_mask1, ious_mask2


class PtTransformer_gcn_loss_type_add_all(nn.Module):
    """
        Transformer based model for single stage action localization
    """

    def __init__(
            self,
            model_configs,
            train_cfg,
    ):
        super().__init__()
        # re-distribute params to backbone / neck / head
        self.fpn_strides = [model_configs["scale_factor"] ** i for i in range(model_configs["backbone_arch"][-1] + 1)]
        self.reg_range = model_configs["regression_range"]
        assert len(self.fpn_strides) == len(self.reg_range)
        self.scale_factor = model_configs["scale_factor"]
        # check the feature pyramid and local attention window size
        self.max_seq_len = model_configs["max_seq_len"]
        if isinstance(model_configs["n_mha_win_size"], int):
            self.mha_win_size = [model_configs["n_mha_win_size"]] * len(self.fpn_strides)
        else:
            assert len(model_configs["n_mha_win_size"]) == len(self.fpn_strides)
            self.mha_win_size = model_configs["n_mha_win_size"]
        max_div_factor = 1
        for l, (s, w) in enumerate(zip(self.fpn_strides, self.mha_win_size)):
            stride = s * (w // 2) * 2 if w > 1 else s
            assert model_configs[
                       "max_seq_len"] % stride == 0, "max_seq_len must be divisible by fpn stride and window size"
            if max_div_factor < stride:
                max_div_factor = stride
        self.max_div_factor = max_div_factor

        # training time config
        self.train_cls_prior_prob = train_cfg['cls_prior_prob']
        self.train_dropout = train_cfg['dropout']
        self.use_offset = False
        self.train_droppath = train_cfg['droppath']
        self.fpn_levels = model_configs["backbone_arch"][-1] + 1
        self.regression_range = model_configs["regression_range"]
        self.num_classes = model_configs["num_classes"]
        # we will need a better way to dispatch the params to backbones / necks
        # backbone network: conv + transformer
        assert model_configs["backbone_type"] in ['convTransformer', 'conv']
        if model_configs["backbone_type"] == 'convTransformer':
            self.backbone = ConvTransformerBackbone(
                n_in=model_configs["input_dim"],
                n_embd=model_configs["embd_dim"],
                n_head=model_configs["n_head"],
                n_embd_ks=model_configs["embd_kernel_size"],
                max_len=model_configs["max_seq_len"],
                arch=model_configs["backbone_arch"],
                mha_win_size=self.mha_win_size,
                scale_factor=model_configs["scale_factor"],
                with_ln=model_configs["embd_with_ln"],
                attn_pdrop=0.0,
                proj_pdrop=self.train_dropout,
                path_pdrop=self.train_droppath,
                use_abs_pe=model_configs["use_abs_pe"],
                use_rel_pe=model_configs["use_rel_pe"]
            )
        else:
            self.backbone = ConvBackbone(n_in=model_configs["input_dim"], n_embd=model_configs["embd_dim"],
                                         n_embd_ks=model_configs["embd_kernel_size"],
                                         arch=model_configs["backbone_arch"],
                                         scale_factor=model_configs["scale_factor"],
                                         with_ln=model_configs["embd_with_ln"])
        # fpn network: convs
        assert model_configs["fpn_type"] in ['fpn', 'identity']
        if model_configs["fpn_type"] == 'fpn':
            self.neck = FPN1D(
                in_channels=[model_configs["embd_dim"]] * (model_configs["backbone_arch"][-1] + 1),
                out_channel=model_configs["fpn_dim"],
                scale_factor=model_configs["scale_factor"],
                with_ln=model_configs["fpn_with_ln"]
            )
        else:
            self.neck = FPNIdentity(
                in_channels=[model_configs["embd_dim"]] * (model_configs["backbone_arch"][-1] + 1),
                out_channel=model_configs["fpn_dim"],
                scale_factor=model_configs["scale_factor"],
                with_ln=model_configs["fpn_with_ln"]
            )
        self.point_generator = PointGenerator(
            max_seq_len=model_configs["max_seq_len"] * model_configs["max_buffer_len_factor"],
            fpn_levels=len(self.fpn_strides),
            scale_factor=model_configs["scale_factor"],
            regression_range=self.reg_range)
        # classfication and regerssion heads
        self.cls_head = PtTransformerClsHead(
            model_configs["fpn_dim"], model_configs["head_dim"], 1,
            kernel_size=model_configs["head_kernel_size"],
            prior_prob=self.train_cls_prior_prob,
            with_ln=model_configs["head_with_ln"],
            empty_cls=train_cfg['head_empty_cls']
        )
        self.reg_head = PtTransformerRegHead(
            model_configs["fpn_dim"], model_configs["head_dim"], len(self.fpn_strides),
            kernel_size=model_configs["head_kernel_size"],
            with_ln=model_configs["head_with_ln"]
        )
        '''
        for p in self.parameters():
            p.requires_grad=False
        '''

        self.pool = RoIPool(1, 1)
        self.proposal_creator = ProposalCreator(parent_model=self, nms_thresh=train_cfg["iou_threshold"],
                                                n_train_pre_nms=train_cfg["pre_train_nms_topk"],
                                                n_train_post_nms=train_cfg["train_max_seg_num"],
                                                n_test_pre_nms=train_cfg["pre_test_nms_topk"],
                                                n_test_post_nms=train_cfg["test_max_seg_num"],
                                                pre_nms_thresh=train_cfg["pre_nms_thresh"],
                                                min_size=train_cfg["duration_thresh"], sigma=train_cfg["nms_sigma"],
                                                min_score=train_cfg["min_score"])
        self.type = 2
        self.childs_num = 4

        self.classify = nn.Linear(model_configs["fpn_dim"] * 2, self.num_classes + 1)
        self.classify1 = Graph_module_net_0_loss_type(model_configs["fpn_dim"], model_configs["fpn_dim"] * 2,
                                                      model_configs["fpn_dim"], self.type, self.childs_num, dropout=0.7)
        self.relu = nn.ReLU()
        '''
        self.compeletness = nn.Sequential(
            MLP(model_configs["fpn_dim"] * 3, model_configs["fpn_dim"] * 6, model_configs["fpn_dim"] * 3, dropout=0.7),
            nn.ReLU(),
            nn.Linear(model_configs["fpn_dim"] * 3, self.num_classes)
        )

        self.classify = nn.Sequential(
            MLP(model_configs["fpn_dim"], model_configs["fpn_dim"] * 2, model_configs["fpn_dim"] , dropout=0.7),
            nn.ReLU(),
            nn.Linear(model_configs["fpn_dim"] , self.num_classes+1)
        )
        self.relu = nn.ReLU()
        '''
        self.compeletness = nn.Linear(model_configs["fpn_dim"] * 6, self.num_classes)
        self.compeletness1 = Graph_module_net_0_loss_type(model_configs["fpn_dim"] * 3, model_configs["fpn_dim"] * 6,
                                                          model_configs["fpn_dim"] * 3, self.type, self.childs_num,
                                                          dropout=0.7)

    @property
    def device(self):
        # a hacky way to get the device type
        # will throw an error if parameters are on different devices
        return list(set(p.device for p in self.parameters()))[0]

    def forward(self, inputs, training=True):
        # batch the video list into feats (B, C, T) and masks (B, 1, T)
        if training:
            batched_inputs, batched_masks, gt_segments, gt_masks = inputs[0], inputs[1], inputs[2], inputs[3]
        else:
            batched_inputs, batched_masks = inputs[0], inputs[1]
            gt_segments = torch.ones(batched_inputs.size()[0], 100, 2)
            gt_masks = torch.ones(batched_inputs.size()[0], 100)
        batch = batched_inputs.size()[0]
        batched_inputs = batched_inputs.cuda()
        batched_masks = batched_masks.unsqueeze(1).cuda()

        # forward the network (backbone -> neck -> heads)
        feats, masks = self.backbone(batched_inputs, batched_masks)
        fpn_feats, fpn_masks = self.neck(feats, masks)
        # out_cls: List[B, #cls + 1, T_i]
        out_cls_logits = self.cls_head(fpn_feats, fpn_masks)
        # out_offset: List[B, 2, T_i]
        out_offsets = self.reg_head(fpn_feats, fpn_masks)
        points = self.point_generator(fpn_feats)
        # permute the outputs
        rois = []
        scorces = []
        rois_mask = []
        for i in range(batch):
            # out_cls: F List[B, #cls, T_i] -> F List[B, T_i, #cls]
            out_cls_logits_i = [x.permute(0, 2, 1)[i, :, :] for x in out_cls_logits]
            # out_offset: F List[B, 2 (xC), T_i] -> F List[B, T_i, 2 (xC)]
            out_offsets_i = [x.permute(0, 2, 1)[i, :, :] for x in out_offsets]
            # fpn_masks: F list[B, 1, T_i] -> F List[B, T_i]
            fpn_masks_i = [x.squeeze(1)[i, :] for x in fpn_masks]
            gt_segment = gt_segments[i]
            gt_mask = gt_masks[i]
            mask_idx = gt_mask > 0
            gt_segment = gt_segment[mask_idx]
            with torch.no_grad():
                rois_i, score_i, roi_mask_i = self.proposal_creator(out_offsets=out_offsets_i,
                                                                    out_cls_logits=out_cls_logits_i,
                                                                    fpn_masks=fpn_masks_i, points=points,
                                                                    gt_rois=gt_segment)
            batch_index = i * torch.ones((rois_i.size()[0],)).unsqueeze(1)
            new_rois_i = torch.cat((batch_index, rois_i), dim=1)
            rois.append(new_rois_i)
            scorces.append(score_i)
            rois_mask.append(roi_mask_i)
        rois = torch.stack(rois, dim=0)
        rois_b, rois_n, _ = rois.size()
        gcn_mask1, gcn_mask2 = self._sample_nodes(rois)
        scorces = torch.stack(scorces, dim=0)
        rois_mask = torch.stack(rois_mask, dim=0)
        out_cls_logits = [x.permute(0, 2, 1) for x in out_cls_logits]
        # out_offset: F List[B, 2 (xC), T_i] -> F List[B, T_i, 2 (xC)]
        out_offsets = [x.permute(0, 2, 1) for x in out_offsets]
        # fpn_masks: F list[B, 1, T_i] -> F List[B, T_i]
        fpn_masks = [x.squeeze(1) for x in fpn_masks]
        rois_mask = rois_mask.long()
        rois_mask = F.one_hot(
            rois_mask, num_classes=len(fpn_masks)
        )
        rois_mask = rois_mask.float().view(-1, 1, len(fpn_masks))
        rois_pool_feats = []
        com_rois_feats = []

        for j in range(len(fpn_feats)):
            new_rois = rois / self.fpn_strides[j]
            extend_rois = new_rois.new_zeros(new_rois.size()[0], new_rois.size()[1], 5)
            extend_rois[:, :, 0] = new_rois[:, :, 0]
            extend_rois[:, :, 1] = new_rois[:, :, 1]
            # extend_rois[:, :, 2] = 0
            extend_rois[:, :, 3] = new_rois[:, :, 2]
            # extend_rois[:, :, 4] = 0
            roi_feat = (fpn_feats[j] * fpn_masks[j][:, None, :]).unsqueeze(2)
            left_extend_rois = extend_rois.new_zeros(new_rois.size()[0], new_rois.size()[1], 5)
            right_extend_rois = extend_rois.new_zeros(new_rois.size()[0], new_rois.size()[1], 5)
            left_extend_rois[:, :, 0] = extend_rois[:, :, 0]
            left_extend_rois[:, :, 1] = extend_rois[:, :, 1] - (extend_rois[:, :, 3] - extend_rois[:, :, 1]) * 0.5
            left_extend_rois[:, :, 3] = extend_rois[:, :, 1]
            left_extend_rois[:, :, 1] = left_extend_rois[:, :, 1].clamp(min=0.0, max=self.max_seq_len)
            right_extend_rois[:, :, 0] = extend_rois[:, :, 0]
            right_extend_rois[:, :, 1] = extend_rois[:, :, 3]
            right_extend_rois[:, :, 3] = extend_rois[:, :, 3] + (extend_rois[:, :, 3] - extend_rois[:, :, 1]) * 0.5
            right_extend_rois[:, :, 1] = right_extend_rois[:, :, 1].clamp(min=0.0, max=self.max_seq_len)
            extend_rois = extend_rois.view(-1, 5).cuda()
            left_extend_rois = left_extend_rois.view(-1, 5).cuda()
            right_extend_rois = right_extend_rois.view(-1, 5).cuda()
            sub_pool_feats = self.pool(roi_feat, extend_rois)
            left_pool_feats = self.pool(roi_feat, left_extend_rois)
            left_pool_feats = left_pool_feats.view(rois_b, rois_n, -1)
            right_pool_feats = self.pool(roi_feat, right_extend_rois)
            right_pool_feats = right_pool_feats.view(rois_b, rois_n, -1)
            sub_pool_feats = sub_pool_feats.view(rois_b, rois_n, -1)
            com_pool_feats = torch.cat((left_pool_feats, sub_pool_feats, right_pool_feats), dim=2)
            rois_pool_feats.append(sub_pool_feats)
            com_rois_feats.append(com_pool_feats)
        rois_pool_feats = torch.stack(rois_pool_feats, dim=2).view(rois_b * rois_n, len(fpn_feats), -1)
        com_rois_feats = torch.stack(com_rois_feats, dim=2).view(rois_b * rois_n, len(fpn_feats), -1)
        final_pool_feats =torch.mean(rois_pool_feats,dim=1)
        #final_pool_feats = torch.bmm(rois_mask, rois_pool_feats).squeeze(1)
        scorces_mask = (scorces > 0.5).float()
        #final_com_feats = torch.bmm(rois_mask, com_rois_feats).squeeze(1)
        # cls_log = self.classify(final_pool_feats).view(rois_b, rois_n, -1)
        final_com_feats = torch.mean(com_rois_feats,dim=1)

        final_pool_feats1 = final_pool_feats.view(rois_b, rois_n, -1)
        gt_pool_feats = final_pool_feats1[:, :gt_segments.size()[1], :]

        cls_log1, cls_gt, cls_node = self.classify1(final_pool_feats1, gcn_mask1, gcn_mask2, scorces_mask,
                                                    gt_pool_feats, rois)
        cls_log1 = self.relu(cls_log1)

        cls_log = torch.cat((final_pool_feats1, cls_log1), dim=2)
        cls_log = cls_log.view(rois_b * rois_n, -1)
        cls_log = self.classify(cls_log).view(rois_b, rois_n, -1)
        # com_log = self.compeletness(final_com_feats).view(rois_b, rois_n, -1)

        final_com_feats1 = final_com_feats.view(rois_b, rois_n, -1)
        gt_com_feats = final_com_feats1[:, :gt_segments.size()[1], :]

        com_log1, com_gt, com_node = self.compeletness1(final_com_feats1, gcn_mask1, gcn_mask2, scorces_mask,
                                                        gt_com_feats, rois)
        com_log1 = self.relu(com_log1)

        com_log = torch.cat((final_com_feats1, com_log1), dim=2)
        com_log = com_log.view(rois_b * rois_n, -1)
        com_log = self.compeletness(com_log).view(rois_b, rois_n, -1)

        # print(cls_log.size())

        # print(new_rois[0][0],2,fpn_feats[j].size()[2])
        return fpn_masks, out_cls_logits, out_offsets, rois.cuda(), scorces.cuda(), rois_mask, cls_log, com_log, cls_gt, cls_node, com_gt, com_node

        # return loss during training

    def _sample_nodes(self, rois):
        batch, num, _ = rois.size()
        new_rois = rois.new_zeros((rois.size()[0], rois.size()[1], 2))
        new_rois[:, :, 0] = rois[:, :, 1] - (rois[:, :, 2] - rois[:, :, 1]) * 0.5
        new_rois[:, :, 1] = rois[:, :, 2] + (rois[:, :, 2] - rois[:, :, 1]) * 0.5
        new_rois = new_rois.clamp(min=0.0, max=self.max_seq_len)
        # print(new_rois.size())
        left_rois = new_rois.unsqueeze(1).repeat(1, new_rois.size()[1], 1, 1)
        rigth_rois = new_rois.unsqueeze(2).repeat(1, 1, new_rois.size()[1], 1)
        min_left = torch.min(left_rois[:, :, :, 0], rigth_rois[:, :, :, 0])
        max_left = torch.max(left_rois[:, :, :, 0], rigth_rois[:, :, :, 0])
        min_right = torch.min(left_rois[:, :, :, 1], rigth_rois[:, :, :, 1])
        max_right = torch.max(left_rois[:, :, :, 1], rigth_rois[:, :, :, 1])
        ious = (min_right - max_left) / (max_right - min_left)
        ious_mask = (ious > 0).float()
        left_rois_center = (left_rois[:, :, :, 0] + left_rois[:, :, :, 1]) / 2.0
        rigth_rois_center = (rigth_rois[:, :, :, 0] + rigth_rois[:, :, :, 1]) / 2.0
        type_idxs = left_rois_center - rigth_rois_center
        ious_mask1 = ious_mask * (type_idxs >= 0).float()
        ious_mask2 = ious_mask * (type_idxs < 0).float()
        return ious_mask1, ious_mask2


class PtTransformer_gcn_loss_type_max_all(nn.Module):
    """
        Transformer based model for single stage action localization
    """

    def __init__(
            self,
            model_configs,
            train_cfg,
    ):
        super().__init__()
        # re-distribute params to backbone / neck / head
        self.fpn_strides = [model_configs["scale_factor"] ** i for i in range(model_configs["backbone_arch"][-1] + 1)]
        self.reg_range = model_configs["regression_range"]
        assert len(self.fpn_strides) == len(self.reg_range)
        self.scale_factor = model_configs["scale_factor"]
        # check the feature pyramid and local attention window size
        self.max_seq_len = model_configs["max_seq_len"]
        if isinstance(model_configs["n_mha_win_size"], int):
            self.mha_win_size = [model_configs["n_mha_win_size"]] * len(self.fpn_strides)
        else:
            assert len(model_configs["n_mha_win_size"]) == len(self.fpn_strides)
            self.mha_win_size = model_configs["n_mha_win_size"]
        max_div_factor = 1
        for l, (s, w) in enumerate(zip(self.fpn_strides, self.mha_win_size)):
            stride = s * (w // 2) * 2 if w > 1 else s
            assert model_configs[
                       "max_seq_len"] % stride == 0, "max_seq_len must be divisible by fpn stride and window size"
            if max_div_factor < stride:
                max_div_factor = stride
        self.max_div_factor = max_div_factor

        # training time config
        self.train_cls_prior_prob = train_cfg['cls_prior_prob']
        self.train_dropout = train_cfg['dropout']
        self.use_offset = False
        self.train_droppath = train_cfg['droppath']
        self.fpn_levels = model_configs["backbone_arch"][-1] + 1
        self.regression_range = model_configs["regression_range"]
        self.num_classes = model_configs["num_classes"]
        # we will need a better way to dispatch the params to backbones / necks
        # backbone network: conv + transformer
        assert model_configs["backbone_type"] in ['convTransformer', 'conv']
        if model_configs["backbone_type"] == 'convTransformer':
            self.backbone = ConvTransformerBackbone(
                n_in=model_configs["input_dim"],
                n_embd=model_configs["embd_dim"],
                n_head=model_configs["n_head"],
                n_embd_ks=model_configs["embd_kernel_size"],
                max_len=model_configs["max_seq_len"],
                arch=model_configs["backbone_arch"],
                mha_win_size=self.mha_win_size,
                scale_factor=model_configs["scale_factor"],
                with_ln=model_configs["embd_with_ln"],
                attn_pdrop=0.0,
                proj_pdrop=self.train_dropout,
                path_pdrop=self.train_droppath,
                use_abs_pe=model_configs["use_abs_pe"],
                use_rel_pe=model_configs["use_rel_pe"]
            )
        else:
            self.backbone = ConvBackbone(n_in=model_configs["input_dim"], n_embd=model_configs["embd_dim"],
                                         n_embd_ks=model_configs["embd_kernel_size"],
                                         arch=model_configs["backbone_arch"],
                                         scale_factor=model_configs["scale_factor"],
                                         with_ln=model_configs["embd_with_ln"])
        # fpn network: convs
        assert model_configs["fpn_type"] in ['fpn', 'identity']
        if model_configs["fpn_type"] == 'fpn':
            self.neck = FPN1D(
                in_channels=[model_configs["embd_dim"]] * (model_configs["backbone_arch"][-1] + 1),
                out_channel=model_configs["fpn_dim"],
                scale_factor=model_configs["scale_factor"],
                with_ln=model_configs["fpn_with_ln"]
            )
        else:
            self.neck = FPNIdentity(
                in_channels=[model_configs["embd_dim"]] * (model_configs["backbone_arch"][-1] + 1),
                out_channel=model_configs["fpn_dim"],
                scale_factor=model_configs["scale_factor"],
                with_ln=model_configs["fpn_with_ln"]
            )
        self.point_generator = PointGenerator(
            max_seq_len=model_configs["max_seq_len"] * model_configs["max_buffer_len_factor"],
            fpn_levels=len(self.fpn_strides),
            scale_factor=model_configs["scale_factor"],
            regression_range=self.reg_range)
        # classfication and regerssion heads
        self.cls_head = PtTransformerClsHead(
            model_configs["fpn_dim"], model_configs["head_dim"], 1,
            kernel_size=model_configs["head_kernel_size"],
            prior_prob=self.train_cls_prior_prob,
            with_ln=model_configs["head_with_ln"],
            empty_cls=train_cfg['head_empty_cls']
        )
        self.reg_head = PtTransformerRegHead(
            model_configs["fpn_dim"], model_configs["head_dim"], len(self.fpn_strides),
            kernel_size=model_configs["head_kernel_size"],
            with_ln=model_configs["head_with_ln"]
        )
        '''
        for p in self.parameters():
            p.requires_grad=False
        '''

        self.pool = RoIPool(1, 1)
        self.proposal_creator = ProposalCreator(parent_model=self, nms_thresh=train_cfg["iou_threshold"],
                                                n_train_pre_nms=train_cfg["pre_train_nms_topk"],
                                                n_train_post_nms=train_cfg["train_max_seg_num"],
                                                n_test_pre_nms=train_cfg["pre_test_nms_topk"],
                                                n_test_post_nms=train_cfg["test_max_seg_num"],
                                                pre_nms_thresh=train_cfg["pre_nms_thresh"],
                                                min_size=train_cfg["duration_thresh"], sigma=train_cfg["nms_sigma"],
                                                min_score=train_cfg["min_score"])
        self.type = 2
        self.childs_num = 4

        self.classify = nn.Linear(model_configs["fpn_dim"] * 2, self.num_classes + 1)
        self.classify1 = Graph_module_net_0_loss_type(model_configs["fpn_dim"], model_configs["fpn_dim"] * 2,
                                                      model_configs["fpn_dim"], self.type, self.childs_num, dropout=0.7)
        self.relu = nn.ReLU()
        '''
        self.compeletness = nn.Sequential(
            MLP(model_configs["fpn_dim"] * 3, model_configs["fpn_dim"] * 6, model_configs["fpn_dim"] * 3, dropout=0.7),
            nn.ReLU(),
            nn.Linear(model_configs["fpn_dim"] * 3, self.num_classes)
        )

        self.classify = nn.Sequential(
            MLP(model_configs["fpn_dim"], model_configs["fpn_dim"] * 2, model_configs["fpn_dim"] , dropout=0.7),
            nn.ReLU(),
            nn.Linear(model_configs["fpn_dim"] , self.num_classes+1)
        )
        self.relu = nn.ReLU()
        '''
        self.compeletness = nn.Linear(model_configs["fpn_dim"] * 6, self.num_classes)
        self.compeletness1 = Graph_module_net_0_loss_type(model_configs["fpn_dim"] * 3, model_configs["fpn_dim"] * 6,
                                                          model_configs["fpn_dim"] * 3, self.type, self.childs_num,
                                                          dropout=0.7)

    @property
    def device(self):
        # a hacky way to get the device type
        # will throw an error if parameters are on different devices
        return list(set(p.device for p in self.parameters()))[0]

    def forward(self, inputs, training=True):
        # batch the video list into feats (B, C, T) and masks (B, 1, T)
        if training:
            batched_inputs, batched_masks, gt_segments, gt_masks = inputs[0], inputs[1], inputs[2], inputs[3]
        else:
            batched_inputs, batched_masks = inputs[0], inputs[1]
            gt_segments = torch.ones(batched_inputs.size()[0], 100, 2)
            gt_masks = torch.ones(batched_inputs.size()[0], 100)
        batch = batched_inputs.size()[0]
        batched_inputs = batched_inputs.cuda()
        batched_masks = batched_masks.unsqueeze(1).cuda()

        # forward the network (backbone -> neck -> heads)
        feats, masks = self.backbone(batched_inputs, batched_masks)
        fpn_feats, fpn_masks = self.neck(feats, masks)
        # out_cls: List[B, #cls + 1, T_i]
        out_cls_logits = self.cls_head(fpn_feats, fpn_masks)
        # out_offset: List[B, 2, T_i]
        out_offsets = self.reg_head(fpn_feats, fpn_masks)
        points = self.point_generator(fpn_feats)
        # permute the outputs
        rois = []
        scorces = []
        rois_mask = []
        for i in range(batch):
            # out_cls: F List[B, #cls, T_i] -> F List[B, T_i, #cls]
            out_cls_logits_i = [x.permute(0, 2, 1)[i, :, :] for x in out_cls_logits]
            # out_offset: F List[B, 2 (xC), T_i] -> F List[B, T_i, 2 (xC)]
            out_offsets_i = [x.permute(0, 2, 1)[i, :, :] for x in out_offsets]
            # fpn_masks: F list[B, 1, T_i] -> F List[B, T_i]
            fpn_masks_i = [x.squeeze(1)[i, :] for x in fpn_masks]
            gt_segment = gt_segments[i]
            gt_mask = gt_masks[i]
            mask_idx = gt_mask > 0
            gt_segment = gt_segment[mask_idx]
            with torch.no_grad():
                rois_i, score_i, roi_mask_i = self.proposal_creator(out_offsets=out_offsets_i,
                                                                    out_cls_logits=out_cls_logits_i,
                                                                    fpn_masks=fpn_masks_i, points=points,
                                                                    gt_rois=gt_segment)
            batch_index = i * torch.ones((rois_i.size()[0],)).unsqueeze(1)
            new_rois_i = torch.cat((batch_index, rois_i), dim=1)
            rois.append(new_rois_i)
            scorces.append(score_i)
            rois_mask.append(roi_mask_i)
        rois = torch.stack(rois, dim=0)
        rois_b, rois_n, _ = rois.size()
        gcn_mask1, gcn_mask2 = self._sample_nodes(rois)
        scorces = torch.stack(scorces, dim=0)
        rois_mask = torch.stack(rois_mask, dim=0)
        out_cls_logits = [x.permute(0, 2, 1) for x in out_cls_logits]
        # out_offset: F List[B, 2 (xC), T_i] -> F List[B, T_i, 2 (xC)]
        out_offsets = [x.permute(0, 2, 1) for x in out_offsets]
        # fpn_masks: F list[B, 1, T_i] -> F List[B, T_i]
        fpn_masks = [x.squeeze(1) for x in fpn_masks]
        rois_mask = rois_mask.long()
        rois_mask = F.one_hot(
            rois_mask, num_classes=len(fpn_masks)
        )
        rois_mask = rois_mask.float().view(-1, 1, len(fpn_masks))
        rois_pool_feats = []
        com_rois_feats = []

        for j in range(len(fpn_feats)):
            new_rois = rois / self.fpn_strides[j]
            extend_rois = new_rois.new_zeros(new_rois.size()[0], new_rois.size()[1], 5)
            extend_rois[:, :, 0] = new_rois[:, :, 0]
            extend_rois[:, :, 1] = new_rois[:, :, 1]
            # extend_rois[:, :, 2] = 0
            extend_rois[:, :, 3] = new_rois[:, :, 2]
            # extend_rois[:, :, 4] = 0
            roi_feat = (fpn_feats[j] * fpn_masks[j][:, None, :]).unsqueeze(2)
            left_extend_rois = extend_rois.new_zeros(new_rois.size()[0], new_rois.size()[1], 5)
            right_extend_rois = extend_rois.new_zeros(new_rois.size()[0], new_rois.size()[1], 5)
            left_extend_rois[:, :, 0] = extend_rois[:, :, 0]
            left_extend_rois[:, :, 1] = extend_rois[:, :, 1] - (extend_rois[:, :, 3] - extend_rois[:, :, 1]) * 0.5
            left_extend_rois[:, :, 3] = extend_rois[:, :, 1]
            left_extend_rois[:, :, 1] = left_extend_rois[:, :, 1].clamp(min=0.0, max=self.max_seq_len)
            right_extend_rois[:, :, 0] = extend_rois[:, :, 0]
            right_extend_rois[:, :, 1] = extend_rois[:, :, 3]
            right_extend_rois[:, :, 3] = extend_rois[:, :, 3] + (extend_rois[:, :, 3] - extend_rois[:, :, 1]) * 0.5
            right_extend_rois[:, :, 1] = right_extend_rois[:, :, 1].clamp(min=0.0, max=self.max_seq_len)
            extend_rois = extend_rois.view(-1, 5).cuda()
            left_extend_rois = left_extend_rois.view(-1, 5).cuda()
            right_extend_rois = right_extend_rois.view(-1, 5).cuda()
            sub_pool_feats = self.pool(roi_feat, extend_rois)
            left_pool_feats = self.pool(roi_feat, left_extend_rois)
            left_pool_feats = left_pool_feats.view(rois_b, rois_n, -1)
            right_pool_feats = self.pool(roi_feat, right_extend_rois)
            right_pool_feats = right_pool_feats.view(rois_b, rois_n, -1)
            sub_pool_feats = sub_pool_feats.view(rois_b, rois_n, -1)
            com_pool_feats = torch.cat((left_pool_feats, sub_pool_feats, right_pool_feats), dim=2)
            rois_pool_feats.append(sub_pool_feats)
            com_rois_feats.append(com_pool_feats)
        rois_pool_feats = torch.stack(rois_pool_feats, dim=2).view(rois_b * rois_n, len(fpn_feats), -1)
        com_rois_feats = torch.stack(com_rois_feats, dim=2).view(rois_b * rois_n, len(fpn_feats), -1)
        final_pool_feats =torch.max(rois_pool_feats,dim=1)[0]
        #final_pool_feats = torch.bmm(rois_mask, rois_pool_feats).squeeze(1)
        scorces_mask = (scorces > 0.5).float()
        #final_com_feats = torch.bmm(rois_mask, com_rois_feats).squeeze(1)
        # cls_log = self.classify(final_pool_feats).view(rois_b, rois_n, -1)
        final_com_feats = torch.max(com_rois_feats,dim=1)[0]

        final_pool_feats1 = final_pool_feats.view(rois_b, rois_n, -1)
        gt_pool_feats = final_pool_feats1[:, :gt_segments.size()[1], :]

        cls_log1, cls_gt, cls_node = self.classify1(final_pool_feats1, gcn_mask1, gcn_mask2, scorces_mask,
                                                    gt_pool_feats, rois)
        cls_log1 = self.relu(cls_log1)

        cls_log = torch.cat((final_pool_feats1, cls_log1), dim=2)
        cls_log = cls_log.view(rois_b * rois_n, -1)
        cls_log = self.classify(cls_log).view(rois_b, rois_n, -1)
        # com_log = self.compeletness(final_com_feats).view(rois_b, rois_n, -1)

        final_com_feats1 = final_com_feats.view(rois_b, rois_n, -1)
        gt_com_feats = final_com_feats1[:, :gt_segments.size()[1], :]

        com_log1, com_gt, com_node = self.compeletness1(final_com_feats1, gcn_mask1, gcn_mask2, scorces_mask,
                                                        gt_com_feats, rois)
        com_log1 = self.relu(com_log1)

        com_log = torch.cat((final_com_feats1, com_log1), dim=2)
        com_log = com_log.view(rois_b * rois_n, -1)
        com_log = self.compeletness(com_log).view(rois_b, rois_n, -1)

        # print(cls_log.size())

        # print(new_rois[0][0],2,fpn_feats[j].size()[2])
        return fpn_masks, out_cls_logits, out_offsets, rois.cuda(), scorces.cuda(), rois_mask, cls_log, com_log, cls_gt, cls_node, com_gt, com_node

        # return loss during training

    def _sample_nodes(self, rois):
        batch, num, _ = rois.size()
        new_rois = rois.new_zeros((rois.size()[0], rois.size()[1], 2))
        new_rois[:, :, 0] = rois[:, :, 1] - (rois[:, :, 2] - rois[:, :, 1]) * 0.5
        new_rois[:, :, 1] = rois[:, :, 2] + (rois[:, :, 2] - rois[:, :, 1]) * 0.5
        new_rois = new_rois.clamp(min=0.0, max=self.max_seq_len)
        # print(new_rois.size())
        left_rois = new_rois.unsqueeze(1).repeat(1, new_rois.size()[1], 1, 1)
        rigth_rois = new_rois.unsqueeze(2).repeat(1, 1, new_rois.size()[1], 1)
        min_left = torch.min(left_rois[:, :, :, 0], rigth_rois[:, :, :, 0])
        max_left = torch.max(left_rois[:, :, :, 0], rigth_rois[:, :, :, 0])
        min_right = torch.min(left_rois[:, :, :, 1], rigth_rois[:, :, :, 1])
        max_right = torch.max(left_rois[:, :, :, 1], rigth_rois[:, :, :, 1])
        ious = (min_right - max_left) / (max_right - min_left)
        ious_mask = (ious > 0).float()
        left_rois_center = (left_rois[:, :, :, 0] + left_rois[:, :, :, 1]) / 2.0
        rigth_rois_center = (rigth_rois[:, :, :, 0] + rigth_rois[:, :, :, 1]) / 2.0
        type_idxs = left_rois_center - rigth_rois_center
        ious_mask1 = ious_mask * (type_idxs >= 0).float()
        ious_mask2 = ious_mask * (type_idxs < 0).float()
        return ious_mask1, ious_mask2


class PtTransformer_gcn_loss_type_add_weight_all(nn.Module):
    """
        Transformer based model for single stage action localization
    """

    def __init__(
            self,
            model_configs,
            train_cfg,
    ):
        super().__init__()
        # re-distribute params to backbone / neck / head
        self.fpn_strides = [model_configs["scale_factor"] ** i for i in range(model_configs["backbone_arch"][-1] + 1)]
        self.reg_range = model_configs["regression_range"]
        assert len(self.fpn_strides) == len(self.reg_range)
        self.scale_factor = model_configs["scale_factor"]
        # check the feature pyramid and local attention window size
        self.max_seq_len = model_configs["max_seq_len"]
        if isinstance(model_configs["n_mha_win_size"], int):
            self.mha_win_size = [model_configs["n_mha_win_size"]] * len(self.fpn_strides)
        else:
            assert len(model_configs["n_mha_win_size"]) == len(self.fpn_strides)
            self.mha_win_size = model_configs["n_mha_win_size"]
        max_div_factor = 1
        for l, (s, w) in enumerate(zip(self.fpn_strides, self.mha_win_size)):
            stride = s * (w // 2) * 2 if w > 1 else s
            assert model_configs[
                       "max_seq_len"] % stride == 0, "max_seq_len must be divisible by fpn stride and window size"
            if max_div_factor < stride:
                max_div_factor = stride
        self.max_div_factor = max_div_factor

        # training time config
        self.train_cls_prior_prob = train_cfg['cls_prior_prob']
        self.train_dropout = train_cfg['dropout']
        self.use_offset = False
        self.train_droppath = train_cfg['droppath']
        self.fpn_levels = model_configs["backbone_arch"][-1] + 1
        self.regression_range = model_configs["regression_range"]
        self.num_classes = model_configs["num_classes"]
        # we will need a better way to dispatch the params to backbones / necks
        # backbone network: conv + transformer
        assert model_configs["backbone_type"] in ['convTransformer', 'conv']
        if model_configs["backbone_type"] == 'convTransformer':
            self.backbone = ConvTransformerBackbone(
                n_in=model_configs["input_dim"],
                n_embd=model_configs["embd_dim"],
                n_head=model_configs["n_head"],
                n_embd_ks=model_configs["embd_kernel_size"],
                max_len=model_configs["max_seq_len"],
                arch=model_configs["backbone_arch"],
                mha_win_size=self.mha_win_size,
                scale_factor=model_configs["scale_factor"],
                with_ln=model_configs["embd_with_ln"],
                attn_pdrop=0.0,
                proj_pdrop=self.train_dropout,
                path_pdrop=self.train_droppath,
                use_abs_pe=model_configs["use_abs_pe"],
                use_rel_pe=model_configs["use_rel_pe"]
            )
        else:
            self.backbone = ConvBackbone(n_in=model_configs["input_dim"], n_embd=model_configs["embd_dim"],
                                         n_embd_ks=model_configs["embd_kernel_size"],
                                         arch=model_configs["backbone_arch"],
                                         scale_factor=model_configs["scale_factor"],
                                         with_ln=model_configs["embd_with_ln"])
        # fpn network: convs
        assert model_configs["fpn_type"] in ['fpn', 'identity']
        if model_configs["fpn_type"] == 'fpn':
            self.neck = FPN1D(
                in_channels=[model_configs["embd_dim"]] * (model_configs["backbone_arch"][-1] + 1),
                out_channel=model_configs["fpn_dim"],
                scale_factor=model_configs["scale_factor"],
                with_ln=model_configs["fpn_with_ln"]
            )
        else:
            self.neck = FPNIdentity(
                in_channels=[model_configs["embd_dim"]] * (model_configs["backbone_arch"][-1] + 1),
                out_channel=model_configs["fpn_dim"],
                scale_factor=model_configs["scale_factor"],
                with_ln=model_configs["fpn_with_ln"]
            )
        self.point_generator = PointGenerator(
            max_seq_len=model_configs["max_seq_len"] * model_configs["max_buffer_len_factor"],
            fpn_levels=len(self.fpn_strides),
            scale_factor=model_configs["scale_factor"],
            regression_range=self.reg_range)
        # classfication and regerssion heads
        self.cls_head = PtTransformerClsHead(
            model_configs["fpn_dim"], model_configs["head_dim"], 1,
            kernel_size=model_configs["head_kernel_size"],
            prior_prob=self.train_cls_prior_prob,
            with_ln=model_configs["head_with_ln"],
            empty_cls=train_cfg['head_empty_cls']
        )
        self.reg_head = PtTransformerRegHead(
            model_configs["fpn_dim"], model_configs["head_dim"], len(self.fpn_strides),
            kernel_size=model_configs["head_kernel_size"],
            with_ln=model_configs["head_with_ln"]
        )
        '''
        for p in self.parameters():
            p.requires_grad=False
        '''

        self.pool = RoIPool(1, 1)
        self.proposal_creator = ProposalCreator(parent_model=self, nms_thresh=train_cfg["iou_threshold"],
                                                n_train_pre_nms=train_cfg["pre_train_nms_topk"],
                                                n_train_post_nms=train_cfg["train_max_seg_num"],
                                                n_test_pre_nms=train_cfg["pre_test_nms_topk"],
                                                n_test_post_nms=train_cfg["test_max_seg_num"],
                                                pre_nms_thresh=train_cfg["pre_nms_thresh"],
                                                min_size=train_cfg["duration_thresh"], sigma=train_cfg["nms_sigma"],
                                                min_score=train_cfg["min_score"])
        self.type = 2
        self.childs_num = 4

        self.classify = nn.Linear(model_configs["fpn_dim"] * 2, self.num_classes + 1)
        self.classify1 = Graph_module_net_0_loss_type(model_configs["fpn_dim"], model_configs["fpn_dim"] * 2,
                                                      model_configs["fpn_dim"], self.type, self.childs_num, dropout=0.7)
        self.relu = nn.ReLU()
        '''
        self.compeletness = nn.Sequential(
            MLP(model_configs["fpn_dim"] * 3, model_configs["fpn_dim"] * 6, model_configs["fpn_dim"] * 3, dropout=0.7),
            nn.ReLU(),
            nn.Linear(model_configs["fpn_dim"] * 3, self.num_classes)
        )

        self.classify = nn.Sequential(
            MLP(model_configs["fpn_dim"], model_configs["fpn_dim"] * 2, model_configs["fpn_dim"] , dropout=0.7),
            nn.ReLU(),
            nn.Linear(model_configs["fpn_dim"] , self.num_classes+1)
        )
        self.relu = nn.ReLU()
        '''
        self.compeletness = nn.Linear(model_configs["fpn_dim"] * 6, self.num_classes)
        self.compeletness1 = Graph_module_net_0_loss_type(model_configs["fpn_dim"] * 3, model_configs["fpn_dim"] * 6,
                                                          model_configs["fpn_dim"] * 3, self.type, self.childs_num,
                                                          dropout=0.7)
        self.pool_weights = nn.Parameter(torch.ones(len(self.fpn_strides)).float() / float(len(self.fpn_strides)))

    @property
    def device(self):
        # a hacky way to get the device type
        # will throw an error if parameters are on different devices
        return list(set(p.device for p in self.parameters()))[0]

    def forward(self, inputs, training=True):
        # batch the video list into feats (B, C, T) and masks (B, 1, T)
        if training:
            batched_inputs, batched_masks, gt_segments, gt_masks = inputs[0], inputs[1], inputs[2], inputs[3]
        else:
            batched_inputs, batched_masks = inputs[0], inputs[1]
            gt_segments = torch.ones(batched_inputs.size()[0], 100, 2)
            gt_masks = torch.ones(batched_inputs.size()[0], 100)
        batch = batched_inputs.size()[0]
        batched_inputs = batched_inputs.cuda()
        batched_masks = batched_masks.unsqueeze(1).cuda()

        # forward the network (backbone -> neck -> heads)
        feats, masks = self.backbone(batched_inputs, batched_masks)
        fpn_feats, fpn_masks = self.neck(feats, masks)
        # out_cls: List[B, #cls + 1, T_i]
        out_cls_logits = self.cls_head(fpn_feats, fpn_masks)
        # out_offset: List[B, 2, T_i]
        out_offsets = self.reg_head(fpn_feats, fpn_masks)
        points = self.point_generator(fpn_feats)
        # permute the outputs
        rois = []
        scorces = []
        rois_mask = []
        for i in range(batch):
            # out_cls: F List[B, #cls, T_i] -> F List[B, T_i, #cls]
            out_cls_logits_i = [x.permute(0, 2, 1)[i, :, :] for x in out_cls_logits]
            # out_offset: F List[B, 2 (xC), T_i] -> F List[B, T_i, 2 (xC)]
            out_offsets_i = [x.permute(0, 2, 1)[i, :, :] for x in out_offsets]
            # fpn_masks: F list[B, 1, T_i] -> F List[B, T_i]
            fpn_masks_i = [x.squeeze(1)[i, :] for x in fpn_masks]
            gt_segment = gt_segments[i]
            gt_mask = gt_masks[i]
            mask_idx = gt_mask > 0
            gt_segment = gt_segment[mask_idx]
            with torch.no_grad():
                rois_i, score_i, roi_mask_i = self.proposal_creator(out_offsets=out_offsets_i,
                                                                    out_cls_logits=out_cls_logits_i,
                                                                    fpn_masks=fpn_masks_i, points=points,
                                                                    gt_rois=gt_segment)
            batch_index = i * torch.ones((rois_i.size()[0],)).unsqueeze(1)
            new_rois_i = torch.cat((batch_index, rois_i), dim=1)
            rois.append(new_rois_i)
            scorces.append(score_i)
            rois_mask.append(roi_mask_i)
        rois = torch.stack(rois, dim=0)
        rois_b, rois_n, _ = rois.size()
        gcn_mask1, gcn_mask2 = self._sample_nodes(rois)
        scorces = torch.stack(scorces, dim=0)
        rois_mask = torch.stack(rois_mask, dim=0)
        out_cls_logits = [x.permute(0, 2, 1) for x in out_cls_logits]
        # out_offset: F List[B, 2 (xC), T_i] -> F List[B, T_i, 2 (xC)]
        out_offsets = [x.permute(0, 2, 1) for x in out_offsets]
        # fpn_masks: F list[B, 1, T_i] -> F List[B, T_i]
        fpn_masks = [x.squeeze(1) for x in fpn_masks]
        rois_mask = rois_mask.long()
        rois_mask = F.one_hot(
            rois_mask, num_classes=len(fpn_masks)
        )
        rois_mask = rois_mask.float().view(-1, 1, len(fpn_masks))
        rois_pool_feats = []
        com_rois_feats = []

        for j in range(len(fpn_feats)):
            new_rois = rois / self.fpn_strides[j]
            extend_rois = new_rois.new_zeros(new_rois.size()[0], new_rois.size()[1], 5)
            extend_rois[:, :, 0] = new_rois[:, :, 0]
            extend_rois[:, :, 1] = new_rois[:, :, 1]
            # extend_rois[:, :, 2] = 0
            extend_rois[:, :, 3] = new_rois[:, :, 2]
            # extend_rois[:, :, 4] = 0
            roi_feat = (fpn_feats[j] * fpn_masks[j][:, None, :]).unsqueeze(2)
            left_extend_rois = extend_rois.new_zeros(new_rois.size()[0], new_rois.size()[1], 5)
            right_extend_rois = extend_rois.new_zeros(new_rois.size()[0], new_rois.size()[1], 5)
            left_extend_rois[:, :, 0] = extend_rois[:, :, 0]
            left_extend_rois[:, :, 1] = extend_rois[:, :, 1] - (extend_rois[:, :, 3] - extend_rois[:, :, 1]) * 0.5
            left_extend_rois[:, :, 3] = extend_rois[:, :, 1]
            left_extend_rois[:, :, 1] = left_extend_rois[:, :, 1].clamp(min=0.0, max=self.max_seq_len)
            right_extend_rois[:, :, 0] = extend_rois[:, :, 0]
            right_extend_rois[:, :, 1] = extend_rois[:, :, 3]
            right_extend_rois[:, :, 3] = extend_rois[:, :, 3] + (extend_rois[:, :, 3] - extend_rois[:, :, 1]) * 0.5
            right_extend_rois[:, :, 1] = right_extend_rois[:, :, 1].clamp(min=0.0, max=self.max_seq_len)
            extend_rois = extend_rois.view(-1, 5).cuda()
            left_extend_rois = left_extend_rois.view(-1, 5).cuda()
            right_extend_rois = right_extend_rois.view(-1, 5).cuda()
            sub_pool_feats = self.pool(roi_feat, extend_rois)
            left_pool_feats = self.pool(roi_feat, left_extend_rois)
            left_pool_feats = left_pool_feats.view(rois_b, rois_n, -1)
            right_pool_feats = self.pool(roi_feat, right_extend_rois)
            right_pool_feats = right_pool_feats.view(rois_b, rois_n, -1)
            sub_pool_feats = sub_pool_feats.view(rois_b, rois_n, -1)
            com_pool_feats = torch.cat((left_pool_feats, sub_pool_feats, right_pool_feats), dim=2)
            rois_pool_feats.append(sub_pool_feats)
            com_rois_feats.append(com_pool_feats)
        rois_pool_feats = torch.stack(rois_pool_feats, dim=2).view(rois_b * rois_n, len(fpn_feats), -1)
        com_rois_feats = torch.stack(com_rois_feats, dim=2).view(rois_b * rois_n, len(fpn_feats), -1)
        rois_pool_feats=rois_pool_feats*self.pool_weights[None,:,None]
        com_rois_feats=com_rois_feats*self.pool_weights[None,:,None]
        final_pool_feats =torch.sum(rois_pool_feats,dim=1)
        #final_pool_feats = torch.bmm(rois_mask, rois_pool_feats).squeeze(1)
        scorces_mask = (scorces > 0.5).float()
        #final_com_feats = torch.bmm(rois_mask, com_rois_feats).squeeze(1)
        # cls_log = self.classify(final_pool_feats).view(rois_b, rois_n, -1)
        final_com_feats = torch.sum(com_rois_feats,dim=1)

        final_pool_feats1 = final_pool_feats.view(rois_b, rois_n, -1)
        gt_pool_feats = final_pool_feats1[:, :gt_segments.size()[1], :]

        cls_log1, cls_gt, cls_node = self.classify1(final_pool_feats1, gcn_mask1, gcn_mask2, scorces_mask,
                                                    gt_pool_feats, rois)
        cls_log1 = self.relu(cls_log1)

        cls_log = torch.cat((final_pool_feats1, cls_log1), dim=2)
        cls_log = cls_log.view(rois_b * rois_n, -1)
        cls_log = self.classify(cls_log).view(rois_b, rois_n, -1)
        # com_log = self.compeletness(final_com_feats).view(rois_b, rois_n, -1)

        final_com_feats1 = final_com_feats.view(rois_b, rois_n, -1)
        gt_com_feats = final_com_feats1[:, :gt_segments.size()[1], :]

        com_log1, com_gt, com_node = self.compeletness1(final_com_feats1, gcn_mask1, gcn_mask2, scorces_mask,
                                                        gt_com_feats, rois)
        com_log1 = self.relu(com_log1)

        com_log = torch.cat((final_com_feats1, com_log1), dim=2)
        com_log = com_log.view(rois_b * rois_n, -1)
        com_log = self.compeletness(com_log).view(rois_b, rois_n, -1)

        # print(cls_log.size())

        # print(new_rois[0][0],2,fpn_feats[j].size()[2])
        return fpn_masks, out_cls_logits, out_offsets, rois.cuda(), scorces.cuda(), rois_mask, cls_log, com_log, cls_gt, cls_node, com_gt, com_node

        # return loss during training

    def _sample_nodes(self, rois):
        batch, num, _ = rois.size()
        new_rois = rois.new_zeros((rois.size()[0], rois.size()[1], 2))
        new_rois[:, :, 0] = rois[:, :, 1] - (rois[:, :, 2] - rois[:, :, 1]) * 0.5
        new_rois[:, :, 1] = rois[:, :, 2] + (rois[:, :, 2] - rois[:, :, 1]) * 0.5
        new_rois = new_rois.clamp(min=0.0, max=self.max_seq_len)
        # print(new_rois.size())
        left_rois = new_rois.unsqueeze(1).repeat(1, new_rois.size()[1], 1, 1)
        rigth_rois = new_rois.unsqueeze(2).repeat(1, 1, new_rois.size()[1], 1)
        min_left = torch.min(left_rois[:, :, :, 0], rigth_rois[:, :, :, 0])
        max_left = torch.max(left_rois[:, :, :, 0], rigth_rois[:, :, :, 0])
        min_right = torch.min(left_rois[:, :, :, 1], rigth_rois[:, :, :, 1])
        max_right = torch.max(left_rois[:, :, :, 1], rigth_rois[:, :, :, 1])
        ious = (min_right - max_left) / (max_right - min_left)
        ious_mask = (ious > 0).float()
        left_rois_center = (left_rois[:, :, :, 0] + left_rois[:, :, :, 1]) / 2.0
        rigth_rois_center = (rigth_rois[:, :, :, 0] + rigth_rois[:, :, :, 1]) / 2.0
        type_idxs = left_rois_center - rigth_rois_center
        ious_mask1 = ious_mask * (type_idxs >= 0).float()
        ious_mask2 = ious_mask * (type_idxs < 0).float()
        return ious_mask1, ious_mask2


class PtTransformer_gcn_loss_type_cat(nn.Module):
    """
        Transformer based model for single stage action localization
    """

    def __init__(
            self,
            model_configs,
            train_cfg,
    ):
        super().__init__()
        # re-distribute params to backbone / neck / head
        self.fpn_strides = [model_configs["scale_factor"] ** i for i in range(model_configs["backbone_arch"][-1] + 1)]
        self.reg_range = model_configs["regression_range"]
        assert len(self.fpn_strides) == len(self.reg_range)
        self.scale_factor = model_configs["scale_factor"]
        # check the feature pyramid and local attention window size
        self.max_seq_len = model_configs["max_seq_len"]
        if isinstance(model_configs["n_mha_win_size"], int):
            self.mha_win_size = [model_configs["n_mha_win_size"]] * len(self.fpn_strides)
        else:
            assert len(model_configs["n_mha_win_size"]) == len(self.fpn_strides)
            self.mha_win_size = model_configs["n_mha_win_size"]
        max_div_factor = 1
        for l, (s, w) in enumerate(zip(self.fpn_strides, self.mha_win_size)):
            stride = s * (w // 2) * 2 if w > 1 else s
            assert model_configs[
                       "max_seq_len"] % stride == 0, "max_seq_len must be divisible by fpn stride and window size"
            if max_div_factor < stride:
                max_div_factor = stride
        self.max_div_factor = max_div_factor

        # training time config
        self.train_cls_prior_prob = train_cfg['cls_prior_prob']
        self.train_dropout = train_cfg['dropout']
        self.use_offset = False
        self.train_droppath = train_cfg['droppath']
        self.fpn_levels = model_configs["backbone_arch"][-1] + 1
        self.regression_range = model_configs["regression_range"]
        self.num_classes = model_configs["num_classes"]
        # we will need a better way to dispatch the params to backbones / necks
        # backbone network: conv + transformer
        assert model_configs["backbone_type"] in ['convTransformer', 'conv']
        if model_configs["backbone_type"] == 'convTransformer':
            self.backbone = ConvTransformerBackbone(
                n_in=model_configs["input_dim"],
                n_embd=model_configs["embd_dim"],
                n_head=model_configs["n_head"],
                n_embd_ks=model_configs["embd_kernel_size"],
                max_len=model_configs["max_seq_len"],
                arch=model_configs["backbone_arch"],
                mha_win_size=self.mha_win_size,
                scale_factor=model_configs["scale_factor"],
                with_ln=model_configs["embd_with_ln"],
                attn_pdrop=0.0,
                proj_pdrop=self.train_dropout,
                path_pdrop=self.train_droppath,
                use_abs_pe=model_configs["use_abs_pe"],
                use_rel_pe=model_configs["use_rel_pe"]
            )
        else:
            self.backbone = ConvBackbone(n_in=model_configs["input_dim"], n_embd=model_configs["embd_dim"],
                                         n_embd_ks=model_configs["embd_kernel_size"],
                                         arch=model_configs["backbone_arch"],
                                         scale_factor=model_configs["scale_factor"],
                                         with_ln=model_configs["embd_with_ln"])
        # fpn network: convs
        assert model_configs["fpn_type"] in ['fpn', 'identity']
        if model_configs["fpn_type"] == 'fpn':
            self.neck = FPN1D(
                in_channels=[model_configs["embd_dim"]] * (model_configs["backbone_arch"][-1] + 1),
                out_channel=model_configs["fpn_dim"],
                scale_factor=model_configs["scale_factor"],
                with_ln=model_configs["fpn_with_ln"]
            )
        else:
            self.neck = FPNIdentity(
                in_channels=[model_configs["embd_dim"]] * (model_configs["backbone_arch"][-1] + 1),
                out_channel=model_configs["fpn_dim"],
                scale_factor=model_configs["scale_factor"],
                with_ln=model_configs["fpn_with_ln"]
            )
        self.point_generator = PointGenerator(
            max_seq_len=model_configs["max_seq_len"] * model_configs["max_buffer_len_factor"],
            fpn_levels=len(self.fpn_strides),
            scale_factor=model_configs["scale_factor"],
            regression_range=self.reg_range)
        # classfication and regerssion heads
        self.cls_head = PtTransformerClsHead(
            model_configs["fpn_dim"], model_configs["head_dim"], 1,
            kernel_size=model_configs["head_kernel_size"],
            prior_prob=self.train_cls_prior_prob,
            with_ln=model_configs["head_with_ln"],
            empty_cls=train_cfg['head_empty_cls']
        )
        self.reg_head = PtTransformerRegHead(
            model_configs["fpn_dim"], model_configs["head_dim"], len(self.fpn_strides),
            kernel_size=model_configs["head_kernel_size"],
            with_ln=model_configs["head_with_ln"]
        )
        '''
        for p in self.parameters():
            p.requires_grad=False
        '''

        self.pool = RoIPool(1, 1)
        self.proposal_creator = ProposalCreator(parent_model=self, nms_thresh=train_cfg["iou_threshold"],
                                                n_train_pre_nms=train_cfg["pre_train_nms_topk"],
                                                n_train_post_nms=train_cfg["train_max_seg_num"],
                                                n_test_pre_nms=train_cfg["pre_test_nms_topk"],
                                                n_test_post_nms=train_cfg["test_max_seg_num"],
                                                pre_nms_thresh=train_cfg["pre_nms_thresh"],
                                                min_size=train_cfg["duration_thresh"], sigma=train_cfg["nms_sigma"],
                                                min_score=train_cfg["min_score"])
        self.type = 2
        self.childs_num = 4

        self.classify = nn.Linear(model_configs["fpn_dim"] * 2, self.num_classes + 1)
        self.classify1 = Graph_module_net_0_loss_type(model_configs["fpn_dim"], model_configs["fpn_dim"] * 2,
                                                      model_configs["fpn_dim"], self.type, self.childs_num, dropout=0.7)
        self.relu = nn.ReLU()
        '''
        self.compeletness = nn.Sequential(
            MLP(model_configs["fpn_dim"] * 3, model_configs["fpn_dim"] * 6, model_configs["fpn_dim"] * 3, dropout=0.7),
            nn.ReLU(),
            nn.Linear(model_configs["fpn_dim"] * 3, self.num_classes)
        )

        self.classify = nn.Sequential(
            MLP(model_configs["fpn_dim"], model_configs["fpn_dim"] * 2, model_configs["fpn_dim"] , dropout=0.7),
            nn.ReLU(),
            nn.Linear(model_configs["fpn_dim"] , self.num_classes+1)
        )
        self.relu = nn.ReLU()
        '''
        self.compeletness = nn.Linear(model_configs["fpn_dim"] * 6, self.num_classes)
        self.compeletness1 = Graph_module_net_0_loss_type(model_configs["fpn_dim"] * 3, model_configs["fpn_dim"] * 6,
                                                          model_configs["fpn_dim"] * 3, self.type, self.childs_num,
                                                          dropout=0.7)
        self.class_pool = nn.Sequential(
            nn.Linear(model_configs["fpn_dim"] * 3, model_configs["fpn_dim"]),
            nn.ReLU())
        self.com_pool = nn.Sequential(
            nn.Linear(model_configs["fpn_dim"] * 3 * 3, model_configs["fpn_dim"] * 3),
            nn.ReLU())

    @property
    def device(self):
        # a hacky way to get the device type
        # will throw an error if parameters are on different devices
        return list(set(p.device for p in self.parameters()))[0]

    def forward(self, inputs, training=True):
        # batch the video list into feats (B, C, T) and masks (B, 1, T)
        if training:
            batched_inputs, batched_masks, gt_segments, gt_masks = inputs[0], inputs[1], inputs[2], inputs[3]
        else:
            batched_inputs, batched_masks = inputs[0], inputs[1]
            gt_segments = torch.ones(batched_inputs.size()[0], 100, 2)
            gt_masks = torch.ones(batched_inputs.size()[0], 100)
        batch = batched_inputs.size()[0]
        batched_inputs = batched_inputs.cuda()
        batched_masks = batched_masks.unsqueeze(1).cuda()

        # forward the network (backbone -> neck -> heads)
        feats, masks = self.backbone(batched_inputs, batched_masks)
        fpn_feats, fpn_masks = self.neck(feats, masks)
        # out_cls: List[B, #cls + 1, T_i]
        out_cls_logits = self.cls_head(fpn_feats, fpn_masks)
        # out_offset: List[B, 2, T_i]
        out_offsets = self.reg_head(fpn_feats, fpn_masks)
        points = self.point_generator(fpn_feats)
        # permute the outputs
        rois = []
        scorces = []
        rois_mask = []
        for i in range(batch):
            # out_cls: F List[B, #cls, T_i] -> F List[B, T_i, #cls]
            out_cls_logits_i = [x.permute(0, 2, 1)[i, :, :] for x in out_cls_logits]
            # out_offset: F List[B, 2 (xC), T_i] -> F List[B, T_i, 2 (xC)]
            out_offsets_i = [x.permute(0, 2, 1)[i, :, :] for x in out_offsets]
            # fpn_masks: F list[B, 1, T_i] -> F List[B, T_i]
            fpn_masks_i = [x.squeeze(1)[i, :] for x in fpn_masks]
            gt_segment = gt_segments[i]
            gt_mask = gt_masks[i]
            mask_idx = gt_mask > 0
            gt_segment = gt_segment[mask_idx]
            with torch.no_grad():
                rois_i, score_i, roi_mask_i = self.proposal_creator(out_offsets=out_offsets_i,
                                                                    out_cls_logits=out_cls_logits_i,
                                                                    fpn_masks=fpn_masks_i, points=points,
                                                                    gt_rois=gt_segment)
            batch_index = i * torch.ones((rois_i.size()[0],)).unsqueeze(1)
            new_rois_i = torch.cat((batch_index, rois_i), dim=1)
            rois.append(new_rois_i)
            scorces.append(score_i)
            rois_mask.append(roi_mask_i)
        rois = torch.stack(rois, dim=0)
        rois_b, rois_n, _ = rois.size()
        gcn_mask1, gcn_mask2 = self._sample_nodes(rois)
        scorces = torch.stack(scorces, dim=0)
        rois_mask = torch.stack(rois_mask, dim=0)
        out_cls_logits = [x.permute(0, 2, 1) for x in out_cls_logits]
        # out_offset: F List[B, 2 (xC), T_i] -> F List[B, T_i, 2 (xC)]
        out_offsets = [x.permute(0, 2, 1) for x in out_offsets]
        # fpn_masks: F list[B, 1, T_i] -> F List[B, T_i]
        fpn_masks = [x.squeeze(1) for x in fpn_masks]
        rois_mask = rois_mask.long()
        rois_mask1=rois_mask+1
        rois_mask1=rois_mask1.clamp(min=0, max=len(fpn_masks)-1)
        rois_mask1 = rois_mask1.long()
        rois_mask2=rois_mask-1
        rois_mask2 = rois_mask2.clamp(min=0, max=len(fpn_masks) - 1)
        rois_mask2 = rois_mask2.long()
        rois_mask = F.one_hot(
            rois_mask, num_classes=len(fpn_masks)
        )
        rois_mask = rois_mask.float().view(-1, 1, len(fpn_masks))
        rois_mask1 = F.one_hot(
            rois_mask1, num_classes=len(fpn_masks)
        )
        rois_mask1 = rois_mask1.float().view(-1, 1, len(fpn_masks))
        rois_mask2 = F.one_hot(
            rois_mask2, num_classes=len(fpn_masks)
        )
        rois_mask2 = rois_mask2.float().view(-1, 1, len(fpn_masks))
        rois_pool_feats = []
        com_rois_feats = []

        for j in range(len(fpn_feats)):
            new_rois = rois / self.fpn_strides[j]
            extend_rois = new_rois.new_zeros(new_rois.size()[0], new_rois.size()[1], 5)
            extend_rois[:, :, 0] = new_rois[:, :, 0]
            extend_rois[:, :, 1] = new_rois[:, :, 1]
            # extend_rois[:, :, 2] = 0
            extend_rois[:, :, 3] = new_rois[:, :, 2]
            # extend_rois[:, :, 4] = 0
            roi_feat = (fpn_feats[j] * fpn_masks[j][:, None, :]).unsqueeze(2)
            left_extend_rois = extend_rois.new_zeros(new_rois.size()[0], new_rois.size()[1], 5)
            right_extend_rois = extend_rois.new_zeros(new_rois.size()[0], new_rois.size()[1], 5)
            left_extend_rois[:, :, 0] = extend_rois[:, :, 0]
            left_extend_rois[:, :, 1] = extend_rois[:, :, 1] - (extend_rois[:, :, 3] - extend_rois[:, :, 1]) * 0.5
            left_extend_rois[:, :, 3] = extend_rois[:, :, 1]
            left_extend_rois[:, :, 1] = left_extend_rois[:, :, 1].clamp(min=0.0, max=self.max_seq_len)
            right_extend_rois[:, :, 0] = extend_rois[:, :, 0]
            right_extend_rois[:, :, 1] = extend_rois[:, :, 3]
            right_extend_rois[:, :, 3] = extend_rois[:, :, 3] + (extend_rois[:, :, 3] - extend_rois[:, :, 1]) * 0.5
            right_extend_rois[:, :, 1] = right_extend_rois[:, :, 1].clamp(min=0.0, max=self.max_seq_len)
            extend_rois = extend_rois.view(-1, 5).cuda()
            left_extend_rois = left_extend_rois.view(-1, 5).cuda()
            right_extend_rois = right_extend_rois.view(-1, 5).cuda()
            sub_pool_feats = self.pool(roi_feat, extend_rois)
            left_pool_feats = self.pool(roi_feat, left_extend_rois)
            left_pool_feats = left_pool_feats.view(rois_b, rois_n, -1)
            right_pool_feats = self.pool(roi_feat, right_extend_rois)
            right_pool_feats = right_pool_feats.view(rois_b, rois_n, -1)
            sub_pool_feats = sub_pool_feats.view(rois_b, rois_n, -1)
            com_pool_feats = torch.cat((left_pool_feats, sub_pool_feats, right_pool_feats), dim=2)
            rois_pool_feats.append(sub_pool_feats)
            com_rois_feats.append(com_pool_feats)
        rois_pool_feats = torch.stack(rois_pool_feats, dim=2).view(rois_b * rois_n, len(fpn_feats), -1)
        com_rois_feats = torch.stack(com_rois_feats, dim=2).view(rois_b * rois_n, len(fpn_feats), -1)
        final_pool_feats_1 = torch.bmm(rois_mask, rois_pool_feats).squeeze(1)
        final_pool_feats_2 = torch.bmm(rois_mask1, rois_pool_feats).squeeze(1)
        final_pool_feats_3 = torch.bmm(rois_mask2, rois_pool_feats).squeeze(1)
        final_pool_feats=torch.stack((final_pool_feats_1,final_pool_feats_2,final_pool_feats_3),dim=1).view(rois_b * rois_n, -1)
        final_pool_feats=self.class_pool(final_pool_feats)

        scorces_mask = (scorces > 0.5).float()
        final_com_feats_1 = torch.bmm(rois_mask, com_rois_feats).squeeze(1)
        final_com_feats_2 = torch.bmm(rois_mask1, com_rois_feats).squeeze(1)
        final_com_feats_3 = torch.bmm(rois_mask2, com_rois_feats).squeeze(1)
        final_com_feats = torch.stack((final_com_feats_1, final_com_feats_2, final_com_feats_3), dim=1).view(rois_b * rois_n,-1)
        final_com_feats=self.com_pool(final_com_feats)
        # cls_log = self.classify(final_pool_feats).view(rois_b, rois_n, -1)

        final_pool_feats1 = final_pool_feats.view(rois_b, rois_n, -1)
        gt_pool_feats = final_pool_feats1[:, :gt_segments.size()[1], :]

        cls_log1, cls_gt, cls_node = self.classify1(final_pool_feats1, gcn_mask1, gcn_mask2, scorces_mask,
                                                    gt_pool_feats, rois)
        cls_log1 = self.relu(cls_log1)

        cls_log = torch.cat((final_pool_feats1, cls_log1), dim=2)
        cls_log = cls_log.view(rois_b * rois_n, -1)
        cls_log = self.classify(cls_log).view(rois_b, rois_n, -1)
        # com_log = self.compeletness(final_com_feats).view(rois_b, rois_n, -1)

        final_com_feats1 = final_com_feats.view(rois_b, rois_n, -1)
        gt_com_feats = final_com_feats1[:, :gt_segments.size()[1], :]

        com_log1, com_gt, com_node = self.compeletness1(final_com_feats1, gcn_mask1, gcn_mask2, scorces_mask,
                                                        gt_com_feats, rois)
        com_log1 = self.relu(com_log1)

        com_log = torch.cat((final_com_feats1, com_log1), dim=2)
        com_log = com_log.view(rois_b * rois_n, -1)
        com_log = self.compeletness(com_log).view(rois_b, rois_n, -1)

        # print(cls_log.size())

        # print(new_rois[0][0],2,fpn_feats[j].size()[2])
        return fpn_masks, out_cls_logits, out_offsets, rois.cuda(), scorces.cuda(), rois_mask, cls_log, com_log, cls_gt, cls_node, com_gt, com_node

        # return loss during training

    def _sample_nodes(self, rois):
        batch, num, _ = rois.size()
        new_rois = rois.new_zeros((rois.size()[0], rois.size()[1], 2))
        new_rois[:, :, 0] = rois[:, :, 1] - (rois[:, :, 2] - rois[:, :, 1]) * 0.5
        new_rois[:, :, 1] = rois[:, :, 2] + (rois[:, :, 2] - rois[:, :, 1]) * 0.5
        new_rois = new_rois.clamp(min=0.0, max=self.max_seq_len)
        # print(new_rois.size())
        left_rois = new_rois.unsqueeze(1).repeat(1, new_rois.size()[1], 1, 1)
        rigth_rois = new_rois.unsqueeze(2).repeat(1, 1, new_rois.size()[1], 1)
        min_left = torch.min(left_rois[:, :, :, 0], rigth_rois[:, :, :, 0])
        max_left = torch.max(left_rois[:, :, :, 0], rigth_rois[:, :, :, 0])
        min_right = torch.min(left_rois[:, :, :, 1], rigth_rois[:, :, :, 1])
        max_right = torch.max(left_rois[:, :, :, 1], rigth_rois[:, :, :, 1])
        ious = (min_right - max_left) / (max_right - min_left)
        ious_mask = (ious > 0).float()
        left_rois_center = (left_rois[:, :, :, 0] + left_rois[:, :, :, 1]) / 2.0
        rigth_rois_center = (rigth_rois[:, :, :, 0] + rigth_rois[:, :, :, 1]) / 2.0
        type_idxs = left_rois_center - rigth_rois_center
        ious_mask1 = ious_mask * (type_idxs >= 0).float()
        ious_mask2 = ious_mask * (type_idxs < 0).float()
        return ious_mask1, ious_mask2


class PtTransformer_gcn_loss_type_add(nn.Module):
    """
        Transformer based model for single stage action localization
    """

    def __init__(
            self,
            model_configs,
            train_cfg,
    ):
        super().__init__()
        # re-distribute params to backbone / neck / head
        self.fpn_strides = [model_configs["scale_factor"] ** i for i in range(model_configs["backbone_arch"][-1] + 1)]
        self.reg_range = model_configs["regression_range"]
        assert len(self.fpn_strides) == len(self.reg_range)
        self.scale_factor = model_configs["scale_factor"]
        # check the feature pyramid and local attention window size
        self.max_seq_len = model_configs["max_seq_len"]
        if isinstance(model_configs["n_mha_win_size"], int):
            self.mha_win_size = [model_configs["n_mha_win_size"]] * len(self.fpn_strides)
        else:
            assert len(model_configs["n_mha_win_size"]) == len(self.fpn_strides)
            self.mha_win_size = model_configs["n_mha_win_size"]
        max_div_factor = 1
        for l, (s, w) in enumerate(zip(self.fpn_strides, self.mha_win_size)):
            stride = s * (w // 2) * 2 if w > 1 else s
            assert model_configs[
                       "max_seq_len"] % stride == 0, "max_seq_len must be divisible by fpn stride and window size"
            if max_div_factor < stride:
                max_div_factor = stride
        self.max_div_factor = max_div_factor

        # training time config
        self.train_cls_prior_prob = train_cfg['cls_prior_prob']
        self.train_dropout = train_cfg['dropout']
        self.use_offset = False
        self.train_droppath = train_cfg['droppath']
        self.fpn_levels = model_configs["backbone_arch"][-1] + 1
        self.regression_range = model_configs["regression_range"]
        self.num_classes = model_configs["num_classes"]
        # we will need a better way to dispatch the params to backbones / necks
        # backbone network: conv + transformer
        assert model_configs["backbone_type"] in ['convTransformer', 'conv']
        if model_configs["backbone_type"] == 'convTransformer':
            self.backbone = ConvTransformerBackbone(
                n_in=model_configs["input_dim"],
                n_embd=model_configs["embd_dim"],
                n_head=model_configs["n_head"],
                n_embd_ks=model_configs["embd_kernel_size"],
                max_len=model_configs["max_seq_len"],
                arch=model_configs["backbone_arch"],
                mha_win_size=self.mha_win_size,
                scale_factor=model_configs["scale_factor"],
                with_ln=model_configs["embd_with_ln"],
                attn_pdrop=0.0,
                proj_pdrop=self.train_dropout,
                path_pdrop=self.train_droppath,
                use_abs_pe=model_configs["use_abs_pe"],
                use_rel_pe=model_configs["use_rel_pe"]
            )
        else:
            self.backbone = ConvBackbone(n_in=model_configs["input_dim"], n_embd=model_configs["embd_dim"],
                                         n_embd_ks=model_configs["embd_kernel_size"],
                                         arch=model_configs["backbone_arch"],
                                         scale_factor=model_configs["scale_factor"],
                                         with_ln=model_configs["embd_with_ln"])
        # fpn network: convs
        assert model_configs["fpn_type"] in ['fpn', 'identity']
        if model_configs["fpn_type"] == 'fpn':
            self.neck = FPN1D(
                in_channels=[model_configs["embd_dim"]] * (model_configs["backbone_arch"][-1] + 1),
                out_channel=model_configs["fpn_dim"],
                scale_factor=model_configs["scale_factor"],
                with_ln=model_configs["fpn_with_ln"]
            )
        else:
            self.neck = FPNIdentity(
                in_channels=[model_configs["embd_dim"]] * (model_configs["backbone_arch"][-1] + 1),
                out_channel=model_configs["fpn_dim"],
                scale_factor=model_configs["scale_factor"],
                with_ln=model_configs["fpn_with_ln"]
            )
        self.point_generator = PointGenerator(
            max_seq_len=model_configs["max_seq_len"] * model_configs["max_buffer_len_factor"],
            fpn_levels=len(self.fpn_strides),
            scale_factor=model_configs["scale_factor"],
            regression_range=self.reg_range)
        # classfication and regerssion heads
        self.cls_head = PtTransformerClsHead(
            model_configs["fpn_dim"], model_configs["head_dim"], 1,
            kernel_size=model_configs["head_kernel_size"],
            prior_prob=self.train_cls_prior_prob,
            with_ln=model_configs["head_with_ln"],
            empty_cls=train_cfg['head_empty_cls']
        )
        self.reg_head = PtTransformerRegHead(
            model_configs["fpn_dim"], model_configs["head_dim"], len(self.fpn_strides),
            kernel_size=model_configs["head_kernel_size"],
            with_ln=model_configs["head_with_ln"]
        )
        '''
        for p in self.parameters():
            p.requires_grad=False
        '''

        self.pool = RoIPool(1, 1)
        self.proposal_creator = ProposalCreator(parent_model=self, nms_thresh=train_cfg["iou_threshold"],
                                                n_train_pre_nms=train_cfg["pre_train_nms_topk"],
                                                n_train_post_nms=train_cfg["train_max_seg_num"],
                                                n_test_pre_nms=train_cfg["pre_test_nms_topk"],
                                                n_test_post_nms=train_cfg["test_max_seg_num"],
                                                pre_nms_thresh=train_cfg["pre_nms_thresh"],
                                                min_size=train_cfg["duration_thresh"], sigma=train_cfg["nms_sigma"],
                                                min_score=train_cfg["min_score"])
        self.type = 2
        self.childs_num = 4

        self.classify = nn.Linear(model_configs["fpn_dim"] * 2, self.num_classes + 1)
        self.classify1 = Graph_module_net_0_loss_type(model_configs["fpn_dim"], model_configs["fpn_dim"] * 2,
                                                      model_configs["fpn_dim"], self.type, self.childs_num, dropout=0.7)
        self.relu = nn.ReLU()
        '''
        self.compeletness = nn.Sequential(
            MLP(model_configs["fpn_dim"] * 3, model_configs["fpn_dim"] * 6, model_configs["fpn_dim"] * 3, dropout=0.7),
            nn.ReLU(),
            nn.Linear(model_configs["fpn_dim"] * 3, self.num_classes)
        )

        self.classify = nn.Sequential(
            MLP(model_configs["fpn_dim"], model_configs["fpn_dim"] * 2, model_configs["fpn_dim"] , dropout=0.7),
            nn.ReLU(),
            nn.Linear(model_configs["fpn_dim"] , self.num_classes+1)
        )
        self.relu = nn.ReLU()
        '''
        self.compeletness = nn.Linear(model_configs["fpn_dim"] * 6, self.num_classes)
        self.compeletness1 = Graph_module_net_0_loss_type(model_configs["fpn_dim"] * 3, model_configs["fpn_dim"] * 6,
                                                          model_configs["fpn_dim"] * 3, self.type, self.childs_num,
                                                          dropout=0.7)

    @property
    def device(self):
        # a hacky way to get the device type
        # will throw an error if parameters are on different devices
        return list(set(p.device for p in self.parameters()))[0]

    def forward(self, inputs, training=True):
        # batch the video list into feats (B, C, T) and masks (B, 1, T)
        if training:
            batched_inputs, batched_masks, gt_segments, gt_masks = inputs[0], inputs[1], inputs[2], inputs[3]
        else:
            batched_inputs, batched_masks = inputs[0], inputs[1]
            gt_segments = torch.ones(batched_inputs.size()[0], 100, 2)
            gt_masks = torch.ones(batched_inputs.size()[0], 100)
        batch = batched_inputs.size()[0]
        batched_inputs = batched_inputs.cuda()
        batched_masks = batched_masks.unsqueeze(1).cuda()

        # forward the network (backbone -> neck -> heads)
        feats, masks = self.backbone(batched_inputs, batched_masks)
        fpn_feats, fpn_masks = self.neck(feats, masks)
        # out_cls: List[B, #cls + 1, T_i]
        out_cls_logits = self.cls_head(fpn_feats, fpn_masks)
        # out_offset: List[B, 2, T_i]
        out_offsets = self.reg_head(fpn_feats, fpn_masks)
        points = self.point_generator(fpn_feats)
        # permute the outputs
        rois = []
        scorces = []
        rois_mask = []
        for i in range(batch):
            # out_cls: F List[B, #cls, T_i] -> F List[B, T_i, #cls]
            out_cls_logits_i = [x.permute(0, 2, 1)[i, :, :] for x in out_cls_logits]
            # out_offset: F List[B, 2 (xC), T_i] -> F List[B, T_i, 2 (xC)]
            out_offsets_i = [x.permute(0, 2, 1)[i, :, :] for x in out_offsets]
            # fpn_masks: F list[B, 1, T_i] -> F List[B, T_i]
            fpn_masks_i = [x.squeeze(1)[i, :] for x in fpn_masks]
            gt_segment = gt_segments[i]
            gt_mask = gt_masks[i]
            mask_idx = gt_mask > 0
            gt_segment = gt_segment[mask_idx]
            with torch.no_grad():
                rois_i, score_i, roi_mask_i = self.proposal_creator(out_offsets=out_offsets_i,
                                                                    out_cls_logits=out_cls_logits_i,
                                                                    fpn_masks=fpn_masks_i, points=points,
                                                                    gt_rois=gt_segment)
            batch_index = i * torch.ones((rois_i.size()[0],)).unsqueeze(1)
            new_rois_i = torch.cat((batch_index, rois_i), dim=1)
            rois.append(new_rois_i)
            scorces.append(score_i)
            rois_mask.append(roi_mask_i)
        rois = torch.stack(rois, dim=0)
        rois_b, rois_n, _ = rois.size()
        gcn_mask1, gcn_mask2 = self._sample_nodes(rois)
        scorces = torch.stack(scorces, dim=0)
        rois_mask = torch.stack(rois_mask, dim=0)
        out_cls_logits = [x.permute(0, 2, 1) for x in out_cls_logits]
        # out_offset: F List[B, 2 (xC), T_i] -> F List[B, T_i, 2 (xC)]
        out_offsets = [x.permute(0, 2, 1) for x in out_offsets]
        # fpn_masks: F list[B, 1, T_i] -> F List[B, T_i]
        fpn_masks = [x.squeeze(1) for x in fpn_masks]
        rois_mask = rois_mask.long()
        rois_mask1=rois_mask+1
        rois_mask1=rois_mask1.clamp(min=0, max=len(fpn_masks)-1)
        rois_mask1 = rois_mask1.long()
        rois_mask2=rois_mask-1
        rois_mask2 = rois_mask2.clamp(min=0, max=len(fpn_masks) - 1)
        rois_mask2 = rois_mask2.long()
        rois_mask = F.one_hot(
            rois_mask, num_classes=len(fpn_masks)
        )
        rois_mask = rois_mask.float().view(-1, 1, len(fpn_masks))
        rois_mask1 = F.one_hot(
            rois_mask1, num_classes=len(fpn_masks)
        )
        rois_mask1 = rois_mask1.float().view(-1, 1, len(fpn_masks))
        rois_mask2 = F.one_hot(
            rois_mask2, num_classes=len(fpn_masks)
        )
        rois_mask2 = rois_mask2.float().view(-1, 1, len(fpn_masks))
        rois_pool_feats = []
        com_rois_feats = []

        for j in range(len(fpn_feats)):
            new_rois = rois / self.fpn_strides[j]
            extend_rois = new_rois.new_zeros(new_rois.size()[0], new_rois.size()[1], 5)
            extend_rois[:, :, 0] = new_rois[:, :, 0]
            extend_rois[:, :, 1] = new_rois[:, :, 1]
            # extend_rois[:, :, 2] = 0
            extend_rois[:, :, 3] = new_rois[:, :, 2]
            # extend_rois[:, :, 4] = 0
            roi_feat = (fpn_feats[j] * fpn_masks[j][:, None, :]).unsqueeze(2)
            left_extend_rois = extend_rois.new_zeros(new_rois.size()[0], new_rois.size()[1], 5)
            right_extend_rois = extend_rois.new_zeros(new_rois.size()[0], new_rois.size()[1], 5)
            left_extend_rois[:, :, 0] = extend_rois[:, :, 0]
            left_extend_rois[:, :, 1] = extend_rois[:, :, 1] - (extend_rois[:, :, 3] - extend_rois[:, :, 1]) * 0.5
            left_extend_rois[:, :, 3] = extend_rois[:, :, 1]
            left_extend_rois[:, :, 1] = left_extend_rois[:, :, 1].clamp(min=0.0, max=self.max_seq_len)
            right_extend_rois[:, :, 0] = extend_rois[:, :, 0]
            right_extend_rois[:, :, 1] = extend_rois[:, :, 3]
            right_extend_rois[:, :, 3] = extend_rois[:, :, 3] + (extend_rois[:, :, 3] - extend_rois[:, :, 1]) * 0.5
            right_extend_rois[:, :, 1] = right_extend_rois[:, :, 1].clamp(min=0.0, max=self.max_seq_len)
            extend_rois = extend_rois.view(-1, 5).cuda()
            left_extend_rois = left_extend_rois.view(-1, 5).cuda()
            right_extend_rois = right_extend_rois.view(-1, 5).cuda()
            sub_pool_feats = self.pool(roi_feat, extend_rois)
            left_pool_feats = self.pool(roi_feat, left_extend_rois)
            left_pool_feats = left_pool_feats.view(rois_b, rois_n, -1)
            right_pool_feats = self.pool(roi_feat, right_extend_rois)
            right_pool_feats = right_pool_feats.view(rois_b, rois_n, -1)
            sub_pool_feats = sub_pool_feats.view(rois_b, rois_n, -1)
            com_pool_feats = torch.cat((left_pool_feats, sub_pool_feats, right_pool_feats), dim=2)
            rois_pool_feats.append(sub_pool_feats)
            com_rois_feats.append(com_pool_feats)
        rois_pool_feats = torch.stack(rois_pool_feats, dim=2).view(rois_b * rois_n, len(fpn_feats), -1)
        com_rois_feats = torch.stack(com_rois_feats, dim=2).view(rois_b * rois_n, len(fpn_feats), -1)
        final_pool_feats_1 = torch.bmm(rois_mask, rois_pool_feats).squeeze(1)
        final_pool_feats_2 = torch.bmm(rois_mask1, rois_pool_feats).squeeze(1)
        final_pool_feats_3 = torch.bmm(rois_mask2, rois_pool_feats).squeeze(1)
        final_pool_feats=torch.stack((final_pool_feats_1,final_pool_feats_2,final_pool_feats_3),dim=1)
        final_pool_feats=torch.mean(final_pool_feats,dim=1)

        scorces_mask = (scorces > 0.5).float()
        final_com_feats_1 = torch.bmm(rois_mask, com_rois_feats).squeeze(1)
        final_com_feats_2 = torch.bmm(rois_mask1, com_rois_feats).squeeze(1)
        final_com_feats_3 = torch.bmm(rois_mask2, com_rois_feats).squeeze(1)
        final_com_feats = torch.stack((final_com_feats_1, final_com_feats_2, final_com_feats_3), dim=1)
        final_com_feats=torch.mean(final_com_feats,dim=1)
        # cls_log = self.classify(final_pool_feats).view(rois_b, rois_n, -1)

        final_pool_feats1 = final_pool_feats.view(rois_b, rois_n, -1)
        gt_pool_feats = final_pool_feats1[:, :gt_segments.size()[1], :]

        cls_log1, cls_gt, cls_node = self.classify1(final_pool_feats1, gcn_mask1, gcn_mask2, scorces_mask,
                                                    gt_pool_feats, rois)
        cls_log1 = self.relu(cls_log1)

        cls_log = torch.cat((final_pool_feats1, cls_log1), dim=2)
        cls_log = cls_log.view(rois_b * rois_n, -1)
        cls_log = self.classify(cls_log).view(rois_b, rois_n, -1)
        # com_log = self.compeletness(final_com_feats).view(rois_b, rois_n, -1)

        final_com_feats1 = final_com_feats.view(rois_b, rois_n, -1)
        gt_com_feats = final_com_feats1[:, :gt_segments.size()[1], :]

        com_log1, com_gt, com_node = self.compeletness1(final_com_feats1, gcn_mask1, gcn_mask2, scorces_mask,
                                                        gt_com_feats, rois)
        com_log1 = self.relu(com_log1)

        com_log = torch.cat((final_com_feats1, com_log1), dim=2)
        com_log = com_log.view(rois_b * rois_n, -1)
        com_log = self.compeletness(com_log).view(rois_b, rois_n, -1)

        # print(cls_log.size())

        # print(new_rois[0][0],2,fpn_feats[j].size()[2])
        return fpn_masks, out_cls_logits, out_offsets, rois.cuda(), scorces.cuda(), rois_mask, cls_log, com_log, cls_gt, cls_node, com_gt, com_node

        # return loss during training

    def _sample_nodes(self, rois):
        batch, num, _ = rois.size()
        new_rois = rois.new_zeros((rois.size()[0], rois.size()[1], 2))
        new_rois[:, :, 0] = rois[:, :, 1] - (rois[:, :, 2] - rois[:, :, 1]) * 0.5
        new_rois[:, :, 1] = rois[:, :, 2] + (rois[:, :, 2] - rois[:, :, 1]) * 0.5
        new_rois = new_rois.clamp(min=0.0, max=self.max_seq_len)
        # print(new_rois.size())
        left_rois = new_rois.unsqueeze(1).repeat(1, new_rois.size()[1], 1, 1)
        rigth_rois = new_rois.unsqueeze(2).repeat(1, 1, new_rois.size()[1], 1)
        min_left = torch.min(left_rois[:, :, :, 0], rigth_rois[:, :, :, 0])
        max_left = torch.max(left_rois[:, :, :, 0], rigth_rois[:, :, :, 0])
        min_right = torch.min(left_rois[:, :, :, 1], rigth_rois[:, :, :, 1])
        max_right = torch.max(left_rois[:, :, :, 1], rigth_rois[:, :, :, 1])
        ious = (min_right - max_left) / (max_right - min_left)
        ious_mask = (ious > 0).float()
        left_rois_center = (left_rois[:, :, :, 0] + left_rois[:, :, :, 1]) / 2.0
        rigth_rois_center = (rigth_rois[:, :, :, 0] + rigth_rois[:, :, :, 1]) / 2.0
        type_idxs = left_rois_center - rigth_rois_center
        ious_mask1 = ious_mask * (type_idxs >= 0).float()
        ious_mask2 = ious_mask * (type_idxs < 0).float()
        return ious_mask1, ious_mask2



class PtTransformer_gcn_loss_type_max(nn.Module):
    """
        Transformer based model for single stage action localization
    """

    def __init__(
            self,
            model_configs,
            train_cfg,
    ):
        super().__init__()
        # re-distribute params to backbone / neck / head
        self.fpn_strides = [model_configs["scale_factor"] ** i for i in range(model_configs["backbone_arch"][-1] + 1)]
        self.reg_range = model_configs["regression_range"]
        assert len(self.fpn_strides) == len(self.reg_range)
        self.scale_factor = model_configs["scale_factor"]
        # check the feature pyramid and local attention window size
        self.max_seq_len = model_configs["max_seq_len"]
        if isinstance(model_configs["n_mha_win_size"], int):
            self.mha_win_size = [model_configs["n_mha_win_size"]] * len(self.fpn_strides)
        else:
            assert len(model_configs["n_mha_win_size"]) == len(self.fpn_strides)
            self.mha_win_size = model_configs["n_mha_win_size"]
        max_div_factor = 1
        for l, (s, w) in enumerate(zip(self.fpn_strides, self.mha_win_size)):
            stride = s * (w // 2) * 2 if w > 1 else s
            assert model_configs[
                       "max_seq_len"] % stride == 0, "max_seq_len must be divisible by fpn stride and window size"
            if max_div_factor < stride:
                max_div_factor = stride
        self.max_div_factor = max_div_factor

        # training time config
        self.train_cls_prior_prob = train_cfg['cls_prior_prob']
        self.train_dropout = train_cfg['dropout']
        self.use_offset = False
        self.train_droppath = train_cfg['droppath']
        self.fpn_levels = model_configs["backbone_arch"][-1] + 1
        self.regression_range = model_configs["regression_range"]
        self.num_classes = model_configs["num_classes"]
        # we will need a better way to dispatch the params to backbones / necks
        # backbone network: conv + transformer
        assert model_configs["backbone_type"] in ['convTransformer', 'conv']
        if model_configs["backbone_type"] == 'convTransformer':
            self.backbone = ConvTransformerBackbone(
                n_in=model_configs["input_dim"],
                n_embd=model_configs["embd_dim"],
                n_head=model_configs["n_head"],
                n_embd_ks=model_configs["embd_kernel_size"],
                max_len=model_configs["max_seq_len"],
                arch=model_configs["backbone_arch"],
                mha_win_size=self.mha_win_size,
                scale_factor=model_configs["scale_factor"],
                with_ln=model_configs["embd_with_ln"],
                attn_pdrop=0.0,
                proj_pdrop=self.train_dropout,
                path_pdrop=self.train_droppath,
                use_abs_pe=model_configs["use_abs_pe"],
                use_rel_pe=model_configs["use_rel_pe"]
            )
        else:
            self.backbone = ConvBackbone(n_in=model_configs["input_dim"], n_embd=model_configs["embd_dim"],
                                         n_embd_ks=model_configs["embd_kernel_size"],
                                         arch=model_configs["backbone_arch"],
                                         scale_factor=model_configs["scale_factor"],
                                         with_ln=model_configs["embd_with_ln"])
        # fpn network: convs
        assert model_configs["fpn_type"] in ['fpn', 'identity']
        if model_configs["fpn_type"] == 'fpn':
            self.neck = FPN1D(
                in_channels=[model_configs["embd_dim"]] * (model_configs["backbone_arch"][-1] + 1),
                out_channel=model_configs["fpn_dim"],
                scale_factor=model_configs["scale_factor"],
                with_ln=model_configs["fpn_with_ln"]
            )
        else:
            self.neck = FPNIdentity(
                in_channels=[model_configs["embd_dim"]] * (model_configs["backbone_arch"][-1] + 1),
                out_channel=model_configs["fpn_dim"],
                scale_factor=model_configs["scale_factor"],
                with_ln=model_configs["fpn_with_ln"]
            )
        self.point_generator = PointGenerator(
            max_seq_len=model_configs["max_seq_len"] * model_configs["max_buffer_len_factor"],
            fpn_levels=len(self.fpn_strides),
            scale_factor=model_configs["scale_factor"],
            regression_range=self.reg_range)
        # classfication and regerssion heads
        self.cls_head = PtTransformerClsHead(
            model_configs["fpn_dim"], model_configs["head_dim"], 1,
            kernel_size=model_configs["head_kernel_size"],
            prior_prob=self.train_cls_prior_prob,
            with_ln=model_configs["head_with_ln"],
            empty_cls=train_cfg['head_empty_cls']
        )
        self.reg_head = PtTransformerRegHead(
            model_configs["fpn_dim"], model_configs["head_dim"], len(self.fpn_strides),
            kernel_size=model_configs["head_kernel_size"],
            with_ln=model_configs["head_with_ln"]
        )
        '''
        for p in self.parameters():
            p.requires_grad=False
        '''

        self.pool = RoIPool(1, 1)
        self.proposal_creator = ProposalCreator(parent_model=self, nms_thresh=train_cfg["iou_threshold"],
                                                n_train_pre_nms=train_cfg["pre_train_nms_topk"],
                                                n_train_post_nms=train_cfg["train_max_seg_num"],
                                                n_test_pre_nms=train_cfg["pre_test_nms_topk"],
                                                n_test_post_nms=train_cfg["test_max_seg_num"],
                                                pre_nms_thresh=train_cfg["pre_nms_thresh"],
                                                min_size=train_cfg["duration_thresh"], sigma=train_cfg["nms_sigma"],
                                                min_score=train_cfg["min_score"])
        self.type = 2
        self.childs_num = 4

        self.classify = nn.Linear(model_configs["fpn_dim"] * 2, self.num_classes + 1)
        self.classify1 = Graph_module_net_0_loss_type(model_configs["fpn_dim"], model_configs["fpn_dim"] * 2,
                                                      model_configs["fpn_dim"], self.type, self.childs_num, dropout=0.7)
        self.relu = nn.ReLU()
        '''
        self.compeletness = nn.Sequential(
            MLP(model_configs["fpn_dim"] * 3, model_configs["fpn_dim"] * 6, model_configs["fpn_dim"] * 3, dropout=0.7),
            nn.ReLU(),
            nn.Linear(model_configs["fpn_dim"] * 3, self.num_classes)
        )

        self.classify = nn.Sequential(
            MLP(model_configs["fpn_dim"], model_configs["fpn_dim"] * 2, model_configs["fpn_dim"] , dropout=0.7),
            nn.ReLU(),
            nn.Linear(model_configs["fpn_dim"] , self.num_classes+1)
        )
        self.relu = nn.ReLU()
        '''
        self.compeletness = nn.Linear(model_configs["fpn_dim"] * 6, self.num_classes)
        self.compeletness1 = Graph_module_net_0_loss_type(model_configs["fpn_dim"] * 3, model_configs["fpn_dim"] * 6,
                                                          model_configs["fpn_dim"] * 3, self.type, self.childs_num,
                                                          dropout=0.7)

    @property
    def device(self):
        # a hacky way to get the device type
        # will throw an error if parameters are on different devices
        return list(set(p.device for p in self.parameters()))[0]

    def forward(self, inputs, training=True):
        # batch the video list into feats (B, C, T) and masks (B, 1, T)
        if training:
            batched_inputs, batched_masks, gt_segments, gt_masks = inputs[0], inputs[1], inputs[2], inputs[3]
        else:
            batched_inputs, batched_masks = inputs[0], inputs[1]
            gt_segments = torch.ones(batched_inputs.size()[0], 100, 2)
            gt_masks = torch.ones(batched_inputs.size()[0], 100)
        batch = batched_inputs.size()[0]
        batched_inputs = batched_inputs.cuda()
        batched_masks = batched_masks.unsqueeze(1).cuda()

        # forward the network (backbone -> neck -> heads)
        feats, masks = self.backbone(batched_inputs, batched_masks)
        fpn_feats, fpn_masks = self.neck(feats, masks)
        # out_cls: List[B, #cls + 1, T_i]
        out_cls_logits = self.cls_head(fpn_feats, fpn_masks)
        # out_offset: List[B, 2, T_i]
        out_offsets = self.reg_head(fpn_feats, fpn_masks)
        points = self.point_generator(fpn_feats)
        # permute the outputs
        rois = []
        scorces = []
        rois_mask = []
        for i in range(batch):
            # out_cls: F List[B, #cls, T_i] -> F List[B, T_i, #cls]
            out_cls_logits_i = [x.permute(0, 2, 1)[i, :, :] for x in out_cls_logits]
            # out_offset: F List[B, 2 (xC), T_i] -> F List[B, T_i, 2 (xC)]
            out_offsets_i = [x.permute(0, 2, 1)[i, :, :] for x in out_offsets]
            # fpn_masks: F list[B, 1, T_i] -> F List[B, T_i]
            fpn_masks_i = [x.squeeze(1)[i, :] for x in fpn_masks]
            gt_segment = gt_segments[i]
            gt_mask = gt_masks[i]
            mask_idx = gt_mask > 0
            gt_segment = gt_segment[mask_idx]
            with torch.no_grad():
                rois_i, score_i, roi_mask_i = self.proposal_creator(out_offsets=out_offsets_i,
                                                                    out_cls_logits=out_cls_logits_i,
                                                                    fpn_masks=fpn_masks_i, points=points,
                                                                    gt_rois=gt_segment)
            batch_index = i * torch.ones((rois_i.size()[0],)).unsqueeze(1)
            new_rois_i = torch.cat((batch_index, rois_i), dim=1)
            rois.append(new_rois_i)
            scorces.append(score_i)
            rois_mask.append(roi_mask_i)
        rois = torch.stack(rois, dim=0)
        rois_b, rois_n, _ = rois.size()
        gcn_mask1, gcn_mask2 = self._sample_nodes(rois)
        scorces = torch.stack(scorces, dim=0)
        rois_mask = torch.stack(rois_mask, dim=0)
        out_cls_logits = [x.permute(0, 2, 1) for x in out_cls_logits]
        # out_offset: F List[B, 2 (xC), T_i] -> F List[B, T_i, 2 (xC)]
        out_offsets = [x.permute(0, 2, 1) for x in out_offsets]
        # fpn_masks: F list[B, 1, T_i] -> F List[B, T_i]
        fpn_masks = [x.squeeze(1) for x in fpn_masks]
        rois_mask = rois_mask.long()
        rois_mask1=rois_mask+1
        rois_mask1=rois_mask1.clamp(min=0, max=len(fpn_masks)-1)
        rois_mask1 = rois_mask1.long()
        rois_mask2=rois_mask-1
        rois_mask2 = rois_mask2.clamp(min=0, max=len(fpn_masks) - 1)
        rois_mask2 = rois_mask2.long()
        rois_mask = F.one_hot(
            rois_mask, num_classes=len(fpn_masks)
        )
        rois_mask = rois_mask.float().view(-1, 1, len(fpn_masks))
        rois_mask1 = F.one_hot(
            rois_mask1, num_classes=len(fpn_masks)
        )
        rois_mask1 = rois_mask1.float().view(-1, 1, len(fpn_masks))
        rois_mask2 = F.one_hot(
            rois_mask2, num_classes=len(fpn_masks)
        )
        rois_mask2 = rois_mask2.float().view(-1, 1, len(fpn_masks))
        rois_pool_feats = []
        com_rois_feats = []

        for j in range(len(fpn_feats)):
            new_rois = rois / self.fpn_strides[j]
            extend_rois = new_rois.new_zeros(new_rois.size()[0], new_rois.size()[1], 5)
            extend_rois[:, :, 0] = new_rois[:, :, 0]
            extend_rois[:, :, 1] = new_rois[:, :, 1]
            # extend_rois[:, :, 2] = 0
            extend_rois[:, :, 3] = new_rois[:, :, 2]
            # extend_rois[:, :, 4] = 0
            roi_feat = (fpn_feats[j] * fpn_masks[j][:, None, :]).unsqueeze(2)
            left_extend_rois = extend_rois.new_zeros(new_rois.size()[0], new_rois.size()[1], 5)
            right_extend_rois = extend_rois.new_zeros(new_rois.size()[0], new_rois.size()[1], 5)
            left_extend_rois[:, :, 0] = extend_rois[:, :, 0]
            left_extend_rois[:, :, 1] = extend_rois[:, :, 1] - (extend_rois[:, :, 3] - extend_rois[:, :, 1]) * 0.5
            left_extend_rois[:, :, 3] = extend_rois[:, :, 1]
            left_extend_rois[:, :, 1] = left_extend_rois[:, :, 1].clamp(min=0.0, max=self.max_seq_len)
            right_extend_rois[:, :, 0] = extend_rois[:, :, 0]
            right_extend_rois[:, :, 1] = extend_rois[:, :, 3]
            right_extend_rois[:, :, 3] = extend_rois[:, :, 3] + (extend_rois[:, :, 3] - extend_rois[:, :, 1]) * 0.5
            right_extend_rois[:, :, 1] = right_extend_rois[:, :, 1].clamp(min=0.0, max=self.max_seq_len)
            extend_rois = extend_rois.view(-1, 5).cuda()
            left_extend_rois = left_extend_rois.view(-1, 5).cuda()
            right_extend_rois = right_extend_rois.view(-1, 5).cuda()
            sub_pool_feats = self.pool(roi_feat, extend_rois)
            left_pool_feats = self.pool(roi_feat, left_extend_rois)
            left_pool_feats = left_pool_feats.view(rois_b, rois_n, -1)
            right_pool_feats = self.pool(roi_feat, right_extend_rois)
            right_pool_feats = right_pool_feats.view(rois_b, rois_n, -1)
            sub_pool_feats = sub_pool_feats.view(rois_b, rois_n, -1)
            com_pool_feats = torch.cat((left_pool_feats, sub_pool_feats, right_pool_feats), dim=2)
            rois_pool_feats.append(sub_pool_feats)
            com_rois_feats.append(com_pool_feats)
        rois_pool_feats = torch.stack(rois_pool_feats, dim=2).view(rois_b * rois_n, len(fpn_feats), -1)
        com_rois_feats = torch.stack(com_rois_feats, dim=2).view(rois_b * rois_n, len(fpn_feats), -1)
        final_pool_feats_1 = torch.bmm(rois_mask, rois_pool_feats).squeeze(1)
        final_pool_feats_2 = torch.bmm(rois_mask1, rois_pool_feats).squeeze(1)
        final_pool_feats_3 = torch.bmm(rois_mask2, rois_pool_feats).squeeze(1)
        final_pool_feats=torch.stack((final_pool_feats_1,final_pool_feats_2,final_pool_feats_3),dim=1)
        final_pool_feats=torch.max(final_pool_feats,dim=1)[0]

        scorces_mask = (scorces > 0.5).float()
        final_com_feats_1 = torch.bmm(rois_mask, com_rois_feats).squeeze(1)
        final_com_feats_2 = torch.bmm(rois_mask1, com_rois_feats).squeeze(1)
        final_com_feats_3 = torch.bmm(rois_mask2, com_rois_feats).squeeze(1)
        final_com_feats = torch.stack((final_com_feats_1, final_com_feats_2, final_com_feats_3), dim=1)
        final_com_feats=torch.max(final_com_feats,dim=1)[0]
        # cls_log = self.classify(final_pool_feats).view(rois_b, rois_n, -1)

        final_pool_feats1 = final_pool_feats.view(rois_b, rois_n, -1)
        gt_pool_feats = final_pool_feats1[:, :gt_segments.size()[1], :]

        cls_log1, cls_gt, cls_node = self.classify1(final_pool_feats1, gcn_mask1, gcn_mask2, scorces_mask,
                                                    gt_pool_feats, rois)
        cls_log1 = self.relu(cls_log1)

        cls_log = torch.cat((final_pool_feats1, cls_log1), dim=2)
        cls_log = cls_log.view(rois_b * rois_n, -1)
        cls_log = self.classify(cls_log).view(rois_b, rois_n, -1)
        # com_log = self.compeletness(final_com_feats).view(rois_b, rois_n, -1)

        final_com_feats1 = final_com_feats.view(rois_b, rois_n, -1)
        gt_com_feats = final_com_feats1[:, :gt_segments.size()[1], :]

        com_log1, com_gt, com_node = self.compeletness1(final_com_feats1, gcn_mask1, gcn_mask2, scorces_mask,
                                                        gt_com_feats, rois)
        com_log1 = self.relu(com_log1)

        com_log = torch.cat((final_com_feats1, com_log1), dim=2)
        com_log = com_log.view(rois_b * rois_n, -1)
        com_log = self.compeletness(com_log).view(rois_b, rois_n, -1)

        # print(cls_log.size())

        # print(new_rois[0][0],2,fpn_feats[j].size()[2])
        return fpn_masks, out_cls_logits, out_offsets, rois.cuda(), scorces.cuda(), rois_mask, cls_log, com_log, cls_gt, cls_node, com_gt, com_node

        # return loss during training

    def _sample_nodes(self, rois):
        batch, num, _ = rois.size()
        new_rois = rois.new_zeros((rois.size()[0], rois.size()[1], 2))
        new_rois[:, :, 0] = rois[:, :, 1] - (rois[:, :, 2] - rois[:, :, 1]) * 0.5
        new_rois[:, :, 1] = rois[:, :, 2] + (rois[:, :, 2] - rois[:, :, 1]) * 0.5
        new_rois = new_rois.clamp(min=0.0, max=self.max_seq_len)
        # print(new_rois.size())
        left_rois = new_rois.unsqueeze(1).repeat(1, new_rois.size()[1], 1, 1)
        rigth_rois = new_rois.unsqueeze(2).repeat(1, 1, new_rois.size()[1], 1)
        min_left = torch.min(left_rois[:, :, :, 0], rigth_rois[:, :, :, 0])
        max_left = torch.max(left_rois[:, :, :, 0], rigth_rois[:, :, :, 0])
        min_right = torch.min(left_rois[:, :, :, 1], rigth_rois[:, :, :, 1])
        max_right = torch.max(left_rois[:, :, :, 1], rigth_rois[:, :, :, 1])
        ious = (min_right - max_left) / (max_right - min_left)
        ious_mask = (ious > 0).float()
        left_rois_center = (left_rois[:, :, :, 0] + left_rois[:, :, :, 1]) / 2.0
        rigth_rois_center = (rigth_rois[:, :, :, 0] + rigth_rois[:, :, :, 1]) / 2.0
        type_idxs = left_rois_center - rigth_rois_center
        ious_mask1 = ious_mask * (type_idxs >= 0).float()
        ious_mask2 = ious_mask * (type_idxs < 0).float()
        return ious_mask1, ious_mask2


class PtTransformer_gcn_loss_type_add_weight(nn.Module):
    """
        Transformer based model for single stage action localization
    """

    def __init__(
            self,
            model_configs,
            train_cfg,
    ):
        super().__init__()
        # re-distribute params to backbone / neck / head
        self.fpn_strides = [model_configs["scale_factor"] ** i for i in range(model_configs["backbone_arch"][-1] + 1)]
        self.reg_range = model_configs["regression_range"]
        assert len(self.fpn_strides) == len(self.reg_range)
        self.scale_factor = model_configs["scale_factor"]
        # check the feature pyramid and local attention window size
        self.max_seq_len = model_configs["max_seq_len"]
        if isinstance(model_configs["n_mha_win_size"], int):
            self.mha_win_size = [model_configs["n_mha_win_size"]] * len(self.fpn_strides)
        else:
            assert len(model_configs["n_mha_win_size"]) == len(self.fpn_strides)
            self.mha_win_size = model_configs["n_mha_win_size"]
        max_div_factor = 1
        for l, (s, w) in enumerate(zip(self.fpn_strides, self.mha_win_size)):
            stride = s * (w // 2) * 2 if w > 1 else s
            assert model_configs[
                       "max_seq_len"] % stride == 0, "max_seq_len must be divisible by fpn stride and window size"
            if max_div_factor < stride:
                max_div_factor = stride
        self.max_div_factor = max_div_factor

        # training time config
        self.train_cls_prior_prob = train_cfg['cls_prior_prob']
        self.train_dropout = train_cfg['dropout']
        self.use_offset = False
        self.train_droppath = train_cfg['droppath']
        self.fpn_levels = model_configs["backbone_arch"][-1] + 1
        self.regression_range = model_configs["regression_range"]
        self.num_classes = model_configs["num_classes"]
        # we will need a better way to dispatch the params to backbones / necks
        # backbone network: conv + transformer
        assert model_configs["backbone_type"] in ['convTransformer', 'conv']
        if model_configs["backbone_type"] == 'convTransformer':
            self.backbone = ConvTransformerBackbone(
                n_in=model_configs["input_dim"],
                n_embd=model_configs["embd_dim"],
                n_head=model_configs["n_head"],
                n_embd_ks=model_configs["embd_kernel_size"],
                max_len=model_configs["max_seq_len"],
                arch=model_configs["backbone_arch"],
                mha_win_size=self.mha_win_size,
                scale_factor=model_configs["scale_factor"],
                with_ln=model_configs["embd_with_ln"],
                attn_pdrop=0.0,
                proj_pdrop=self.train_dropout,
                path_pdrop=self.train_droppath,
                use_abs_pe=model_configs["use_abs_pe"],
                use_rel_pe=model_configs["use_rel_pe"]
            )
        else:
            self.backbone = ConvBackbone(n_in=model_configs["input_dim"], n_embd=model_configs["embd_dim"],
                                         n_embd_ks=model_configs["embd_kernel_size"],
                                         arch=model_configs["backbone_arch"],
                                         scale_factor=model_configs["scale_factor"],
                                         with_ln=model_configs["embd_with_ln"])
        # fpn network: convs
        assert model_configs["fpn_type"] in ['fpn', 'identity']
        if model_configs["fpn_type"] == 'fpn':
            self.neck = FPN1D(
                in_channels=[model_configs["embd_dim"]] * (model_configs["backbone_arch"][-1] + 1),
                out_channel=model_configs["fpn_dim"],
                scale_factor=model_configs["scale_factor"],
                with_ln=model_configs["fpn_with_ln"]
            )
        else:
            self.neck = FPNIdentity(
                in_channels=[model_configs["embd_dim"]] * (model_configs["backbone_arch"][-1] + 1),
                out_channel=model_configs["fpn_dim"],
                scale_factor=model_configs["scale_factor"],
                with_ln=model_configs["fpn_with_ln"]
            )
        self.point_generator = PointGenerator(
            max_seq_len=model_configs["max_seq_len"] * model_configs["max_buffer_len_factor"],
            fpn_levels=len(self.fpn_strides),
            scale_factor=model_configs["scale_factor"],
            regression_range=self.reg_range)
        # classfication and regerssion heads
        self.cls_head = PtTransformerClsHead(
            model_configs["fpn_dim"], model_configs["head_dim"], 1,
            kernel_size=model_configs["head_kernel_size"],
            prior_prob=self.train_cls_prior_prob,
            with_ln=model_configs["head_with_ln"],
            empty_cls=train_cfg['head_empty_cls']
        )
        self.reg_head = PtTransformerRegHead(
            model_configs["fpn_dim"], model_configs["head_dim"], len(self.fpn_strides),
            kernel_size=model_configs["head_kernel_size"],
            with_ln=model_configs["head_with_ln"]
        )
        '''
        for p in self.parameters():
            p.requires_grad=False
        '''

        self.pool = RoIPool(1, 1)
        self.proposal_creator = ProposalCreator(parent_model=self, nms_thresh=train_cfg["iou_threshold"],
                                                n_train_pre_nms=train_cfg["pre_train_nms_topk"],
                                                n_train_post_nms=train_cfg["train_max_seg_num"],
                                                n_test_pre_nms=train_cfg["pre_test_nms_topk"],
                                                n_test_post_nms=train_cfg["test_max_seg_num"],
                                                pre_nms_thresh=train_cfg["pre_nms_thresh"],
                                                min_size=train_cfg["duration_thresh"], sigma=train_cfg["nms_sigma"],
                                                min_score=train_cfg["min_score"])
        self.type = 2
        self.childs_num = 4

        self.classify = nn.Linear(model_configs["fpn_dim"] * 2, self.num_classes + 1)
        self.classify1 = Graph_module_net_0_loss_type(model_configs["fpn_dim"], model_configs["fpn_dim"] * 2,
                                                      model_configs["fpn_dim"], self.type, self.childs_num, dropout=0.7)
        self.relu = nn.ReLU()
        '''
        self.compeletness = nn.Sequential(
            MLP(model_configs["fpn_dim"] * 3, model_configs["fpn_dim"] * 6, model_configs["fpn_dim"] * 3, dropout=0.7),
            nn.ReLU(),
            nn.Linear(model_configs["fpn_dim"] * 3, self.num_classes)
        )

        self.classify = nn.Sequential(
            MLP(model_configs["fpn_dim"], model_configs["fpn_dim"] * 2, model_configs["fpn_dim"] , dropout=0.7),
            nn.ReLU(),
            nn.Linear(model_configs["fpn_dim"] , self.num_classes+1)
        )
        self.relu = nn.ReLU()
        '''
        self.compeletness = nn.Linear(model_configs["fpn_dim"] * 6, self.num_classes)
        self.compeletness1 = Graph_module_net_0_loss_type(model_configs["fpn_dim"] * 3, model_configs["fpn_dim"] * 6,
                                                          model_configs["fpn_dim"] * 3, self.type, self.childs_num,
                                                          dropout=0.7)
        self.pool_weights = nn.Parameter(torch.ones(3).float() / float(3))

    @property
    def device(self):
        # a hacky way to get the device type
        # will throw an error if parameters are on different devices
        return list(set(p.device for p in self.parameters()))[0]

    def forward(self, inputs, training=True):
        # batch the video list into feats (B, C, T) and masks (B, 1, T)
        if training:
            batched_inputs, batched_masks, gt_segments, gt_masks = inputs[0], inputs[1], inputs[2], inputs[3]
        else:
            batched_inputs, batched_masks = inputs[0], inputs[1]
            gt_segments = torch.ones(batched_inputs.size()[0], 100, 2)
            gt_masks = torch.ones(batched_inputs.size()[0], 100)
        batch = batched_inputs.size()[0]
        batched_inputs = batched_inputs.cuda()
        batched_masks = batched_masks.unsqueeze(1).cuda()

        # forward the network (backbone -> neck -> heads)
        feats, masks = self.backbone(batched_inputs, batched_masks)
        fpn_feats, fpn_masks = self.neck(feats, masks)
        # out_cls: List[B, #cls + 1, T_i]
        out_cls_logits = self.cls_head(fpn_feats, fpn_masks)
        # out_offset: List[B, 2, T_i]
        out_offsets = self.reg_head(fpn_feats, fpn_masks)
        points = self.point_generator(fpn_feats)
        # permute the outputs
        rois = []
        scorces = []
        rois_mask = []
        for i in range(batch):
            # out_cls: F List[B, #cls, T_i] -> F List[B, T_i, #cls]
            out_cls_logits_i = [x.permute(0, 2, 1)[i, :, :] for x in out_cls_logits]
            # out_offset: F List[B, 2 (xC), T_i] -> F List[B, T_i, 2 (xC)]
            out_offsets_i = [x.permute(0, 2, 1)[i, :, :] for x in out_offsets]
            # fpn_masks: F list[B, 1, T_i] -> F List[B, T_i]
            fpn_masks_i = [x.squeeze(1)[i, :] for x in fpn_masks]
            gt_segment = gt_segments[i]
            gt_mask = gt_masks[i]
            mask_idx = gt_mask > 0
            gt_segment = gt_segment[mask_idx]
            with torch.no_grad():
                rois_i, score_i, roi_mask_i = self.proposal_creator(out_offsets=out_offsets_i,
                                                                    out_cls_logits=out_cls_logits_i,
                                                                    fpn_masks=fpn_masks_i, points=points,
                                                                    gt_rois=gt_segment)
            batch_index = i * torch.ones((rois_i.size()[0],)).unsqueeze(1)
            new_rois_i = torch.cat((batch_index, rois_i), dim=1)
            rois.append(new_rois_i)
            scorces.append(score_i)
            rois_mask.append(roi_mask_i)
        rois = torch.stack(rois, dim=0)
        rois_b, rois_n, _ = rois.size()
        gcn_mask1, gcn_mask2 = self._sample_nodes(rois)
        scorces = torch.stack(scorces, dim=0)
        rois_mask = torch.stack(rois_mask, dim=0)
        out_cls_logits = [x.permute(0, 2, 1) for x in out_cls_logits]
        # out_offset: F List[B, 2 (xC), T_i] -> F List[B, T_i, 2 (xC)]
        out_offsets = [x.permute(0, 2, 1) for x in out_offsets]
        # fpn_masks: F list[B, 1, T_i] -> F List[B, T_i]
        fpn_masks = [x.squeeze(1) for x in fpn_masks]
        rois_mask = rois_mask.long()
        rois_mask1=rois_mask+1
        rois_mask1=rois_mask1.clamp(min=0, max=len(fpn_masks)-1)
        rois_mask1 = rois_mask1.long()
        rois_mask2=rois_mask-1
        rois_mask2 = rois_mask2.clamp(min=0, max=len(fpn_masks) - 1)
        rois_mask2 = rois_mask2.long()
        rois_mask = F.one_hot(
            rois_mask, num_classes=len(fpn_masks)
        )
        rois_mask = rois_mask.float().view(-1, 1, len(fpn_masks))
        rois_mask1 = F.one_hot(
            rois_mask1, num_classes=len(fpn_masks)
        )
        rois_mask1 = rois_mask1.float().view(-1, 1, len(fpn_masks))
        rois_mask2 = F.one_hot(
            rois_mask2, num_classes=len(fpn_masks)
        )
        rois_mask2 = rois_mask2.float().view(-1, 1, len(fpn_masks))
        rois_pool_feats = []
        com_rois_feats = []

        for j in range(len(fpn_feats)):
            new_rois = rois / self.fpn_strides[j]
            extend_rois = new_rois.new_zeros(new_rois.size()[0], new_rois.size()[1], 5)
            extend_rois[:, :, 0] = new_rois[:, :, 0]
            extend_rois[:, :, 1] = new_rois[:, :, 1]
            # extend_rois[:, :, 2] = 0
            extend_rois[:, :, 3] = new_rois[:, :, 2]
            # extend_rois[:, :, 4] = 0
            roi_feat = (fpn_feats[j] * fpn_masks[j][:, None, :]).unsqueeze(2)
            left_extend_rois = extend_rois.new_zeros(new_rois.size()[0], new_rois.size()[1], 5)
            right_extend_rois = extend_rois.new_zeros(new_rois.size()[0], new_rois.size()[1], 5)
            left_extend_rois[:, :, 0] = extend_rois[:, :, 0]
            left_extend_rois[:, :, 1] = extend_rois[:, :, 1] - (extend_rois[:, :, 3] - extend_rois[:, :, 1]) * 0.5
            left_extend_rois[:, :, 3] = extend_rois[:, :, 1]
            left_extend_rois[:, :, 1] = left_extend_rois[:, :, 1].clamp(min=0.0, max=self.max_seq_len)
            right_extend_rois[:, :, 0] = extend_rois[:, :, 0]
            right_extend_rois[:, :, 1] = extend_rois[:, :, 3]
            right_extend_rois[:, :, 3] = extend_rois[:, :, 3] + (extend_rois[:, :, 3] - extend_rois[:, :, 1]) * 0.5
            right_extend_rois[:, :, 1] = right_extend_rois[:, :, 1].clamp(min=0.0, max=self.max_seq_len)
            extend_rois = extend_rois.view(-1, 5).cuda()
            left_extend_rois = left_extend_rois.view(-1, 5).cuda()
            right_extend_rois = right_extend_rois.view(-1, 5).cuda()
            sub_pool_feats = self.pool(roi_feat, extend_rois)
            left_pool_feats = self.pool(roi_feat, left_extend_rois)
            left_pool_feats = left_pool_feats.view(rois_b, rois_n, -1)
            right_pool_feats = self.pool(roi_feat, right_extend_rois)
            right_pool_feats = right_pool_feats.view(rois_b, rois_n, -1)
            sub_pool_feats = sub_pool_feats.view(rois_b, rois_n, -1)
            com_pool_feats = torch.cat((left_pool_feats, sub_pool_feats, right_pool_feats), dim=2)
            rois_pool_feats.append(sub_pool_feats)
            com_rois_feats.append(com_pool_feats)
        rois_pool_feats = torch.stack(rois_pool_feats, dim=2).view(rois_b * rois_n, len(fpn_feats), -1)
        com_rois_feats = torch.stack(com_rois_feats, dim=2).view(rois_b * rois_n, len(fpn_feats), -1)
        final_pool_feats_1 = torch.bmm(rois_mask, rois_pool_feats).squeeze(1)
        final_pool_feats_2 = torch.bmm(rois_mask1, rois_pool_feats).squeeze(1)
        final_pool_feats_3 = torch.bmm(rois_mask2, rois_pool_feats).squeeze(1)
        final_pool_feats=torch.stack((final_pool_feats_1,final_pool_feats_2,final_pool_feats_3),dim=1)
        final_pool_feats=final_pool_feats*self.pool_weights[None,:,None]
        final_pool_feats=torch.sum(final_pool_feats,dim=1)

        scorces_mask = (scorces > 0.5).float()
        final_com_feats_1 = torch.bmm(rois_mask, com_rois_feats).squeeze(1)
        final_com_feats_2 = torch.bmm(rois_mask1, com_rois_feats).squeeze(1)
        final_com_feats_3 = torch.bmm(rois_mask2, com_rois_feats).squeeze(1)
        final_com_feats = torch.stack((final_com_feats_1, final_com_feats_2, final_com_feats_3), dim=1)
        final_com_feats=final_com_feats*self.pool_weights[None,:,None]
        final_com_feats=torch.sum(final_com_feats,dim=1)
        # cls_log = self.classify(final_pool_feats).view(rois_b, rois_n, -1)

        final_pool_feats1 = final_pool_feats.view(rois_b, rois_n, -1)
        gt_pool_feats = final_pool_feats1[:, :gt_segments.size()[1], :]

        cls_log1, cls_gt, cls_node = self.classify1(final_pool_feats1, gcn_mask1, gcn_mask2, scorces_mask,
                                                    gt_pool_feats, rois)
        cls_log1 = self.relu(cls_log1)

        cls_log = torch.cat((final_pool_feats1, cls_log1), dim=2)
        cls_log = cls_log.view(rois_b * rois_n, -1)
        cls_log = self.classify(cls_log).view(rois_b, rois_n, -1)
        # com_log = self.compeletness(final_com_feats).view(rois_b, rois_n, -1)

        final_com_feats1 = final_com_feats.view(rois_b, rois_n, -1)
        gt_com_feats = final_com_feats1[:, :gt_segments.size()[1], :]

        com_log1, com_gt, com_node = self.compeletness1(final_com_feats1, gcn_mask1, gcn_mask2, scorces_mask,
                                                        gt_com_feats, rois)
        com_log1 = self.relu(com_log1)

        com_log = torch.cat((final_com_feats1, com_log1), dim=2)
        com_log = com_log.view(rois_b * rois_n, -1)
        com_log = self.compeletness(com_log).view(rois_b, rois_n, -1)

        # print(cls_log.size())

        # print(new_rois[0][0],2,fpn_feats[j].size()[2])
        return fpn_masks, out_cls_logits, out_offsets, rois.cuda(), scorces.cuda(), rois_mask, cls_log, com_log, cls_gt, cls_node, com_gt, com_node

        # return loss during training

    def _sample_nodes(self, rois):
        batch, num, _ = rois.size()
        new_rois = rois.new_zeros((rois.size()[0], rois.size()[1], 2))
        new_rois[:, :, 0] = rois[:, :, 1] - (rois[:, :, 2] - rois[:, :, 1]) * 0.5
        new_rois[:, :, 1] = rois[:, :, 2] + (rois[:, :, 2] - rois[:, :, 1]) * 0.5
        new_rois = new_rois.clamp(min=0.0, max=self.max_seq_len)
        # print(new_rois.size())
        left_rois = new_rois.unsqueeze(1).repeat(1, new_rois.size()[1], 1, 1)
        rigth_rois = new_rois.unsqueeze(2).repeat(1, 1, new_rois.size()[1], 1)
        min_left = torch.min(left_rois[:, :, :, 0], rigth_rois[:, :, :, 0])
        max_left = torch.max(left_rois[:, :, :, 0], rigth_rois[:, :, :, 0])
        min_right = torch.min(left_rois[:, :, :, 1], rigth_rois[:, :, :, 1])
        max_right = torch.max(left_rois[:, :, :, 1], rigth_rois[:, :, :, 1])
        ious = (min_right - max_left) / (max_right - min_left)
        ious_mask = (ious > 0).float()
        left_rois_center = (left_rois[:, :, :, 0] + left_rois[:, :, :, 1]) / 2.0
        rigth_rois_center = (rigth_rois[:, :, :, 0] + rigth_rois[:, :, :, 1]) / 2.0
        type_idxs = left_rois_center - rigth_rois_center
        ious_mask1 = ious_mask * (type_idxs >= 0).float()
        ious_mask2 = ious_mask * (type_idxs < 0).float()
        return ious_mask1, ious_mask2




class PtTransformer_gcn_loss_type_add_weight_diff(nn.Module):
    """
        Transformer based model for single stage action localization
    """

    def __init__(
            self,
            model_configs,
            train_cfg,
    ):
        super().__init__()
        # re-distribute params to backbone / neck / head
        self.fpn_strides = [model_configs["scale_factor"] ** i for i in range(model_configs["backbone_arch"][-1] + 1)]
        self.reg_range = model_configs["regression_range"]
        assert len(self.fpn_strides) == len(self.reg_range)
        self.scale_factor = model_configs["scale_factor"]
        # check the feature pyramid and local attention window size
        self.max_seq_len = model_configs["max_seq_len"]
        if isinstance(model_configs["n_mha_win_size"], int):
            self.mha_win_size = [model_configs["n_mha_win_size"]] * len(self.fpn_strides)
        else:
            assert len(model_configs["n_mha_win_size"]) == len(self.fpn_strides)
            self.mha_win_size = model_configs["n_mha_win_size"]
        max_div_factor = 1
        for l, (s, w) in enumerate(zip(self.fpn_strides, self.mha_win_size)):
            stride = s * (w // 2) * 2 if w > 1 else s
            assert model_configs[
                       "max_seq_len"] % stride == 0, "max_seq_len must be divisible by fpn stride and window size"
            if max_div_factor < stride:
                max_div_factor = stride
        self.max_div_factor = max_div_factor

        # training time config
        self.train_cls_prior_prob = train_cfg['cls_prior_prob']
        self.train_dropout = train_cfg['dropout']
        self.use_offset = False
        self.train_droppath = train_cfg['droppath']
        self.fpn_levels = model_configs["backbone_arch"][-1] + 1
        self.regression_range = model_configs["regression_range"]
        self.num_classes = model_configs["num_classes"]
        # we will need a better way to dispatch the params to backbones / necks
        # backbone network: conv + transformer
        assert model_configs["backbone_type"] in ['convTransformer', 'conv']
        if model_configs["backbone_type"] == 'convTransformer':
            self.backbone = ConvTransformerBackbone(
                n_in=model_configs["input_dim"],
                n_embd=model_configs["embd_dim"],
                n_head=model_configs["n_head"],
                n_embd_ks=model_configs["embd_kernel_size"],
                max_len=model_configs["max_seq_len"],
                arch=model_configs["backbone_arch"],
                mha_win_size=self.mha_win_size,
                scale_factor=model_configs["scale_factor"],
                with_ln=model_configs["embd_with_ln"],
                attn_pdrop=0.0,
                proj_pdrop=self.train_dropout,
                path_pdrop=self.train_droppath,
                use_abs_pe=model_configs["use_abs_pe"],
                use_rel_pe=model_configs["use_rel_pe"]
            )
        else:
            self.backbone = ConvBackbone(n_in=model_configs["input_dim"], n_embd=model_configs["embd_dim"],
                                         n_embd_ks=model_configs["embd_kernel_size"],
                                         arch=model_configs["backbone_arch"],
                                         scale_factor=model_configs["scale_factor"],
                                         with_ln=model_configs["embd_with_ln"])
        # fpn network: convs
        assert model_configs["fpn_type"] in ['fpn', 'identity']
        if model_configs["fpn_type"] == 'fpn':
            self.neck = FPN1D(
                in_channels=[model_configs["embd_dim"]] * (model_configs["backbone_arch"][-1] + 1),
                out_channel=model_configs["fpn_dim"],
                scale_factor=model_configs["scale_factor"],
                with_ln=model_configs["fpn_with_ln"]
            )
        else:
            self.neck = FPNIdentity(
                in_channels=[model_configs["embd_dim"]] * (model_configs["backbone_arch"][-1] + 1),
                out_channel=model_configs["fpn_dim"],
                scale_factor=model_configs["scale_factor"],
                with_ln=model_configs["fpn_with_ln"]
            )
        self.point_generator = PointGenerator(
            max_seq_len=model_configs["max_seq_len"] * model_configs["max_buffer_len_factor"],
            fpn_levels=len(self.fpn_strides),
            scale_factor=model_configs["scale_factor"],
            regression_range=self.reg_range)
        # classfication and regerssion heads
        self.cls_head = PtTransformerClsHead(
            model_configs["fpn_dim"], model_configs["head_dim"], 1,
            kernel_size=model_configs["head_kernel_size"],
            prior_prob=self.train_cls_prior_prob,
            with_ln=model_configs["head_with_ln"],
            empty_cls=train_cfg['head_empty_cls']
        )
        self.reg_head = PtTransformerRegHead(
            model_configs["fpn_dim"], model_configs["head_dim"], len(self.fpn_strides),
            kernel_size=model_configs["head_kernel_size"],
            with_ln=model_configs["head_with_ln"]
        )
        '''
        for p in self.parameters():
            p.requires_grad=False
        '''

        self.pool = RoIPool(1, 1)
        self.proposal_creator = ProposalCreator(parent_model=self, nms_thresh=train_cfg["iou_threshold"],
                                                n_train_pre_nms=train_cfg["pre_train_nms_topk"],
                                                n_train_post_nms=train_cfg["train_max_seg_num"],
                                                n_test_pre_nms=train_cfg["pre_test_nms_topk"],
                                                n_test_post_nms=train_cfg["test_max_seg_num"],
                                                pre_nms_thresh=train_cfg["pre_nms_thresh"],
                                                min_size=train_cfg["duration_thresh"], sigma=train_cfg["nms_sigma"],
                                                min_score=train_cfg["min_score"])
        self.type = 2
        self.childs_num = 8

        self.classify = nn.Linear(model_configs["fpn_dim"] * 2, self.num_classes + 1)
        self.classify1 = Graph_module_net_0_loss_type(model_configs["fpn_dim"], model_configs["fpn_dim"] * 2,
                                                      model_configs["fpn_dim"], self.type, self.childs_num, dropout=0.7)
        self.relu = nn.ReLU()
        '''
        self.compeletness = nn.Sequential(
            MLP(model_configs["fpn_dim"] * 3, model_configs["fpn_dim"] * 6, model_configs["fpn_dim"] * 3, dropout=0.7),
            nn.ReLU(),
            nn.Linear(model_configs["fpn_dim"] * 3, self.num_classes)
        )

        self.classify = nn.Sequential(
            MLP(model_configs["fpn_dim"], model_configs["fpn_dim"] * 2, model_configs["fpn_dim"] , dropout=0.7),
            nn.ReLU(),
            nn.Linear(model_configs["fpn_dim"] , self.num_classes+1)
        )
        self.relu = nn.ReLU()
        '''
        self.compeletness = nn.Linear(model_configs["fpn_dim"] * 6, self.num_classes)
        self.compeletness1 = Graph_module_net_0_loss_type(model_configs["fpn_dim"] * 3, model_configs["fpn_dim"] * 6,
                                                          model_configs["fpn_dim"] * 3, self.type, self.childs_num,
                                                          dropout=0.7)
        self.pool_weights=nn.Linear(model_configs["fpn_dim"] *2,1)


    @property
    def device(self):
        # a hacky way to get the device type
        # will throw an error if parameters are on different devices
        return list(set(p.device for p in self.parameters()))[0]

    def forward(self, inputs, training=True):
        # batch the video list into feats (B, C, T) and masks (B, 1, T)
        if training:
            batched_inputs, batched_masks, gt_segments, gt_masks = inputs[0], inputs[1], inputs[2], inputs[3]
        else:
            batched_inputs, batched_masks = inputs[0], inputs[1]
            gt_segments = torch.ones(batched_inputs.size()[0], 100, 2)
            gt_masks = torch.ones(batched_inputs.size()[0], 100)
        batch = batched_inputs.size()[0]
        batched_inputs = batched_inputs.cuda()
        batched_masks = batched_masks.unsqueeze(1).cuda()

        # forward the network (backbone -> neck -> heads)
        feats, masks = self.backbone(batched_inputs, batched_masks)
        fpn_feats, fpn_masks = self.neck(feats, masks)
        # out_cls: List[B, #cls + 1, T_i]
        out_cls_logits = self.cls_head(fpn_feats, fpn_masks)
        # out_offset: List[B, 2, T_i]
        out_offsets = self.reg_head(fpn_feats, fpn_masks)
        points = self.point_generator(fpn_feats)
        # permute the outputs
        rois = []
        scorces = []
        rois_mask = []
        for i in range(batch):
            # out_cls: F List[B, #cls, T_i] -> F List[B, T_i, #cls]
            out_cls_logits_i = [x.permute(0, 2, 1)[i, :, :] for x in out_cls_logits]
            # out_offset: F List[B, 2 (xC), T_i] -> F List[B, T_i, 2 (xC)]
            out_offsets_i = [x.permute(0, 2, 1)[i, :, :] for x in out_offsets]
            # fpn_masks: F list[B, 1, T_i] -> F List[B, T_i]
            fpn_masks_i = [x.squeeze(1)[i, :] for x in fpn_masks]
            gt_segment = gt_segments[i]
            gt_mask = gt_masks[i]
            mask_idx = gt_mask > 0
            gt_segment = gt_segment[mask_idx]
            with torch.no_grad():
                rois_i, score_i, roi_mask_i = self.proposal_creator(out_offsets=out_offsets_i,
                                                                    out_cls_logits=out_cls_logits_i,
                                                                    fpn_masks=fpn_masks_i, points=points,
                                                                    gt_rois=gt_segment)
            batch_index = i * torch.ones((rois_i.size()[0],)).unsqueeze(1)
            new_rois_i = torch.cat((batch_index, rois_i), dim=1)
            rois.append(new_rois_i)
            scorces.append(score_i)
            rois_mask.append(roi_mask_i)
        rois = torch.stack(rois, dim=0)
        rois_b, rois_n, _ = rois.size()
        gcn_mask1, gcn_mask2 = self._sample_nodes(rois)
        scorces = torch.stack(scorces, dim=0)
        rois_mask = torch.stack(rois_mask, dim=0)
        out_cls_logits = [x.permute(0, 2, 1) for x in out_cls_logits]
        # out_offset: F List[B, 2 (xC), T_i] -> F List[B, T_i, 2 (xC)]
        out_offsets = [x.permute(0, 2, 1) for x in out_offsets]
        # fpn_masks: F list[B, 1, T_i] -> F List[B, T_i]
        fpn_masks = [x.squeeze(1) for x in fpn_masks]
        rois_mask = rois_mask.long()
        rois_mask = F.one_hot(
            rois_mask, num_classes=len(fpn_masks)
        )
        rois_mask = rois_mask.float().view(-1, 1, len(fpn_masks))

        rois_pool_feats = []
        com_rois_feats = []

        for j in range(len(fpn_feats)):
            new_rois = rois / self.fpn_strides[j]
            extend_rois = new_rois.new_zeros(new_rois.size()[0], new_rois.size()[1], 5)
            extend_rois[:, :, 0] = new_rois[:, :, 0]
            extend_rois[:, :, 1] = new_rois[:, :, 1]
            # extend_rois[:, :, 2] = 0
            extend_rois[:, :, 3] = new_rois[:, :, 2]
            # extend_rois[:, :, 4] = 0
            roi_feat = (fpn_feats[j] * fpn_masks[j][:, None, :]).unsqueeze(2)
            left_extend_rois = extend_rois.new_zeros(new_rois.size()[0], new_rois.size()[1], 5)
            right_extend_rois = extend_rois.new_zeros(new_rois.size()[0], new_rois.size()[1], 5)
            left_extend_rois[:, :, 0] = extend_rois[:, :, 0]
            left_extend_rois[:, :, 1] = extend_rois[:, :, 1] - (extend_rois[:, :, 3] - extend_rois[:, :, 1]) * 0.5
            left_extend_rois[:, :, 3] = extend_rois[:, :, 1]
            left_extend_rois[:, :, 1] = left_extend_rois[:, :, 1].clamp(min=0.0, max=self.max_seq_len)
            right_extend_rois[:, :, 0] = extend_rois[:, :, 0]
            right_extend_rois[:, :, 1] = extend_rois[:, :, 3]
            right_extend_rois[:, :, 3] = extend_rois[:, :, 3] + (extend_rois[:, :, 3] - extend_rois[:, :, 1]) * 0.5
            right_extend_rois[:, :, 1] = right_extend_rois[:, :, 1].clamp(min=0.0, max=self.max_seq_len)
            extend_rois = extend_rois.view(-1, 5).cuda()
            left_extend_rois = left_extend_rois.view(-1, 5).cuda()
            right_extend_rois = right_extend_rois.view(-1, 5).cuda()
            sub_pool_feats = self.pool(roi_feat, extend_rois)
            left_pool_feats = self.pool(roi_feat, left_extend_rois)
            left_pool_feats = left_pool_feats.view(rois_b, rois_n, -1)
            right_pool_feats = self.pool(roi_feat, right_extend_rois)
            right_pool_feats = right_pool_feats.view(rois_b, rois_n, -1)
            sub_pool_feats = sub_pool_feats.view(rois_b, rois_n, -1)
            com_pool_feats = torch.cat((left_pool_feats, sub_pool_feats, right_pool_feats), dim=2)
            rois_pool_feats.append(sub_pool_feats)
            com_rois_feats.append(com_pool_feats)
        rois_pool_feats = torch.stack(rois_pool_feats, dim=2).view(rois_b * rois_n, len(fpn_feats), -1)
        com_rois_feats = torch.stack(com_rois_feats, dim=2).view(rois_b * rois_n, len(fpn_feats), -1)
        final_pool_feats_1 = torch.bmm(rois_mask, rois_pool_feats).squeeze(1)
        final_pool_feats_1=final_pool_feats_1.unsqueeze(1).repeat(1,len(fpn_feats),1)
        attention_feat=torch.cat((final_pool_feats_1,rois_pool_feats),dim=2).view(rois_b * rois_n* len(fpn_feats), -1)
        pool_weights=self.pool_weights(attention_feat).view(rois_b * rois_n, len(fpn_feats))
        pool_weights=pool_weights.softmax(dim=1)
        final_pool_feats=rois_pool_feats*pool_weights[:,:,None]
        final_pool_feats=torch.sum(final_pool_feats,dim=1)

        scorces_mask = (scorces > 0.5).float()
        final_com_feats=com_rois_feats*pool_weights[:,:,None]
        final_com_feats=torch.sum(final_com_feats,dim=1)
        # cls_log = self.classify(final_pool_feats).view(rois_b, rois_n, -1)

        final_pool_feats1 = final_pool_feats.view(rois_b, rois_n, -1)
        gt_pool_feats = final_pool_feats1[:, :gt_segments.size()[1], :]

        cls_log1, cls_gt, cls_node = self.classify1(final_pool_feats1, gcn_mask1, gcn_mask2, scorces_mask,
                                                    gt_pool_feats, rois)
        cls_log1 = self.relu(cls_log1)

        cls_log = torch.cat((final_pool_feats1, cls_log1), dim=2)
        cls_log = cls_log.view(rois_b * rois_n, -1)
        cls_log = self.classify(cls_log).view(rois_b, rois_n, -1)
        # com_log = self.compeletness(final_com_feats).view(rois_b, rois_n, -1)

        final_com_feats1 = final_com_feats.view(rois_b, rois_n, -1)
        gt_com_feats = final_com_feats1[:, :gt_segments.size()[1], :]

        com_log1, com_gt, com_node = self.compeletness1(final_com_feats1, gcn_mask1, gcn_mask2, scorces_mask,
                                                        gt_com_feats, rois)
        com_log1 = self.relu(com_log1)

        com_log = torch.cat((final_com_feats1, com_log1), dim=2)
        com_log = com_log.view(rois_b * rois_n, -1)
        com_log = self.compeletness(com_log).view(rois_b, rois_n, -1)

        # print(cls_log.size())

        # print(new_rois[0][0],2,fpn_feats[j].size()[2])
        return fpn_masks, out_cls_logits, out_offsets, rois.cuda(), scorces.cuda(), rois_mask, cls_log, com_log, cls_gt, cls_node, com_gt, com_node

        # return loss during training

    def _sample_nodes(self, rois):
        batch, num, _ = rois.size()
        new_rois = rois.new_zeros((rois.size()[0], rois.size()[1], 2))
        new_rois[:, :, 0] = rois[:, :, 1] - (rois[:, :, 2] - rois[:, :, 1]) * 0.5
        new_rois[:, :, 1] = rois[:, :, 2] + (rois[:, :, 2] - rois[:, :, 1]) * 0.5
        new_rois = new_rois.clamp(min=0.0, max=self.max_seq_len)
        # print(new_rois.size())
        left_rois = new_rois.unsqueeze(1).repeat(1, new_rois.size()[1], 1, 1)
        rigth_rois = new_rois.unsqueeze(2).repeat(1, 1, new_rois.size()[1], 1)
        min_left = torch.min(left_rois[:, :, :, 0], rigth_rois[:, :, :, 0])
        max_left = torch.max(left_rois[:, :, :, 0], rigth_rois[:, :, :, 0])
        min_right = torch.min(left_rois[:, :, :, 1], rigth_rois[:, :, :, 1])
        max_right = torch.max(left_rois[:, :, :, 1], rigth_rois[:, :, :, 1])
        ious = (min_right - max_left) / (max_right - min_left)
        ious_mask = (ious > 0).float()
        left_rois_center = (left_rois[:, :, :, 0] + left_rois[:, :, :, 1]) / 2.0
        rigth_rois_center = (rigth_rois[:, :, :, 0] + rigth_rois[:, :, :, 1]) / 2.0
        type_idxs = left_rois_center - rigth_rois_center
        ious_mask1 = ious_mask * (type_idxs >= 0).float()
        ious_mask2 = ious_mask * (type_idxs < 0).float()
        return ious_mask1, ious_mask2




class PtTransformer_gcn_loss_type_add_weight_diff_com(nn.Module):
    """
        Transformer based model for single stage action localization
    """

    def __init__(
            self,
            model_configs,
            train_cfg,
    ):
        super().__init__()
        # re-distribute params to backbone / neck / head
        self.fpn_strides = [model_configs["scale_factor"] ** i for i in range(model_configs["backbone_arch"][-1] + 1)]
        self.reg_range = model_configs["regression_range"]
        assert len(self.fpn_strides) == len(self.reg_range)
        self.scale_factor = model_configs["scale_factor"]
        # check the feature pyramid and local attention window size
        self.max_seq_len = model_configs["max_seq_len"]
        if isinstance(model_configs["n_mha_win_size"], int):
            self.mha_win_size = [model_configs["n_mha_win_size"]] * len(self.fpn_strides)
        else:
            assert len(model_configs["n_mha_win_size"]) == len(self.fpn_strides)
            self.mha_win_size = model_configs["n_mha_win_size"]
        max_div_factor = 1
        for l, (s, w) in enumerate(zip(self.fpn_strides, self.mha_win_size)):
            stride = s * (w // 2) * 2 if w > 1 else s
            assert model_configs[
                       "max_seq_len"] % stride == 0, "max_seq_len must be divisible by fpn stride and window size"
            if max_div_factor < stride:
                max_div_factor = stride
        self.max_div_factor = max_div_factor

        # training time config
        self.train_cls_prior_prob = train_cfg['cls_prior_prob']
        self.train_dropout = train_cfg['dropout']
        self.use_offset = False
        self.train_droppath = train_cfg['droppath']
        self.fpn_levels = model_configs["backbone_arch"][-1] + 1
        self.regression_range = model_configs["regression_range"]
        self.num_classes = model_configs["num_classes"]
        # we will need a better way to dispatch the params to backbones / necks
        # backbone network: conv + transformer
        assert model_configs["backbone_type"] in ['convTransformer', 'conv']
        if model_configs["backbone_type"] == 'convTransformer':
            self.backbone = ConvTransformerBackbone(
                n_in=model_configs["input_dim"],
                n_embd=model_configs["embd_dim"],
                n_head=model_configs["n_head"],
                n_embd_ks=model_configs["embd_kernel_size"],
                max_len=model_configs["max_seq_len"],
                arch=model_configs["backbone_arch"],
                mha_win_size=self.mha_win_size,
                scale_factor=model_configs["scale_factor"],
                with_ln=model_configs["embd_with_ln"],
                attn_pdrop=0.0,
                proj_pdrop=self.train_dropout,
                path_pdrop=self.train_droppath,
                use_abs_pe=model_configs["use_abs_pe"],
                use_rel_pe=model_configs["use_rel_pe"]
            )
        else:
            self.backbone = ConvBackbone(n_in=model_configs["input_dim"], n_embd=model_configs["embd_dim"],
                                         n_embd_ks=model_configs["embd_kernel_size"],
                                         arch=model_configs["backbone_arch"],
                                         scale_factor=model_configs["scale_factor"],
                                         with_ln=model_configs["embd_with_ln"])
        # fpn network: convs
        assert model_configs["fpn_type"] in ['fpn', 'identity']
        if model_configs["fpn_type"] == 'fpn':
            self.neck = FPN1D(
                in_channels=[model_configs["embd_dim"]] * (model_configs["backbone_arch"][-1] + 1),
                out_channel=model_configs["fpn_dim"],
                scale_factor=model_configs["scale_factor"],
                with_ln=model_configs["fpn_with_ln"]
            )
        else:
            self.neck = FPNIdentity(
                in_channels=[model_configs["embd_dim"]] * (model_configs["backbone_arch"][-1] + 1),
                out_channel=model_configs["fpn_dim"],
                scale_factor=model_configs["scale_factor"],
                with_ln=model_configs["fpn_with_ln"]
            )
        self.point_generator = PointGenerator(
            max_seq_len=model_configs["max_seq_len"] * model_configs["max_buffer_len_factor"],
            fpn_levels=len(self.fpn_strides),
            scale_factor=model_configs["scale_factor"],
            regression_range=self.reg_range)
        # classfication and regerssion heads
        self.cls_head = PtTransformerClsHead(
            model_configs["fpn_dim"], model_configs["head_dim"], 1,
            kernel_size=model_configs["head_kernel_size"],
            prior_prob=self.train_cls_prior_prob,
            with_ln=model_configs["head_with_ln"],
            empty_cls=train_cfg['head_empty_cls']
        )
        self.reg_head = PtTransformerRegHead(
            model_configs["fpn_dim"], model_configs["head_dim"], len(self.fpn_strides),
            kernel_size=model_configs["head_kernel_size"],
            with_ln=model_configs["head_with_ln"]
        )
        '''
        for p in self.parameters():
            p.requires_grad=False
        '''

        self.pool = RoIPool(1, 1)
        self.proposal_creator = ProposalCreator(parent_model=self, nms_thresh=train_cfg["iou_threshold"],
                                                n_train_pre_nms=train_cfg["pre_train_nms_topk"],
                                                n_train_post_nms=train_cfg["train_max_seg_num"],
                                                n_test_pre_nms=train_cfg["pre_test_nms_topk"],
                                                n_test_post_nms=train_cfg["test_max_seg_num"],
                                                pre_nms_thresh=train_cfg["pre_nms_thresh"],
                                                min_size=train_cfg["duration_thresh"], sigma=train_cfg["nms_sigma"],
                                                min_score=train_cfg["min_score"])
        self.type = 2
        self.childs_num = 4

        self.classify = nn.Linear(model_configs["fpn_dim"] * 2, self.num_classes + 1)
        self.classify1 = Graph_module_net_0_loss_type(model_configs["fpn_dim"], model_configs["fpn_dim"] * 2,
                                                      model_configs["fpn_dim"], self.type, self.childs_num, dropout=0.7)
        self.relu = nn.ReLU()
        '''
        self.compeletness = nn.Sequential(
            MLP(model_configs["fpn_dim"] * 3, model_configs["fpn_dim"] * 6, model_configs["fpn_dim"] * 3, dropout=0.7),
            nn.ReLU(),
            nn.Linear(model_configs["fpn_dim"] * 3, self.num_classes)
        )

        self.classify = nn.Sequential(
            MLP(model_configs["fpn_dim"], model_configs["fpn_dim"] * 2, model_configs["fpn_dim"] , dropout=0.7),
            nn.ReLU(),
            nn.Linear(model_configs["fpn_dim"] , self.num_classes+1)
        )
        self.relu = nn.ReLU()
        '''
        self.compeletness = nn.Linear(model_configs["fpn_dim"] * 6, self.num_classes)
        self.compeletness1 = Graph_module_net_0_loss_type(model_configs["fpn_dim"] * 3, model_configs["fpn_dim"] * 6,
                                                          model_configs["fpn_dim"] * 3, self.type, self.childs_num,
                                                          dropout=0.7)
        self.atten_pool_weight=nn.Linear(model_configs["fpn_dim"] *2,1)
        self.atten_pool_weight_com = nn.Linear(model_configs["fpn_dim"] * 6, 1)


    @property
    def device(self):
        # a hacky way to get the device type
        # will throw an error if parameters are on different devices
        return list(set(p.device for p in self.parameters()))[0]

    def forward(self, inputs, training=True):
        # batch the video list into feats (B, C, T) and masks (B, 1, T)
        if training:
            batched_inputs, batched_masks, gt_segments, gt_masks = inputs[0], inputs[1], inputs[2], inputs[3]
        else:
            batched_inputs, batched_masks = inputs[0], inputs[1]
            gt_segments = torch.ones(batched_inputs.size()[0], 100, 2)
            gt_masks = torch.ones(batched_inputs.size()[0], 100)
        batch = batched_inputs.size()[0]
        batched_inputs = batched_inputs.cuda()
        batched_masks = batched_masks.unsqueeze(1).cuda()

        # forward the network (backbone -> neck -> heads)
        feats, masks = self.backbone(batched_inputs, batched_masks)
        fpn_feats, fpn_masks = self.neck(feats, masks)
        # out_cls: List[B, #cls + 1, T_i]
        out_cls_logits = self.cls_head(fpn_feats, fpn_masks)
        # out_offset: List[B, 2, T_i]
        out_offsets = self.reg_head(fpn_feats, fpn_masks)
        points = self.point_generator(fpn_feats)
        # permute the outputs
        rois = []
        scorces = []
        rois_mask = []
        for i in range(batch):
            # out_cls: F List[B, #cls, T_i] -> F List[B, T_i, #cls]
            out_cls_logits_i = [x.permute(0, 2, 1)[i, :, :] for x in out_cls_logits]
            # out_offset: F List[B, 2 (xC), T_i] -> F List[B, T_i, 2 (xC)]
            out_offsets_i = [x.permute(0, 2, 1)[i, :, :] for x in out_offsets]
            # fpn_masks: F list[B, 1, T_i] -> F List[B, T_i]
            fpn_masks_i = [x.squeeze(1)[i, :] for x in fpn_masks]
            gt_segment = gt_segments[i]
            gt_mask = gt_masks[i]
            mask_idx = gt_mask > 0
            gt_segment = gt_segment[mask_idx]
            with torch.no_grad():
                rois_i, score_i, roi_mask_i = self.proposal_creator(out_offsets=out_offsets_i,
                                                                    out_cls_logits=out_cls_logits_i,
                                                                    fpn_masks=fpn_masks_i, points=points,
                                                                    gt_rois=gt_segment)
            batch_index = i * torch.ones((rois_i.size()[0],)).unsqueeze(1)
            new_rois_i = torch.cat((batch_index, rois_i), dim=1)
            rois.append(new_rois_i)
            scorces.append(score_i)
            rois_mask.append(roi_mask_i)
        rois = torch.stack(rois, dim=0)
        rois_b, rois_n, _ = rois.size()
        gcn_mask1, gcn_mask2 = self._sample_nodes(rois)
        scorces = torch.stack(scorces, dim=0)
        rois_mask = torch.stack(rois_mask, dim=0)
        out_cls_logits = [x.permute(0, 2, 1) for x in out_cls_logits]
        # out_offset: F List[B, 2 (xC), T_i] -> F List[B, T_i, 2 (xC)]
        out_offsets = [x.permute(0, 2, 1) for x in out_offsets]
        # fpn_masks: F list[B, 1, T_i] -> F List[B, T_i]
        fpn_masks = [x.squeeze(1) for x in fpn_masks]
        rois_mask = rois_mask.long()
        rois_mask = F.one_hot(
            rois_mask, num_classes=len(fpn_masks)
        )
        rois_mask = rois_mask.float().view(-1, 1, len(fpn_masks))
        rois_pool_feats = []
        com_rois_feats = []

        for j in range(len(fpn_feats)):
            new_rois = rois / self.fpn_strides[j]
            extend_rois = new_rois.new_zeros(new_rois.size()[0], new_rois.size()[1], 5)
            extend_rois[:, :, 0] = new_rois[:, :, 0]
            extend_rois[:, :, 1] = new_rois[:, :, 1]
            # extend_rois[:, :, 2] = 0
            extend_rois[:, :, 3] = new_rois[:, :, 2]
            # extend_rois[:, :, 4] = 0
            roi_feat = (fpn_feats[j] * fpn_masks[j][:, None, :]).unsqueeze(2)
            left_extend_rois = extend_rois.new_zeros(new_rois.size()[0], new_rois.size()[1], 5)
            right_extend_rois = extend_rois.new_zeros(new_rois.size()[0], new_rois.size()[1], 5)
            left_extend_rois[:, :, 0] = extend_rois[:, :, 0]
            left_extend_rois[:, :, 1] = extend_rois[:, :, 1] - (extend_rois[:, :, 3] - extend_rois[:, :, 1]) * 0.5
            left_extend_rois[:, :, 3] = extend_rois[:, :, 1]
            left_extend_rois[:, :, 1] = left_extend_rois[:, :, 1].clamp(min=0.0, max=self.max_seq_len)
            right_extend_rois[:, :, 0] = extend_rois[:, :, 0]
            right_extend_rois[:, :, 1] = extend_rois[:, :, 3]
            right_extend_rois[:, :, 3] = extend_rois[:, :, 3] + (extend_rois[:, :, 3] - extend_rois[:, :, 1]) * 0.5
            right_extend_rois[:, :, 1] = right_extend_rois[:, :, 1].clamp(min=0.0, max=self.max_seq_len)
            extend_rois = extend_rois.view(-1, 5).cuda()
            left_extend_rois = left_extend_rois.view(-1, 5).cuda()
            right_extend_rois = right_extend_rois.view(-1, 5).cuda()
            sub_pool_feats = self.pool(roi_feat, extend_rois)
            left_pool_feats = self.pool(roi_feat, left_extend_rois)
            left_pool_feats = left_pool_feats.view(rois_b, rois_n, -1)
            right_pool_feats = self.pool(roi_feat, right_extend_rois)
            right_pool_feats = right_pool_feats.view(rois_b, rois_n, -1)
            sub_pool_feats = sub_pool_feats.view(rois_b, rois_n, -1)
            com_pool_feats = torch.cat((left_pool_feats, sub_pool_feats, right_pool_feats), dim=2)
            rois_pool_feats.append(sub_pool_feats)
            com_rois_feats.append(com_pool_feats)
        rois_pool_feats = torch.stack(rois_pool_feats, dim=2).view(rois_b * rois_n, len(fpn_feats), -1)
        com_rois_feats = torch.stack(com_rois_feats, dim=2).view(rois_b * rois_n, len(fpn_feats), -1)
        final_pool_feats_1 = torch.bmm(rois_mask, rois_pool_feats).squeeze(1)
        final_pool_feats_1=final_pool_feats_1.unsqueeze(1).repeat(1,len(fpn_feats),1)
        attention_feat=torch.cat((final_pool_feats_1,rois_pool_feats),dim=2).view(rois_b * rois_n* len(fpn_feats), -1)
        pool_weights=self.atten_pool_weight(attention_feat).view(rois_b * rois_n, len(fpn_feats))
        final_pool_feats=rois_pool_feats*pool_weights[:,:,None]
        final_pool_feats=torch.sum(final_pool_feats,dim=1)

        scorces_mask = (scorces > 0.5).float()
        final_com_feats_1 = torch.bmm(rois_mask, com_rois_feats).squeeze(1)
        final_com_feats_1 = final_com_feats_1.unsqueeze(1).repeat(1, len(fpn_feats), 1)
        attention_feat_com = torch.cat((final_com_feats_1, com_rois_feats), dim=2).view(rois_b * rois_n * len(fpn_feats),
                                                                                      -1)
        pool_weights_com = self.atten_pool_weight_com(attention_feat_com).view(rois_b * rois_n, len(fpn_feats))
        final_com_feats=com_rois_feats*pool_weights_com[:,:,None]
        final_com_feats=torch.sum(final_com_feats,dim=1)
        # cls_log = self.classify(final_pool_feats).view(rois_b, rois_n, -1)

        final_pool_feats1 = final_pool_feats.view(rois_b, rois_n, -1)
        gt_pool_feats = final_pool_feats1[:, :gt_segments.size()[1], :]

        cls_log1, cls_gt, cls_node = self.classify1(final_pool_feats1, gcn_mask1, gcn_mask2, scorces_mask,
                                                    gt_pool_feats, rois)
        cls_log1 = self.relu(cls_log1)

        cls_log = torch.cat((final_pool_feats1, cls_log1), dim=2)
        cls_log = cls_log.view(rois_b * rois_n, -1)
        cls_log = self.classify(cls_log).view(rois_b, rois_n, -1)
        # com_log = self.compeletness(final_com_feats).view(rois_b, rois_n, -1)

        final_com_feats1 = final_com_feats.view(rois_b, rois_n, -1)
        gt_com_feats = final_com_feats1[:, :gt_segments.size()[1], :]

        com_log1, com_gt, com_node = self.compeletness1(final_com_feats1, gcn_mask1, gcn_mask2, scorces_mask,
                                                        gt_com_feats, rois)
        com_log1 = self.relu(com_log1)

        com_log = torch.cat((final_com_feats1, com_log1), dim=2)
        com_log = com_log.view(rois_b * rois_n, -1)
        com_log = self.compeletness(com_log).view(rois_b, rois_n, -1)

        # print(cls_log.size())

        # print(new_rois[0][0],2,fpn_feats[j].size()[2])
        return fpn_masks, out_cls_logits, out_offsets, rois.cuda(), scorces.cuda(), rois_mask, cls_log, com_log, cls_gt, cls_node, com_gt, com_node

        # return loss during training

    def _sample_nodes(self, rois):
        batch, num, _ = rois.size()
        new_rois = rois.new_zeros((rois.size()[0], rois.size()[1], 2))
        new_rois[:, :, 0] = rois[:, :, 1] - (rois[:, :, 2] - rois[:, :, 1]) * 0.5
        new_rois[:, :, 1] = rois[:, :, 2] + (rois[:, :, 2] - rois[:, :, 1]) * 0.5
        new_rois = new_rois.clamp(min=0.0, max=self.max_seq_len)
        # print(new_rois.size())
        left_rois = new_rois.unsqueeze(1).repeat(1, new_rois.size()[1], 1, 1)
        rigth_rois = new_rois.unsqueeze(2).repeat(1, 1, new_rois.size()[1], 1)
        min_left = torch.min(left_rois[:, :, :, 0], rigth_rois[:, :, :, 0])
        max_left = torch.max(left_rois[:, :, :, 0], rigth_rois[:, :, :, 0])
        min_right = torch.min(left_rois[:, :, :, 1], rigth_rois[:, :, :, 1])
        max_right = torch.max(left_rois[:, :, :, 1], rigth_rois[:, :, :, 1])
        ious = (min_right - max_left) / (max_right - min_left)
        ious_mask = (ious > 0).float()
        left_rois_center = (left_rois[:, :, :, 0] + left_rois[:, :, :, 1]) / 2.0
        rigth_rois_center = (rigth_rois[:, :, :, 0] + rigth_rois[:, :, :, 1]) / 2.0
        type_idxs = left_rois_center - rigth_rois_center
        ious_mask1 = ious_mask * (type_idxs >= 0).float()
        ious_mask2 = ious_mask * (type_idxs < 0).float()
        return ious_mask1, ious_mask2