import os
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
class Attention_4(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """
    def forward(self, query, key, mask=None, dropout=None,softmax=True):
        scores = F.cosine_similarity(query, key,dim=3)

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, scores.size()[1], 1)
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.relu(scores)
        return p_attn



class MultiHeadedAttention_3(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1,flag=1,softmax=True):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.h = h
        self.soft_max=softmax
        self.sptatial_flag=flag
        self.attention = Attention_4()
        if self.sptatial_flag == 0:
            self.len_d_model = d_model * 2
        elif self.sptatial_flag == 1:
            self.len_d_model = d_model * 2 + 4
        elif self.sptatial_flag == 2:
            self.len_d_model = d_model * 2 + 6
        elif self.sptatial_flag == 3:
            self.len_d_model = d_model * 2 + 5
        elif self.sptatial_flag == 4:
            self.len_d_model = d_model * 2 + 2
        elif self.sptatial_flag == 5:
            self.len_d_model = d_model * 2 + 3
        self.linear_layer = nn.Linear(self.len_d_model, self.h)
    def get_sptatial_feature(self,s_query,s_key):
        if self.sptatial_flag == 1:
            out = torch.cat((s_query, s_key), dim=3)
        elif self.sptatial_flag == 2:
            s_query_len = s_query[:, :, :,1] - s_query[:, :,:, 0]
            s_query_len = s_query_len.unsqueeze(3)
            s_key_len = s_key[:,:, :, 1] - s_key[:,:, :, 0]
            s_key_len = s_key_len.unsqueeze(3)
            out = torch.cat((s_query, s_query_len, s_key, s_key_len), dim=3)

        elif self.sptatial_flag == 3:
            s_query_len = s_query[:, :,:, 1] - s_query[:, :, :,0]
            s_key_len = s_key[:, :,:, 1] - s_key[:, :,:, 0]
            s_1 = (s_query - s_key).sum(dim=3) / 2 / s_key_len
            s_1 = s_1.unsqueeze(3)
            s_2 = s_query_len / s_key_len
            s_2 = s_2.unsqueeze(3)
            union1 = torch.min(s_query[:, :,:, 0], s_key[:, :,:, 0])
            union2 = torch.max(s_query[:, :,:, 1], s_key[:, :,:, 1])
            inter1 = torch.max(s_query[:, :,:, 0], s_key[:, :, :,0])
            inter2 = torch.min(s_query[:, :,:, 1], s_key[:, :,:, 1])
            s_3 = F.relu((inter2 - inter1) / (union2 - union1))
            s_3 = s_3.unsqueeze(3)
            s_query_len = s_query_len.unsqueeze(3)
            s_key_len = s_key_len.unsqueeze(3)
            out = torch.cat((s_1, s_2, s_3, s_query_len, s_key_len), dim=3)

        elif self.sptatial_flag == 4:
            s_query_len = s_query[:, :,:, 1] - s_query[:, :,:, 0]
            s_key_len = s_key[:, :,:, 1] - s_key[:, :,:, 0]
            s_1 = (s_query - s_key).sum(dim=3) / 2 / s_key_len
            s_2 = s_query_len / s_key_len
            s_2 = torch.exp(s_2)
            s_1 = s_1.unsqueeze(3)
            s_2 = s_2.unsqueeze(3)
            out = torch.cat((s_1, s_2), dim=3)

        elif self.sptatial_flag == 5:
            s_query_len = s_query[:, :,:, 1] - s_query[:, :,:, 0]
            s_key_len = s_key[:, :, :,1] - s_key[:, :,:, 0]
            s_1 = (s_query - s_key).sum(dim=3) / 2 / s_key_len
            s_2 = s_query_len / s_key_len
            s_2 = torch.exp(s_2)
            s_3 = (s_key - s_query).sum(dim=3) / 2 / s_query_len
            s_1 = s_1.unsqueeze(3)
            s_2 = s_2.unsqueeze(3)
            s_3 = s_3.unsqueeze(3)
            out = torch.cat((s_1, s_2, s_3), dim=3)
        return out

    def forward(self, query, key,s_query,s_key, roi_mask,node_num):
        ft_num = query.size()[-2]
        batch_size, num, _, chinnal = query.size()
        spat_feature=self.get_sptatial_feature(s_query,s_key)
        ft=torch.cat((query,key,spat_feature),dim=3)
        ft=ft.view(-1,self.len_d_model)
        # 2) Apply attention on all the projected vectors in batch.
        attn1 = self.linear_layer(ft)
        attn1 = attn1.view(-1, ft_num, ft_num, self.h)
        attn1 = torch.sigmoid(attn1)
        attn = self.attention(query, key, mask=None, dropout=None, softmax=self.soft_max)
        attn = attn.view(batch_size, -1, num, num).contiguous()
        attn = attn.permute(0, 2, 3, 1).contiguous()
        # one_mm=torch.eye(num).cuda()
        atten_mask = torch.zeros(attn.size()).cuda()
        values, idces = torch.topk(attn, node_num, dim=2)
        atten_mask[:, :, idces, :] = 1.0
        attn1 = attn1 * roi_mask[:, :, :, None].cuda() * atten_mask.cuda()
        return attn1


class MultiHeadedAttention_3_min(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1, flag=1, softmax=True):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.h = h
        self.soft_max = softmax
        self.sptatial_flag = flag
        if self.sptatial_flag == 0:
            self.len_d_model = d_model * 2
        elif self.sptatial_flag == 1:
            self.len_d_model = d_model * 2 + 4
        elif self.sptatial_flag == 2:
            self.len_d_model = d_model * 2 + 6
        elif self.sptatial_flag == 3:
            self.len_d_model = d_model * 2 + 5
        elif self.sptatial_flag == 4:
            self.len_d_model = d_model * 2 + 2
        elif self.sptatial_flag == 5:
            self.len_d_model = d_model * 2 + 3
        self.linear_layer = nn.Linear(self.len_d_model, self.h)
        self.attention = Attention_4()

    def get_sptatial_feature(self, s_query, s_key):
        if self.sptatial_flag == 1:
            out = torch.cat((s_query, s_key), dim=3)
        elif self.sptatial_flag == 2:
            s_query_len = s_query[:, :, :, 1] - s_query[:, :, :, 0]
            s_query_len = s_query_len.unsqueeze(3)
            s_key_len = s_key[:, :, :, 1] - s_key[:, :, :, 0]
            s_key_len = s_key_len.unsqueeze(3)
            out = torch.cat((s_query, s_query_len, s_key, s_key_len), dim=3)

        elif self.sptatial_flag == 3:
            s_query_len = s_query[:, :, :, 1] - s_query[:, :, :, 0]
            s_key_len = s_key[:, :, :, 1] - s_key[:, :, :, 0]
            s_1 = (s_query - s_key).sum(dim=3) / 2 / s_key_len
            s_1 = s_1.unsqueeze(3)
            s_2 = s_query_len / s_key_len
            s_2 = s_2.unsqueeze(3)
            union1 = torch.min(s_query[:, :, :, 0], s_key[:, :, :, 0])
            union2 = torch.max(s_query[:, :, :, 1], s_key[:, :, :, 1])
            inter1 = torch.max(s_query[:, :, :, 0], s_key[:, :, :, 0])
            inter2 = torch.min(s_query[:, :, :, 1], s_key[:, :, :, 1])
            s_3 = F.relu((inter2 - inter1) / (union2 - union1))
            s_3 = s_3.unsqueeze(3)
            s_query_len = s_query_len.unsqueeze(3)
            s_key_len = s_key_len.unsqueeze(3)
            out = torch.cat((s_1, s_2, s_3, s_query_len, s_key_len), dim=3)

        elif self.sptatial_flag == 4:
            s_query_len = s_query[:, :, :, 1] - s_query[:, :, :, 0]
            s_key_len = s_key[:, :, :, 1] - s_key[:, :, :, 0]
            s_1 = (s_query - s_key).sum(dim=3) / 2 / s_key_len
            s_2 = s_query_len / s_key_len
            s_2 = torch.exp(s_2)
            s_1 = s_1.unsqueeze(3)
            s_2 = s_2.unsqueeze(3)
            out = torch.cat((s_1, s_2), dim=3)

        elif self.sptatial_flag == 5:
            s_query_len = s_query[:, :, :, 1] - s_query[:, :, :, 0]
            s_key_len = s_key[:, :, :, 1] - s_key[:, :, :, 0]
            s_1 = (s_query - s_key).sum(dim=3) / 2 / s_key_len
            s_2 = s_query_len / s_key_len
            s_2 = torch.exp(s_2)
            s_3 = (s_key - s_query).sum(dim=3) / 2 / s_query_len
            s_1 = s_1.unsqueeze(3)
            s_2 = s_2.unsqueeze(3)
            s_3 = s_3.unsqueeze(3)
            out = torch.cat((s_1, s_2, s_3), dim=3)
        return out

    def forward(self, query, key, s_query, s_key, roi_mask, node_num):
        ft_num = query.size()[-2]
        batch_size, num, _, chinnal = query.size()
        spat_feature = self.get_sptatial_feature(s_query, s_key)
        ft = torch.cat((query, key, spat_feature), dim=3)
        ft = ft.view(-1, self.len_d_model)
        # 2) Apply attention on all the projected vectors in batch.
        attn1 = self.linear_layer(ft)
        attn1 = attn1.view(-1, ft_num, ft_num, self.h)
        attn1 = torch.sigmoid(attn1)
        attn = self.attention(query, key, mask=None, dropout=None, softmax=self.soft_max)
        attn = attn.view(batch_size, -1, num, num).contiguous()
        attn = attn.permute(0, 2, 3, 1).contiguous()
        one_mm = torch.eye(num).cuda()
        attn = attn * roi_mask[:, :, :, None].cuda()
        atten_mask = torch.zeros(attn.size()).cuda()
        values1, idces1 = torch.topk(attn, node_num // 2, dim=2)
        atten_mask[:, :, idces1, :] = 1.0
        attn_min = F.relu(attn - one_mm[None, :, :, None]) * -1
        values2, idces2 = torch.topk(attn_min, node_num // 2, dim=2)
        atten_mask[:, :, idces2, :] = 1.0
        attn1 = attn1 * roi_mask[:, :, :, None].cuda() * atten_mask.cuda()
        return attn1

class MultiHeadedAttention_3_type(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1, flag=1, softmax=True):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.h = h
        self.soft_max = softmax
        self.sptatial_flag = flag
        if self.sptatial_flag == 0:
            self.len_d_model = d_model * 2
        elif self.sptatial_flag == 1:
            self.len_d_model = d_model * 2 + 4
        elif self.sptatial_flag == 2:
            self.len_d_model = d_model * 2 + 6
        elif self.sptatial_flag == 3:
            self.len_d_model = d_model * 2 + 5
        elif self.sptatial_flag == 4:
            self.len_d_model = d_model * 2 + 2
        elif self.sptatial_flag == 5:
            self.len_d_model = d_model * 2 + 3
        self.linear_layer = nn.Linear(self.len_d_model, self.h)
        self.attention = Attention_4()

    def get_sptatial_feature(self, s_query, s_key):
        if self.sptatial_flag == 1:
            out = torch.cat((s_query, s_key), dim=3)
        elif self.sptatial_flag == 2:
            s_query_len = s_query[:, :, :, 1] - s_query[:, :, :, 0]
            s_query_len = s_query_len.unsqueeze(3)
            s_key_len = s_key[:, :, :, 1] - s_key[:, :, :, 0]
            s_key_len = s_key_len.unsqueeze(3)
            out = torch.cat((s_query, s_query_len, s_key, s_key_len), dim=3)

        elif self.sptatial_flag == 3:
            s_query_len = s_query[:, :, :, 1] - s_query[:, :, :, 0]
            s_key_len = s_key[:, :, :, 1] - s_key[:, :, :, 0]
            s_1 = (s_query - s_key).sum(dim=3) / 2 / s_key_len
            s_1 = s_1.unsqueeze(3)
            s_2 = s_query_len / s_key_len
            s_2 = s_2.unsqueeze(3)
            union1 = torch.min(s_query[:, :, :, 0], s_key[:, :, :, 0])
            union2 = torch.max(s_query[:, :, :, 1], s_key[:, :, :, 1])
            inter1 = torch.max(s_query[:, :, :, 0], s_key[:, :, :, 0])
            inter2 = torch.min(s_query[:, :, :, 1], s_key[:, :, :, 1])
            s_3 = F.relu((inter2 - inter1) / (union2 - union1))
            s_3 = s_3.unsqueeze(3)
            s_query_len = s_query_len.unsqueeze(3)
            s_key_len = s_key_len.unsqueeze(3)
            out = torch.cat((s_1, s_2, s_3, s_query_len, s_key_len), dim=3)

        elif self.sptatial_flag == 4:
            s_query_len = s_query[:, :, :, 1] - s_query[:, :, :, 0]
            s_key_len = s_key[:, :, :, 1] - s_key[:, :, :, 0]
            s_1 = (s_query - s_key).sum(dim=3) / 2 / s_key_len
            s_2 = s_query_len / s_key_len
            s_2 = torch.exp(s_2)
            s_1 = s_1.unsqueeze(3)
            s_2 = s_2.unsqueeze(3)
            out = torch.cat((s_1, s_2), dim=3)

        elif self.sptatial_flag == 5:
            s_query_len = s_query[:, :, :, 1] - s_query[:, :, :, 0]
            s_key_len = s_key[:, :, :, 1] - s_key[:, :, :, 0]
            s_1 = (s_query - s_key).sum(dim=3) / 2 / s_key_len
            s_2 = s_query_len / s_key_len
            s_2 = torch.exp(s_2)
            s_3 = (s_key - s_query).sum(dim=3) / 2 / s_query_len
            s_1 = s_1.unsqueeze(3)
            s_2 = s_2.unsqueeze(3)
            s_3 = s_3.unsqueeze(3)
            out = torch.cat((s_1, s_2, s_3), dim=3)
        return out

    def forward(self, query, key, s_query, s_key,roi_mask1,roi_mask2,node_num):
        ft_num = query.size()[-2]
        batch_size, num, _, chinnal = query.size()
        spat_feature = self.get_sptatial_feature(s_query, s_key)
        ft = torch.cat((query, key, spat_feature), dim=3)
        ft = ft.view(-1, self.len_d_model)
        # 2) Apply attention on all the projected vectors in batch.
        attn1 = self.linear_layer(ft)
        attn1 = attn1.view(-1, ft_num, ft_num, self.h)
        attn1 = torch.sigmoid(attn1)
        attn = self.attention(query, key, mask=None, dropout=None, softmax=self.soft_max)
        attn = attn.view(batch_size, -1, num, num).contiguous()
        attn = attn.permute(0, 2, 3, 1).contiguous()
        attn_1 = attn * roi_mask1[:, :, :, None].cuda()
        attn2 = attn * roi_mask2[:, :, :, None].cuda()
        atten_mask = torch.zeros(attn.size()).cuda()
        values1, idces1 = torch.topk(attn_1, node_num // 2, dim=2)
        atten_mask[:, :, idces1, :] = 1.0
        values2, idces2 = torch.topk(attn2, node_num // 2, dim=2)
        atten_mask[:, :, idces2, :] = 1.0
        roi_mask = roi_mask2 + roi_mask1
        attn1 = attn1 * roi_mask[:, :, :, None].cuda() * atten_mask.cuda()
        return attn1

class MultiHeadedAttention_3_type_min(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1, flag=1, softmax=True):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.h = h
        self.soft_max = softmax
        self.sptatial_flag = flag
        if self.sptatial_flag == 0:
            self.len_d_model = d_model * 2
        elif self.sptatial_flag == 1:
            self.len_d_model = d_model * 2 + 4
        elif self.sptatial_flag == 2:
            self.len_d_model = d_model * 2 + 6
        elif self.sptatial_flag == 3:
            self.len_d_model = d_model * 2 + 5
        elif self.sptatial_flag == 4:
            self.len_d_model = d_model * 2 + 2
        elif self.sptatial_flag == 5:
            self.len_d_model = d_model * 2 + 3
        self.linear_layer = nn.Linear(self.len_d_model, self.h)
        self.attention = Attention_4()

    def get_sptatial_feature(self, s_query, s_key):
        if self.sptatial_flag == 1:
            out = torch.cat((s_query, s_key), dim=3)
        elif self.sptatial_flag == 2:
            s_query_len = s_query[:, :, :, 1] - s_query[:, :, :, 0]
            s_query_len = s_query_len.unsqueeze(3)
            s_key_len = s_key[:, :, :, 1] - s_key[:, :, :, 0]
            s_key_len = s_key_len.unsqueeze(3)
            out = torch.cat((s_query, s_query_len, s_key, s_key_len), dim=3)

        elif self.sptatial_flag == 3:
            s_query_len = s_query[:, :, :, 1] - s_query[:, :, :, 0]
            s_key_len = s_key[:, :, :, 1] - s_key[:, :, :, 0]
            s_1 = (s_query - s_key).sum(dim=3) / 2 / s_key_len
            s_1 = s_1.unsqueeze(3)
            s_2 = s_query_len / s_key_len
            s_2 = s_2.unsqueeze(3)
            union1 = torch.min(s_query[:, :, :, 0], s_key[:, :, :, 0])
            union2 = torch.max(s_query[:, :, :, 1], s_key[:, :, :, 1])
            inter1 = torch.max(s_query[:, :, :, 0], s_key[:, :, :, 0])
            inter2 = torch.min(s_query[:, :, :, 1], s_key[:, :, :, 1])
            s_3 = F.relu((inter2 - inter1) / (union2 - union1))
            s_3 = s_3.unsqueeze(3)
            s_query_len = s_query_len.unsqueeze(3)
            s_key_len = s_key_len.unsqueeze(3)
            out = torch.cat((s_1, s_2, s_3, s_query_len, s_key_len), dim=3)

        elif self.sptatial_flag == 4:
            s_query_len = s_query[:, :, :, 1] - s_query[:, :, :, 0]
            s_key_len = s_key[:, :, :, 1] - s_key[:, :, :, 0]
            s_1 = (s_query - s_key).sum(dim=3) / 2 / s_key_len
            s_2 = s_query_len / s_key_len
            s_2 = torch.exp(s_2)
            s_1 = s_1.unsqueeze(3)
            s_2 = s_2.unsqueeze(3)
            out = torch.cat((s_1, s_2), dim=3)

        elif self.sptatial_flag == 5:
            s_query_len = s_query[:, :, :, 1] - s_query[:, :, :, 0]
            s_key_len = s_key[:, :, :, 1] - s_key[:, :, :, 0]
            s_1 = (s_query - s_key).sum(dim=3) / 2 / s_key_len
            s_2 = s_query_len / s_key_len
            s_2 = torch.exp(s_2)
            s_3 = (s_key - s_query).sum(dim=3) / 2 / s_query_len
            s_1 = s_1.unsqueeze(3)
            s_2 = s_2.unsqueeze(3)
            s_3 = s_3.unsqueeze(3)
            out = torch.cat((s_1, s_2, s_3), dim=3)
        return out

    def forward(self, query, key, s_query, s_key, roi_mask1, roi_mask2, node_num):
        ft_num = query.size()[-2]
        batch_size, num, _, chinnal = query.size()
        spat_feature = self.get_sptatial_feature(s_query, s_key)
        ft = torch.cat((query, key, spat_feature), dim=3)
        ft = ft.view(-1, self.len_d_model)
        # 2) Apply attention on all the projected vectors in batch.
        attn1 = self.linear_layer(ft)
        attn1 = attn1.view(-1, ft_num, ft_num, self.h)
        attn1 = torch.sigmoid(attn1)
        attn = self.attention(query, key, mask=None, dropout=None, softmax=self.soft_max)
        attn = attn.view(batch_size, -1, num, num).contiguous()
        attn = attn.permute(0, 2, 3, 1).contiguous()
        one_mm = torch.eye(num).cuda()
        attn_1 = attn * roi_mask1[:, :, :, None].cuda()
        attn2 = attn * roi_mask2[:, :, :, None].cuda()
        atten_mask = torch.zeros(attn.size()).cuda()
        values1, idces1 = torch.topk(attn_1, node_num // 4, dim=2)
        atten_mask[:, :, idces1, :] = 1.0
        attn1_min = F.relu(attn_1 - one_mm[None, :, :, None]) * -1
        values1_min, idces1_min = torch.topk(attn1_min, node_num // 4, dim=2)
        atten_mask[:, :, idces1_min, :] = 1.0
        values2, idces2 = torch.topk(attn2, node_num // 2, dim=2)
        atten_mask[:, :, idces2, :] = 1.0
        attn2_min = F.relu(attn2 - one_mm[None, :, :, None]) * -1
        values2_min, idces2_min = torch.topk(attn2_min, node_num // 4, dim=2)
        atten_mask[:, :, idces2_min, :] = 1.0
        roi_mask = roi_mask2 + roi_mask1
        attn1 = attn1 * roi_mask[:, :, :, None].cuda() * atten_mask.cuda()
        return attn1




class Graph_module_net_0(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features,mid_features, out_features,type_num,childs_num,multi_head=1,dropout=0.1, atten_dropout=0.1,group=4,bias=True):
        super(Graph_module_net_0, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.type_num=type_num
        self.multi_head=multi_head
        self.childs_num=childs_num
        self.dropout = dropout
        self.atten=MultiHeadedAttention_3(h=self.multi_head,d_model=in_features,dropout=atten_dropout)
        self.atten2 = MultiHeadedAttention_3(h=self.multi_head, d_model=mid_features, dropout=atten_dropout)
        self.agg1 = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=mid_features,
                                            kernel_size= 1, stride=1, groups=group,
                                            bias=bias),
                                  nn.ReLU(inplace=True))
        self.agg2 = nn.Sequential(nn.Conv1d(in_channels=mid_features,out_channels=out_features,
                                           kernel_size=1,stride=1,groups=group,bias=bias),
                                 nn.ReLU(inplace=True))
        self.layer_nomarl=nn.LayerNorm(out_features,eps=0.000001)
        for module in self.modules():
            if isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(0.0)
        #self.get_adj_mm()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    def get_weigth_fuction(self,x,roi_mask):
        batch_size, num, chinnal = x.size()
        source_feat=x.unsqueeze(1).repeat(1,num,1,1)
        adj_feat=x.unsqueeze(2).repeat(1,1,num,1)
        atten=self.atten(source_feat,adj_feat,roi_mask)
        return atten
    def get_weigth_fuction2(self,x,roi_mask):
        batch_size, num, chinnal = x.size()
        source_feat=x.unsqueeze(1).repeat(1,num,1,1)
        adj_feat=x.unsqueeze(2).repeat(1,1,num,1)
        atten=self.atten2(source_feat,adj_feat,roi_mask)
        return atten
    def forward(self, input,masks_roi,score_mask):
        bacth,num,chinal=input.size()
        one_mm=torch.eye(num).cuda()
        masks_roi=masks_roi*score_mask[:,None,:]
        f_socrce_mask=(score_mask==0).float()
        f_mask=torch.diag_embed(f_socrce_mask)
        atten=self.get_weigth_fuction(input,masks_roi)

        atten=atten+f_mask[:,:,:,None].cuda()
        atten_mask = masks_roi +f_mask
        #sun_mask=atten_mask.sum(dim=2)

        atten=atten/4#sun_mask[:,:,None,None].cuda()#+one_mm[None,:,:,None]
        atten=atten.permute(0,3,2,1).contiguous()
        input=input.permute(0,2,1).contiguous()
        output1 = self.agg1(input)

        output1_mul=output1.view(bacth,self.multi_head,-1,num)
        output1_mul=torch.matmul(output1_mul,atten)
        output1_mul=output1_mul.view(bacth,-1,num)
        output1=output1+output1_mul
        out_input=output1.permute(0,2,1).contiguous()

        atten2 = self.get_weigth_fuction2(out_input, masks_roi)

        atten2 = atten2 + f_mask[:, :, :, None].cuda()


        atten2 = atten2 / 4  # sun_mask[:,:,None,None].cuda()#+one_mm[None,:,:,None]
        atten2 = atten2.permute(0, 3, 2, 1).contiguous()
        output2 = self.agg2(output1)

        output2_mul = output2.view(bacth, self.multi_head, -1, num)

        output2_mul = torch.matmul(output2_mul, atten2)

        output2_mul = output2_mul.view(bacth, -1, num)

        output2_mul=output2_mul.permute(0,2,1).contiguous().view(bacth*num,-1)
        output2_mul=self.layer_nomarl(output2_mul)
        output2_mul=output2_mul.view(bacth,num,-1).permute(0,2,1).contiguous()
        output2 = output2 + output2_mul

        output2=output2.permute(0,2,1)
        return output2


def feature_measure(gt,node_feat,weight_type ='euclidean',alte=3):
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
class Graph_module_net_0_loss_2(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features,mid_features, out_features,type_num,childs_num,multi_head=1,dropout=0.1, atten_dropout=0.1,group=4,bias=True):
        super(Graph_module_net_0_loss_2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.type_num=type_num
        self.multi_head=multi_head
        self.childs_num=childs_num
        self.dropout = dropout
        self.atten=MultiHeadedAttention_3(h=self.multi_head,d_model=in_features,dropout=atten_dropout)
        self.atten2 = MultiHeadedAttention_3(h=self.multi_head, d_model=mid_features, dropout=atten_dropout)
        self.agg1 = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=mid_features,
                                            kernel_size= 1, stride=1, groups=group,
                                            bias=bias),
                                  nn.ReLU(inplace=True))
        self.agg2 = nn.Sequential(nn.Conv1d(in_channels=mid_features,out_channels=out_features,
                                           kernel_size=1,stride=1,groups=group,bias=bias),
                                 nn.ReLU(inplace=True))
        self.layer_nomarl1=nn.LayerNorm(mid_features,eps=0.000001)
        self.layer_nomarl2 = nn.LayerNorm(out_features, eps=0.000001)
        self.gt=nn.Sequential(nn.Linear(in_features,out_features),
                              nn.ReLU())
        for module in self.modules():
            if isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(0.0)
        #self.get_adj_mm()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    def get_weigth_fuction(self,x,roi_mask):
        batch_size, num, chinnal = x.size()
        source_feat=x.unsqueeze(1).repeat(1,num,1,1)
        adj_feat=x.unsqueeze(2).repeat(1,1,num,1)
        atten=self.atten(source_feat,adj_feat,roi_mask,self.childs_num)
        return atten
    def get_weigth_fuction2(self,x,roi_mask):
        batch_size, num, chinnal = x.size()
        source_feat=x.unsqueeze(1).repeat(1,num,1,1)
        adj_feat=x.unsqueeze(2).repeat(1,1,num,1)
        atten=self.atten2(source_feat,adj_feat,roi_mask,self.childs_num)
        return atten
    def forward(self, input,masks_roi,score_mask,gt_feat):
        gts=self.gt(gt_feat)
        bacth,num,chinal=input.size()
        gts = gts.view(bacth,-1,self.out_features)
        masks_roi=masks_roi*score_mask[:,None,:]
        f_socrce_mask=(score_mask==0).float()
        f_mask=torch.diag_embed(f_socrce_mask)
        atten=self.get_weigth_fuction(input,masks_roi)

        atten=atten+f_mask[:,:,:,None].cuda()
        atten=atten/self.childs_num
        atten=atten.permute(0,3,2,1).contiguous()
        input=input.permute(0,2,1).contiguous()
        output1 = self.agg1(input)
        output1_mul=output1.view(bacth,self.multi_head,-1,num)
        output1_mul=torch.matmul(output1_mul,atten)
        output1_mul=output1_mul.view(bacth,-1,num)
        output1_mul = output1_mul.permute(0, 2, 1).contiguous().view(bacth * num, -1)
        output1_mul = self.layer_nomarl1(output1_mul)
        output1_mul = output1_mul.view(bacth, num, -1).permute(0, 2, 1).contiguous()
        output1=output1+output1_mul
        out_input=output1.permute(0,2,1).contiguous()
        atten2 = self.get_weigth_fuction2(out_input, masks_roi)
        atten2 = atten2 + f_mask[:, :, :, None].cuda()
        atten2 = atten2 / self.childs_num
        atten2 = atten2.permute(0, 3, 2, 1).contiguous()
        output2 = self.agg2(output1)
        output2_mul = output2.view(bacth, self.multi_head, -1, num)
        output2_mul = torch.matmul(output2_mul, atten2)
        output2_mul = output2_mul.view(bacth, -1, num)
        output2_mul=output2_mul.permute(0,2,1).contiguous().view(bacth*num,-1)
        output2_mul=self.layer_nomarl2(output2_mul)
        node_feat = output2_mul
        node_feat = node_feat.view(bacth, -1, self.out_features)
        output2_mul=output2_mul.view(bacth,num,-1).permute(0,2,1).contiguous()
        output2 = output2 + output2_mul
        output2=output2.permute(0,2,1)
        return output2,gts,node_feat

class Graph_module_net_0_loss(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features,mid_features, out_features,type_num,childs_num,multi_head=1,dropout=0.1, atten_dropout=0.1,group=4,bias=True):
        super(Graph_module_net_0_loss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.type_num=type_num
        self.multi_head=multi_head
        self.childs_num=childs_num
        self.dropout = dropout
        self.atten=MultiHeadedAttention_3_min(h=self.multi_head,d_model=in_features,dropout=atten_dropout)
        self.agg1 = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=mid_features,
                                            kernel_size= 1, stride=1, groups=group,
                                            bias=bias),
                                  nn.ReLU(inplace=True))
        self.agg2 = nn.Sequential(nn.Conv1d(in_channels=mid_features,out_channels=out_features,
                                           kernel_size=1,stride=1,groups=group,bias=bias),
                                 nn.ReLU(inplace=True))
        self.layer_nomarl1=nn.LayerNorm(mid_features,eps=0.000001)
        self.layer_nomarl2 = nn.LayerNorm(out_features, eps=0.000001)
        self.gt=nn.Sequential(nn.Linear(in_features,out_features),
                              nn.ReLU())
        for module in self.modules():
            if isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(0.0)
        #self.get_adj_mm()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    def get_weigth_fuction(self,x,roi_mask,x_spatial):
        batch_size, num, chinnal = x.size()
        new_spat=x_spatial[:,:,1:].cuda()
        source_spat=new_spat.unsqueeze(1).repeat(1,num,1,1)
        source_feat=x.unsqueeze(1).repeat(1,num,1,1)
        adj_spat=new_spat.unsqueeze(2).repeat(1,1,num,1)
        adj_feat=x.unsqueeze(2).repeat(1,1,num,1)
        atten=self.atten(source_feat,adj_feat,source_spat,adj_spat,roi_mask,self.childs_num)
        return atten
    def forward(self, input,masks_roi,score_mask,gt_feat,spat_feature):
        gts=self.gt(gt_feat)
        bacth,num,chinal=input.size()
        gts = gts.view(bacth,-1,self.out_features)
        masks_roi=masks_roi*score_mask[:,None,:]
        f_socrce_mask=(score_mask==0).float()
        f_mask=torch.diag_embed(f_socrce_mask)
        atten=self.get_weigth_fuction(input,masks_roi,spat_feature)
        atten=atten+f_mask[:,:,:,None].cuda()
        atten=atten/self.childs_num
        atten=atten.permute(0,3,2,1).contiguous()
        input=input.permute(0,2,1).contiguous()
        output1 = self.agg1(input)
        output1_mul=output1.view(bacth,self.multi_head,-1,num)
        output1_mul=torch.matmul(output1_mul,atten)
        output1_mul=output1_mul.view(bacth,-1,num)
        output1_mul = output1_mul.permute(0, 2, 1).contiguous().view(bacth * num, -1)
        output1_mul = self.layer_nomarl1(output1_mul)
        output1_mul = output1_mul.view(bacth, num, -1).permute(0, 2, 1).contiguous()
        output1=output1+output1_mul
        output2 = self.agg2(output1)
        output2_mul = output2.view(bacth, self.multi_head, -1, num)
        output2_mul = torch.matmul(output2_mul, atten)
        output2_mul = output2_mul.view(bacth, -1, num)
        output2_mul=output2_mul.permute(0,2,1).contiguous().view(bacth*num,-1)
        output2_mul=self.layer_nomarl2(output2_mul)
        node_feat = output2_mul
        node_feat = node_feat.view(bacth, -1, self.out_features)
        output2_mul=output2_mul.view(bacth,num,-1).permute(0,2,1).contiguous()
        output2 = output2 + output2_mul
        output2=output2.permute(0,2,1)
        return output2,gts,node_feat



class Graph_module_net_0_loss_type(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features,mid_features, out_features,type_num,childs_num,multi_head=1,dropout=0.1, atten_dropout=0.1,group=4,bias=True):
        super(Graph_module_net_0_loss_type, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.type_num=type_num
        self.multi_head=multi_head
        self.childs_num=childs_num
        self.dropout = dropout
        self.atten=MultiHeadedAttention_3_type_min(h=self.multi_head,d_model=in_features,dropout=atten_dropout)
        self.agg1 = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=mid_features,
                                            kernel_size= 1, stride=1, groups=group,
                                            bias=bias),
                                  nn.ReLU(inplace=True))
        self.agg2 = nn.Sequential(nn.Conv1d(in_channels=mid_features,out_channels=out_features,
                                           kernel_size=1,stride=1,groups=group,bias=bias),
                                 nn.ReLU(inplace=True))
        self.layer_nomarl1=nn.LayerNorm(mid_features,eps=0.000001)
        self.layer_nomarl2 = nn.LayerNorm(out_features, eps=0.000001)
        self.gt=nn.Sequential(nn.Linear(in_features,out_features),
                              nn.ReLU())
        for module in self.modules():
            if isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(0.0)
        #self.get_adj_mm()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    def get_weigth_fuction(self,x,roi_mask1,roi_mask2,x_spatial):
        batch_size, num, chinnal = x.size()
        new_spat = x_spatial[:, :, 1:].cuda()
        source_spat = new_spat.unsqueeze(1).repeat(1, num, 1, 1)
        source_feat = x.unsqueeze(1).repeat(1, num, 1, 1)
        adj_spat = new_spat.unsqueeze(2).repeat(1, 1, num, 1)
        adj_feat = x.unsqueeze(2).repeat(1, 1, num, 1)
        atten=self.atten(source_feat,adj_feat,source_spat,adj_spat,roi_mask1,roi_mask2,self.childs_num)
        return atten
    def forward(self, input,masks_roi1,masks_roi2,score_mask,gt_feat,spat_feature):
        gts=self.gt(gt_feat)
        bacth,num,chinal=input.size()
        gts = gts.view(bacth,-1,self.out_features)
        masks_roi1=masks_roi1*score_mask[:,None,:]
        masks_roi2 = masks_roi2 * score_mask[:, None, :]
        f_socrce_mask=(score_mask==0).float()
        f_mask=torch.diag_embed(f_socrce_mask)
        atten=self.get_weigth_fuction(input,masks_roi1,masks_roi2,spat_feature)
        atten=atten+f_mask[:,:,:,None].cuda()
        atten=atten/self.childs_num
        atten=atten.permute(0,3,2,1).contiguous()
        input=input.permute(0,2,1).contiguous()
        output1 = self.agg1(input)
        output1_mul=output1.view(bacth,self.multi_head,-1,num)
        output1_mul=torch.matmul(output1_mul,atten)
        output1_mul=output1_mul.view(bacth,-1,num)
        output1_mul = output1_mul.permute(0, 2, 1).contiguous().view(bacth * num, -1)
        output1_mul = self.layer_nomarl1(output1_mul)
        output1_mul = output1_mul.view(bacth, num, -1).permute(0, 2, 1).contiguous()
        output1=output1+output1_mul
        output2 = self.agg2(output1)
        output2_mul = output2.view(bacth, self.multi_head, -1, num)
        output2_mul = torch.matmul(output2_mul, atten)
        output2_mul = output2_mul.view(bacth, -1, num)
        output2_mul=output2_mul.permute(0,2,1).contiguous().view(bacth*num,-1)
        output2_mul=self.layer_nomarl2(output2_mul)
        node_feat = output2_mul
        node_feat = node_feat.view(bacth, -1, self.out_features)
        output2_mul=output2_mul.view(bacth,num,-1).permute(0,2,1).contiguous()
        output2 = output2 + output2_mul
        output2=output2.permute(0,2,1)
        return output2,gts,node_feat


class Graph_module_net_0_loss_type_2(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features,mid_features, out_features,type_num,childs_num,multi_head=1,dropout=0.1, atten_dropout=0.1,group=4,bias=True):
        super(Graph_module_net_0_loss_type_2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.type_num=type_num
        self.multi_head=multi_head
        self.childs_num=childs_num
        self.dropout = dropout
        self.atten=MultiHeadedAttention_3_type_min(h=self.multi_head,d_model=in_features,dropout=atten_dropout)
        self.atten2 = MultiHeadedAttention_3_type(h=self.multi_head, d_model=mid_features, dropout=atten_dropout)
        self.agg1 = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=mid_features,
                                            kernel_size= 1, stride=1, groups=group,
                                            bias=bias),
                                  nn.ReLU(inplace=True))
        self.agg2 = nn.Sequential(nn.Conv1d(in_channels=mid_features,out_channels=out_features,
                                           kernel_size=1,stride=1,groups=group,bias=bias),
                                 nn.ReLU(inplace=True))
        self.layer_nomarl1=nn.LayerNorm(mid_features,eps=0.000001)
        self.layer_nomarl2 = nn.LayerNorm(out_features, eps=0.000001)
        self.gt=nn.Sequential(nn.Linear(in_features,out_features),
                              nn.ReLU())
        for module in self.modules():
            if isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(0.0)
        #self.get_adj_mm()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    def get_weigth_fuction(self,x,roi_mask1,roi_mask2):
        batch_size, num, chinnal = x.size()
        source_feat=x.unsqueeze(1).repeat(1,num,1,1)
        adj_feat=x.unsqueeze(2).repeat(1,1,num,1)
        atten=self.atten(source_feat,adj_feat,roi_mask1,roi_mask2,self.childs_num)
        return atten
    def get_weigth_fuction2(self,x,roi_mask1,roi_mask2):
        batch_size, num, chinnal = x.size()
        source_feat=x.unsqueeze(1).repeat(1,num,1,1)
        adj_feat=x.unsqueeze(2).repeat(1,1,num,1)
        atten=self.atten2(source_feat,adj_feat,roi_mask1,roi_mask2,self.childs_num)
        return atten
    def forward(self, input,masks_roi1,masks_roi2,score_mask,gt_feat):
        gts=self.gt(gt_feat)
        bacth,num,chinal=input.size()
        gts = gts.view(bacth,-1,self.out_features)
        masks_roi1=masks_roi1*score_mask[:,None,:]
        masks_roi2 = masks_roi2 * score_mask[:, None, :]
        f_socrce_mask=(score_mask==0).float()
        f_mask=torch.diag_embed(f_socrce_mask)
        atten=self.get_weigth_fuction(input,masks_roi1,masks_roi2)

        atten=atten+f_mask[:,:,:,None].cuda()
        atten=atten/self.childs_num
        atten=atten.permute(0,3,2,1).contiguous()
        input=input.permute(0,2,1).contiguous()
        output1 = self.agg1(input)
        output1_mul=output1.view(bacth,self.multi_head,-1,num)
        output1_mul=torch.matmul(output1_mul,atten)
        output1_mul=output1_mul.view(bacth,-1,num)
        output1_mul = output1_mul.permute(0, 2, 1).contiguous().view(bacth * num, -1)
        output1_mul = self.layer_nomarl1(output1_mul)
        output1_mul = output1_mul.view(bacth, num, -1).permute(0, 2, 1).contiguous()
        output1=output1+output1_mul
        out_input=output1.permute(0,2,1).contiguous()
        atten2 = self.get_weigth_fuction2(out_input, masks_roi1,masks_roi2)
        atten2 = atten2 + f_mask[:, :, :, None].cuda()
        atten2 = atten2 / self.childs_num
        atten2 = atten2.permute(0, 3, 2, 1).contiguous()
        output2 = self.agg2(output1)
        output2_mul = output2.view(bacth, self.multi_head, -1, num)
        output2_mul = torch.matmul(output2_mul, atten2)
        output2_mul = output2_mul.view(bacth, -1, num)
        output2_mul=output2_mul.permute(0,2,1).contiguous().view(bacth*num,-1)
        output2_mul=self.layer_nomarl2(output2_mul)
        node_feat = output2_mul
        node_feat = node_feat.view(bacth, -1, self.out_features)
        output2_mul=output2_mul.view(bacth,num,-1).permute(0,2,1).contiguous()
        output2 = output2 + output2_mul
        output2=output2.permute(0,2,1)
        return output2,gts,node_feat


if __name__ == "__main__":
    x=torch.arange(1,464)
    x=x.view(1,1,-1).contiguous().float().cuda()
    x=x.repeat(2,1024,1).permute(0,2,1)
    models=Graph_module_net_0(in_features=1024,mid_features=512,out_features=1024,type_num=6,childs_num=[1,2,3,4,5,6],multi_head=8,dropout=0.1).cuda()
    out_put=models(x)