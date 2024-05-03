import os
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """
    def forward(self, query, key, mask=None, dropout=None,softmax=True):
        scores = F.cosine_similarity(query.repeat(1,1,key.size()[2],1), key,dim=3)
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, scores.size()[1], 1)
            scores = scores.masked_fill(mask == 0, -1e9)
        if softmax:
            p_attn = torch.softmax(scores,dim=2)
        else:
            p_attn=scores
        if dropout is not None:
            p_attn = dropout(p_attn)
        return p_attn
class Attention_1(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """
    def forward(self, query, key, mask=None, dropout=None,softmax=True):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        scores=scores.squeeze(2)
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, scores.size()[1], 1)
            scores = scores.masked_fill(mask == 0, -1e9)
        if softmax:
            p_attn = torch.softmax(scores, dim=2)
        else:
            p_attn = scores
        if dropout is not None:
            p_attn = dropout(p_attn)
        #print(p_attn.size())
        return p_attn

class Attention_2(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """
    def forward(self, query, key, mask=None, dropout=None,softmax=True):
        scores = torch.matmul(query, key.transpose(-2, -1))
        scores=scores.squeeze(2)
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, scores.size()[1], 1)
            scores = scores.masked_fill(mask == 0, -1e9)
        if softmax:
            p_attn = torch.softmax(scores, dim=2)
        else:
            p_attn = scores
        if dropout is not None:
            p_attn = dropout(p_attn)
        #print(p_attn.size())
        return p_attn

class Attention_3(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """
    def __init__(self):
        super().__init__()
        self.bas=nn.Parameter(torch.tensor(0.0).float())

    def forward(self, query, key, mask=None, dropout=None,softmax=True):
        scores = torch.matmul(query, key.transpose(-2, -1))
        scores=scores.squeeze(2)
        scores=scores+self.bas
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, scores.size()[1], 1)
            scores = scores.masked_fill(mask == 0, -1e9)
        if softmax:
            p_attn = torch.softmax(scores, dim=2)
        else:
            p_attn = scores
        if dropout is not None:
            p_attn = dropout(p_attn)

        return p_attn

class Attention_4(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """
    def forward(self, query, key, mask=None, dropout=None,softmax=True):
        scores = F.cosine_similarity(query.repeat(1,1,key.size()[2],1), key,dim=3)
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, scores.size()[1], 1)
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.relu(scores)
        if softmax:
            p_attn=p_attn/p_attn.size(2)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return p_attn
class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1,softmax=True):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.soft_max=softmax
        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(2)])
        self.attention = Attention_2()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key))]

        # 2) Apply attention on all the projected vectors in batch.
        attn = self.attention(query, key, mask=mask, dropout=self.dropout,softmax=self.soft_max)

        # 3) "Concat" using a view and apply a final linear.

        return attn

class MultiHeadedAttention_2(nn.Module):
    """
    Take in model size and number of heads.
    """
    def __init__(self, h, d_model, dropout=0.1,softmax=True):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.soft_max=softmax
        self.attention = Attention_4()
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, query, key, mask=None):
        batch_size = query.size(0)
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query=query.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        key=key.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        # 2) Apply attention on all the projected vectors in batch.
        attn = self.attention(query, key, mask=mask, dropout=None,softmax=self.soft_max)

        # 3) "Concat" using a view and apply a final linear.

        return attn

class MultiHeadedAttention_3(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1,softmax=True):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.h = h
        self.soft_max=softmax
        self.linear_layer = nn.Linear(d_model*2, self.h)

    def forward(self, query, key, mask=None):
        node_num=key.size(1)
        chinal=query.size(2)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query=query.repeat(1,key.size()[1],1)
        ft=torch.cat((query,key),dim=2)
        ft=ft.view(-1,chinal*2)
        # 2) Apply attention on all the projected vectors in batch.
        attn = self.linear_layer(ft)
        attn=attn.view(-1,node_num,self.h)
        attn=attn.permute(0,2,1).contiguous()
        if self.soft_max:
            attn = torch.softmax(attn, dim=2)
        # 3) "Concat" using a view and apply a final linear.
        return attn

class MultiHeadedAttention_4(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1,softmax=True):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.h = h
        self.soft_max=softmax
        self.linear_layer = nn.Linear(d_model, self.h)

    def forward(self, query, key, mask=None):
        node_num=key.size(1)
        chinal=query.size(2)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query=query.squeeze()
        ft=query[:,None,:]+key
        ft=ft.view(-1,chinal)
        # 2) Apply attention on all the projected vectors in batch.
        attn = self.linear_layer(ft)
        attn=attn.view(-1,node_num,self.h)
        attn=attn.permute(0,2,1).contiguous()
        if self.soft_max:
            attn = torch.softmax(attn, dim=2)
        # 3) "Concat" using a view and apply a final linear.
        return attn

class MultiHeadedAttention_5(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1,softmax=True):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.h = h
        self.soft_max=softmax
        self.linear_layer = nn.Linear(d_model, self.h)

    def forward(self, query, key, mask=None):
        node_num=key.size(1)
        chinal=query.size(2)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query=query.squeeze()
        ft=query[:,None,:]-key
        ft=ft.view(-1,chinal)
        # 2) Apply attention on all the projected vectors in batch.
        attn = self.linear_layer(ft)
        attn=attn.view(-1,node_num,self.h)
        attn=attn.permute(0,2,1).contiguous()
        if self.soft_max:
            attn = torch.softmax(attn, dim=2)
        # 3) "Concat" using a view and apply a final linear.
        return attn

class MultiHeadedAttention_6(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1,softmax=True):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.h = h
        self.soft_max=softmax
        self.linear_layer = nn.Linear(d_model, self.h)

    def forward(self, query, key, mask=None):
        node_num=key.size(1)
        chinal=query.size(2)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query=query.squeeze()
        ft=query[:,None,:]*key
        ft=ft.view(-1,chinal)
        # 2) Apply attention on all the projected vectors in batch.
        attn = self.linear_layer(ft)
        attn=attn.view(-1,node_num,self.h)
        attn=attn.permute(0,2,1).contiguous()
        if self.soft_max:
            attn = torch.softmax(attn, dim=2)
        # 3) "Concat" using a view and apply a final linear.
        return attn

class Graph_aggre_fuction(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features,type_num, group=1,bias=True):
        super(Graph_aggre_fuction, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.type_num=type_num
        self.agg = nn.Sequential(nn.Conv1d(in_channels=in_features,out_channels=out_features,
                                           kernel_size=self.type_num+1,stride=self.type_num+1,padding=0,groups=group,bias=bias),
                                 nn.ReLU(inplace=True))

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def get_graph_feature(self,x, type_num=4):
        """
        :param x:
        :param prev_x:
        :param k:
        :param idx:
        :param r: output downsampling factor (-1 for no downsampling)
        :param style: method to get graph feature
        :return:
        """
        batch_size = x.size(0)
        chinnals = x.size(1)  # if prev_x is None else prev_x.size(2)
        x = x.view(batch_size, chinnals, -1)
        node_x = x[:, :, 1:].contiguous()
        node_x = node_x.permute(0, 2, 1).contiguous()

        node_x = node_x.view(batch_size, -1, chinnals * type_num).contiguous()
        # print(idx_knn.shape)
        padding_x = x.new_zeros(batch_size, x.size(2) - node_x.size(1), chinnals * type_num)
        cat_x = torch.cat((node_x, padding_x), dim=1)
        x = x.permute(0, 2, 1).contiguous()
        feature = torch.cat((cat_x, x), dim=2)
        feature = feature.view(batch_size, -1, type_num + 1, chinnals).contiguous()  # (batch_size,T,type_num+1,chinnal)
        return feature

    def forward(self, input):
        x=self.get_graph_feature(input,self.type_num) #batch_size,  chinnal,length = input.size()
        #print(x.size())
        batch_size,length, adj_num,chinnal  = x.size()
        x=x.view(batch_size*length,adj_num,chinnal).contiguous()
        x=x.permute(0,2,1)
        output=self.agg(x)
        #print(output.size())
        output=output.view(batch_size,length,chinnal).contiguous()
        output=output.permute(0,2,1)
        return output

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
class Graph_module_loss(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features,type_num,childs_num,multi_head,dropout=0.1, group=1,bias=True):
        super(Graph_module_loss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.type_num=type_num
        self.multi_head=multi_head
        self.childs_num=childs_num
        self.atten=MultiHeadedAttention(h=self.multi_head,d_model=in_features,dropout=dropout)
        self.agg = nn.Sequential(nn.Conv1d(in_channels=in_features,out_channels=out_features,
                                           kernel_size=self.type_num,stride=self.type_num,groups=group,bias=bias),
                                 nn.ReLU(inplace=True))
        self.gt = nn.Sequential(nn.Linear(in_features, out_features),
                                 nn.ReLU(inplace=True))
        self.source = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=out_features,
                                               kernel_size=1, stride=1, groups=group,
                                               bias=bias),
                                     nn.ReLU(inplace=True))
        self.get_adj_mm()
    def get_adj_mm(self):
        childs_num=np.array(self.childs_num)
        childs_num=childs_num[:self.type_num]
        self.node_num=np.sum(childs_num)
        mask=torch.zeros((self.type_num,self.node_num))
        start=0
        end=0
        for i in range(self.type_num):
            end=end+childs_num[i]
            mask[i,start:end]=1
            start=start+childs_num[i]
        self.mask=mask
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    def get_graph_feature(self,x, type_num):
        """
        :param x:
        :param prev_x:
        :param k:
        :param idx:
        :param r: output downsampling factor (-1 for no downsampling)
        :param style: method to get graph feature
        :return:
        """
        batch_size = x.size(0)
        chinnals = x.size(2)  # if prev_x is None else prev_x.size(2)
        x = x.view(batch_size,-1, chinnals)
        node_x = x[:, 1:,:].contiguous()
        node_x = node_x.view(batch_size, -1, chinnals * type_num).contiguous()
        source_x=x[:,:node_x.size()[1],:]
        # print(idx_knn.shape)
        feature = torch.cat((node_x, source_x), dim=2)
        feature=feature.view(batch_size, -1, type_num + 1, chinnals).contiguous()
        return feature
    def get_weigth_fuction(self,x):
        batch_size, length, adj_num, chinnal = x.size()
        source_feat=x[:,:,-1,:]
        adj_feat=x[:,:,:-1,:]
        source_feat=source_feat.view(batch_size*length,1,chinnal)
        adj_feat=adj_feat.view(batch_size*length,adj_num-1,chinnal)
        atten=self.atten(source_feat,adj_feat)
        return atten
    def agg_node_fuction(self,x,atten):
        batch_size, length, adj_num, chinnal = x.size()
        atten=atten.permute(0,2,1).contiguous()
        atten=atten.view(batch_size,length,adj_num-1,self.multi_head)
        node_feat=x[:,:,:-1,:]
        source_feat = x[:, :, -1, :]
        source_feat=source_feat.unsqueeze(2)
        node_feat=node_feat.view(batch_size,length,adj_num-1,self.multi_head,-1)
        node_feat=node_feat*atten[:,:,:,:,None]
        node_feat=node_feat.view(batch_size*length,adj_num-1,chinnal)
        mask=self.mask.repeat(batch_size*length,1,1)
        mask=mask.cuda(node_feat.device)
        node_feat=torch.bmm(mask,node_feat)
        node_feat = node_feat.view(batch_size, length, self.type_num, chinnal)
        return node_feat,source_feat
    def forward(self, input,gt,):
        gt_batch,gt_adj,gt_chinnal=gt.size()
        gt=gt.view(-1,gt_chinnal)
        gt=self.gt(gt)
        gt=gt.view(gt_batch,gt_adj,-1)
        x=self.get_graph_feature(input,self.node_num) #batch_size,  chinnal,length = input.size()
        atten=self.get_weigth_fuction(x)
        out_node_feat,out_source_feat=self.agg_node_fuction(x,atten)
        batch_size,length, adj_num,chinnal  = out_node_feat.size()
        out_node_feat=out_node_feat.view(batch_size*length,adj_num,chinnal).contiguous()
        out_source_feat = out_node_feat.view(batch_size * length, 1, chinnal).contiguous()
        out_node_feat=out_node_feat.permute(0,2,1)
        out_source_feat = out_source_feat.permute(0, 2, 1)
        output_node_feat=self.agg(out_node_feat)
        node_feat=out_node_feat.view(batch_size,length,-1)[:,0,:]
        gt=gt[:,0,:].contiguous()
        scorce=feature_measure(gt=gt,node_feat=node_feat)
        out_source_feat=self.source(out_source_feat)
        output=output_node_feat+out_source_feat
        output=output.view(batch_size,length,-1).contiguous()
        return output,scorce

class Graph_module_0_loss(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features,type_num,childs_num,multi_head,dropout=0.1, group=1,bias=True):
        super(Graph_module_0_loss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.type_num=type_num
        self.multi_head=multi_head
        self.childs_num=childs_num
        self.atten=MultiHeadedAttention(h=self.multi_head,d_model=in_features,dropout=dropout)
        self.agg = nn.Sequential(nn.Conv1d(in_channels=in_features,out_channels=out_features,
                                           kernel_size=1,stride=1,groups=group,bias=bias),
                                 nn.ReLU(inplace=True))
        self.gt = nn.Sequential(nn.Linear(in_features, out_features),
                                nn.ReLU(inplace=True))
        self.get_adj_mm()
    def get_adj_mm(self):
        childs_num=np.array(self.childs_num)
        childs_num=childs_num[:self.type_num]
        self.node_num=np.sum(childs_num)
        mask=torch.zeros((self.type_num,self.node_num))
        start=0
        end=0
        for i in range(self.type_num):
            end=end+childs_num[i]
            mask[i,start:end]=1
            start=start+childs_num[i]
        self.mask=mask
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    def get_graph_feature(self,x, type_num):
        """
        :param x:
        :param prev_x:
        :param k:
        :param idx:
        :param r: output downsampling factor (-1 for no downsampling)
        :param style: method to get graph feature
        :return:
        """
        batch_size = x.size(0)
        chinnals = x.size(2)  # if prev_x is None else prev_x.size(2)
        x = x.view(batch_size,-1, chinnals)
        node_x = x[:, 1:,:].contiguous()
        node_x = node_x.view(batch_size, -1, chinnals * type_num).contiguous()
        source_x=x[:,:node_x.size()[1],:]
        # print(idx_knn.shape)
        feature = torch.cat((node_x, source_x), dim=2)
        feature=feature.view(batch_size, -1, type_num + 1, chinnals).contiguous()
        return feature
    def get_weigth_fuction(self,x):
        batch_size, length, adj_num, chinnal = x.size()
        source_feat=x[:,:,-1,:]
        adj_feat=x[:,:,:-1,:]
        source_feat=source_feat.view(batch_size*length,1,chinnal)
        adj_feat=adj_feat.view(batch_size*length,adj_num-1,chinnal)
        atten=self.atten(source_feat,adj_feat)
        return atten
    def agg_node_fuction(self,x,atten):
        batch_size, length, adj_num, chinnal = x.size()
        atten=atten.permute(0,2,1).contiguous()
        atten=atten.view(batch_size,length,adj_num-1,self.multi_head)
        node_feat=x[:,:,:-1,:]
        source_feat = x[:, :, -1, :]
        source_feat=source_feat.unsqueeze(2)
        node_feat=node_feat.view(batch_size,length,adj_num-1,self.multi_head,-1)
        node_feat=node_feat*atten[:,:,:,:,None]
        node_feat=node_feat.view(batch_size*length,adj_num-1,chinnal)
        node_feat=node_feat.sum(dim=1)
        node_feat = node_feat.view(batch_size, length, 1, chinnal)
        return node_feat,source_feat
    def forward(self, input,gt):
        gt_batch, gt_adj, gt_chinnal = gt.size()
        gt = gt.view(-1, gt_chinnal)
        gt = self.gt(gt)
        gt = gt.view(gt_batch, gt_adj, -1)
        x=self.get_graph_feature(input,self.node_num) #batch_size,  chinnal,length = input.size()
        atten=self.get_weigth_fuction(x)
        out_node_feat,out_source_feat=self.agg_node_fuction(x,atten)
        batch_size,length, adj_num,chinnal  = out_node_feat.size()
        out_node_feat=out_node_feat.view(batch_size*length,adj_num,chinnal).contiguous()
        out_source_feat = out_source_feat.view(batch_size * length, 1, chinnal).contiguous()
        out_node_feat=out_node_feat.permute(0,2,1)
        out_source_feat = out_source_feat.permute(0, 2, 1)
        output_node_feat=self.agg(out_node_feat)
        node_feat = out_node_feat.view(batch_size, length, -1)[:, 0, :]
        gt = gt[:, 0, :].contiguous()
        scorce = feature_measure(gt=gt, node_feat=node_feat)
        output_source_feat = self.agg(out_source_feat)
        output=output_node_feat+output_source_feat
        output=output.view(batch_size,length,-1).contiguous()
        return output,scorce


class Graph_module_1_loss(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, type_num, childs_num, multi_head, dropout=0.1, group=1, bias=True):
        super(Graph_module_1_loss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.type_num = type_num
        self.multi_head = multi_head
        self.childs_num = childs_num
        self.atten = MultiHeadedAttention(h=self.multi_head, d_model=in_features, dropout=dropout)
        self.agg = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=out_features,
                                           kernel_size=self.type_num, stride=self.type_num, groups=group, bias=bias),
                                 nn.ReLU(inplace=True))
        self.gt = nn.Sequential(nn.Linear(in_features, out_features),
                                nn.ReLU(inplace=True))
        self.source = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=out_features,
                                              kernel_size=1, stride=1, groups=group,
                                              bias=bias),
                                    nn.ReLU(inplace=True))
        self.get_adj_mm()

    def get_adj_mm(self):
        childs_num = np.array(self.childs_num)
        childs_num = childs_num[:self.type_num]
        self.node_num = np.sum(childs_num)
        mask = torch.zeros((self.type_num, self.node_num))
        start = 0
        end = 0
        for i in range(self.type_num):
            end = end + childs_num[i]
            mask[i, start:end] = 1
            start = start + childs_num[i]
        self.mask = mask

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def get_graph_feature(self, x, type_num):
        """
        :param x:
        :param prev_x:
        :param k:
        :param idx:
        :param r: output downsampling factor (-1 for no downsampling)
        :param style: method to get graph feature
        :return:
        """
        batch_size = x.size(0)
        chinnals = x.size(2)  # if prev_x is None else prev_x.size(2)
        x = x.view(batch_size, -1, chinnals)
        node_x = x[:, 1:, :].contiguous()
        node_x = node_x.view(batch_size, -1, chinnals * type_num).contiguous()
        source_x = x[:, :node_x.size()[1], :]
        # print(idx_knn.shape)
        feature = torch.cat((node_x, source_x), dim=2)
        feature = feature.view(batch_size, -1, type_num + 1, chinnals).contiguous()
        return feature

    def get_weigth_fuction(self, x):
        batch_size, length, adj_num, chinnal = x.size()
        source_feat=torch.mean(x,dim=2)
        adj_feat = x[:, :, :-1, :]
        source_feat = source_feat.view(batch_size * length, 1, chinnal)
        adj_feat = adj_feat.view(batch_size * length, adj_num - 1, chinnal)
        atten = self.atten(source_feat, adj_feat)
        return atten

    def agg_node_fuction(self, x, atten):
        batch_size, length, adj_num, chinnal = x.size()
        atten = atten.permute(0, 2, 1).contiguous()
        atten = atten.view(batch_size, length, adj_num - 1, self.multi_head)
        node_feat = x[:, :, :-1, :]
        source_feat = x[:, :, -1, :]
        source_feat = source_feat.unsqueeze(2)
        node_feat = node_feat.view(batch_size, length, adj_num - 1, self.multi_head, -1)
        node_feat = node_feat * atten[:, :, :, :, None]
        node_feat = node_feat.view(batch_size * length, adj_num - 1, chinnal)
        mask = self.mask.repeat(batch_size * length, 1, 1)
        mask = mask.cuda(node_feat.device)
        node_feat = torch.bmm(mask, node_feat)
        node_feat = node_feat.view(batch_size, length, self.type_num, chinnal)
        return node_feat, source_feat

    def forward(self, input, gt):
        gt_batch, gt_adj, gt_chinnal = gt.size()
        gt = gt.view(-1, gt_chinnal)
        gt = self.gt(gt)
        gt = gt.view(gt_batch, gt_adj, -1)
        x = self.get_graph_feature(input, self.node_num)  # batch_size,  chinnal,length = input.size()
        atten = self.get_weigth_fuction(x)
        out_node_feat, out_source_feat = self.agg_node_fuction(x, atten)
        batch_size, length, adj_num, chinnal = out_node_feat.size()
        out_node_feat = out_node_feat.view(batch_size * length, adj_num, chinnal).contiguous()
        out_source_feat = out_node_feat.view(batch_size * length, 1, chinnal).contiguous()
        out_node_feat = out_node_feat.permute(0, 2, 1)
        out_source_feat = out_source_feat.permute(0, 2, 1)
        output_node_feat = self.agg(out_node_feat)
        node_feat = out_node_feat.view(batch_size, length, -1)[:, 0, :]
        gt = gt[:, 0, :].contiguous()
        scorce = feature_measure(gt=gt, node_feat=node_feat)
        out_source_feat = self.source(out_source_feat)
        output = output_node_feat + out_source_feat
        output = output.view(batch_size, length, -1).contiguous()
        return output, scorce


class Graph_module_2_loss(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, type_num, childs_num, multi_head, dropout=0.1, group=1, bias=True):
        super(Graph_module_2_loss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.type_num = type_num
        self.multi_head = multi_head
        self.childs_num = childs_num
        self.atten = MultiHeadedAttention(h=self.multi_head, d_model=in_features, dropout=dropout)
        self.agg = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=out_features,
                                           kernel_size=self.type_num, stride=self.type_num, groups=group, bias=bias),
                                 nn.ReLU(inplace=True))
        self.gt = nn.Sequential(nn.Linear(in_features, out_features),
                                nn.ReLU(inplace=True))
        self.source = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=out_features,
                                              kernel_size=1, stride=1, groups=group,
                                              bias=bias),
                                    nn.ReLU(inplace=True))
        self.get_adj_mm()

    def get_adj_mm(self):
        childs_num = np.array(self.childs_num)
        childs_num = childs_num[:self.type_num]
        self.node_num = np.sum(childs_num)
        mask = torch.zeros((self.type_num, self.node_num))
        start = 0
        end = 0
        for i in range(self.type_num):
            end = end + childs_num[i]
            mask[i, start:end] = 1
            start = start + childs_num[i]
        self.mask = mask

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def get_graph_feature(self, x, type_num):
        """
        :param x:
        :param prev_x:
        :param k:
        :param idx:
        :param r: output downsampling factor (-1 for no downsampling)
        :param style: method to get graph feature
        :return:
        """
        batch_size = x.size(0)
        chinnals = x.size(2)  # if prev_x is None else prev_x.size(2)
        x = x.view(batch_size, -1, chinnals)
        node_x = x[:, 1:, :].contiguous()
        node_x = node_x.view(batch_size, -1, chinnals * type_num).contiguous()
        source_x = x[:, :node_x.size()[1], :]
        # print(idx_knn.shape)
        feature = torch.cat((node_x, source_x), dim=2)
        feature = feature.view(batch_size, -1, type_num + 1, chinnals).contiguous()
        return feature

    def get_weigth_fuction(self, x):
        batch_size, length, adj_num, chinnal = x.size()
        source_feat = torch.max(x, dim=2)[0]
        adj_feat = x[:, :, :-1, :]
        source_feat = source_feat.view(batch_size * length, 1, chinnal)
        adj_feat = adj_feat.view(batch_size * length, adj_num - 1, chinnal)
        atten = self.atten(source_feat, adj_feat)
        return atten

    def agg_node_fuction(self, x, atten):
        batch_size, length, adj_num, chinnal = x.size()
        atten = atten.permute(0, 2, 1).contiguous()
        atten = atten.view(batch_size, length, adj_num - 1, self.multi_head)
        node_feat = x[:, :, :-1, :]
        source_feat = x[:, :, -1, :]
        source_feat = source_feat.unsqueeze(2)
        node_feat = node_feat.view(batch_size, length, adj_num - 1, self.multi_head, -1)
        node_feat = node_feat * atten[:, :, :, :, None]
        node_feat = node_feat.view(batch_size * length, adj_num - 1, chinnal)
        mask = self.mask.repeat(batch_size * length, 1, 1)
        mask = mask.cuda(node_feat.device)
        node_feat = torch.bmm(mask, node_feat)
        node_feat = node_feat.view(batch_size, length, self.type_num, chinnal)
        return node_feat, source_feat

    def forward(self, input, gt):
        gt_batch, gt_adj, gt_chinnal = gt.size()
        gt = gt.view(-1, gt_chinnal)
        gt = self.gt(gt)
        gt = gt.view(gt_batch, gt_adj, -1)
        x = self.get_graph_feature(input, self.node_num)  # batch_size,  chinnal,length = input.size()
        atten = self.get_weigth_fuction(x)
        out_node_feat, out_source_feat = self.agg_node_fuction(x, atten)
        batch_size, length, adj_num, chinnal = out_node_feat.size()
        out_node_feat = out_node_feat.view(batch_size * length, adj_num, chinnal).contiguous()
        out_source_feat = out_node_feat.view(batch_size * length, 1, chinnal).contiguous()
        out_node_feat = out_node_feat.permute(0, 2, 1)
        out_source_feat = out_source_feat.permute(0, 2, 1)
        output_node_feat = self.agg(out_node_feat)
        node_feat = out_node_feat.view(batch_size, length, -1)[:, 0, :]
        gt = gt[:, 0, :].contiguous()
        scorce = feature_measure(gt=gt, node_feat=node_feat)
        out_source_feat = self.source(out_source_feat)
        output = output_node_feat + out_source_feat
        output = output.view(batch_size, length, -1).contiguous()
        return output, scorce

class Graph_module_3_loss(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, type_num, childs_num, multi_head, dropout=0.1, group=1, bias=True):
        super(Graph_module_3_loss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.type_num = type_num
        self.multi_head = multi_head
        self.childs_num = childs_num
        self.atten_linear = nn.Linear(in_features * 3, in_features)
        self.atten = MultiHeadedAttention(h=self.multi_head, d_model=in_features, dropout=dropout)
        self.agg = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=out_features,
                                           kernel_size=self.type_num, stride=self.type_num, groups=group, bias=bias),
                                 nn.ReLU(inplace=True))
        self.gt = nn.Sequential(nn.Linear(in_features, out_features),
                                nn.ReLU(inplace=True))
        self.source = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=out_features,
                                              kernel_size=1, stride=1, groups=group,
                                              bias=bias),
                                    nn.ReLU(inplace=True))
        self.get_adj_mm()

    def get_adj_mm(self):
        childs_num = np.array(self.childs_num)
        childs_num = childs_num[:self.type_num]
        self.node_num = np.sum(childs_num)
        mask = torch.zeros((self.type_num, self.node_num))
        start = 0
        end = 0
        for i in range(self.type_num):
            end = end + childs_num[i]
            mask[i, start:end] = 1
            start = start + childs_num[i]
        self.mask = mask

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def get_graph_feature(self, x, type_num):
        """
        :param x:
        :param prev_x:
        :param k:
        :param idx:
        :param r: output downsampling factor (-1 for no downsampling)
        :param style: method to get graph feature
        :return:
        """
        batch_size = x.size(0)
        chinnals = x.size(2)  # if prev_x is None else prev_x.size(2)
        x = x.view(batch_size, -1, chinnals)
        node_x = x[:, 1:, :].contiguous()
        node_x = node_x.view(batch_size, -1, chinnals * type_num).contiguous()
        source_x = x[:, :node_x.size()[1], :]
        # print(idx_knn.shape)
        feature = torch.cat((node_x, source_x), dim=2)
        feature = feature.view(batch_size, -1, type_num + 1, chinnals).contiguous()
        return feature

    def get_weigth_fuction(self, x):
        batch_size, length, adj_num, chinnal = x.size()
        adj_feat = x[:, :, :-1, :]
        source_feat_1 = torch.max(adj_feat, dim=2)[0]
        source_feat_2 = torch.mean(adj_feat, dim=2)
        source_feat_3 = x[:, :, -1, :]
        source_feat = torch.cat((source_feat_1, source_feat_2, source_feat_3), dim=2)
        source_feat = source_feat.view(-1, chinnal * 3)
        source_feat = self.atten_linear(source_feat)
        source_feat = source_feat.view(batch_size * length, 1, chinnal)
        adj_feat = adj_feat.view(batch_size * length, adj_num - 1, chinnal)
        atten = self.atten(source_feat, adj_feat)
        return atten

    def agg_node_fuction(self, x, atten):
        batch_size, length, adj_num, chinnal = x.size()
        atten = atten.permute(0, 2, 1).contiguous()
        atten = atten.view(batch_size, length, adj_num - 1, self.multi_head)
        node_feat = x[:, :, :-1, :]
        source_feat = x[:, :, -1, :]
        source_feat = source_feat.unsqueeze(2)
        node_feat = node_feat.view(batch_size, length, adj_num - 1, self.multi_head, -1)
        node_feat = node_feat * atten[:, :, :, :, None]
        node_feat = node_feat.view(batch_size * length, adj_num - 1, chinnal)
        mask = self.mask.repeat(batch_size * length, 1, 1)
        mask = mask.cuda(node_feat.device)
        node_feat = torch.bmm(mask, node_feat)
        node_feat = node_feat.view(batch_size, length, self.type_num, chinnal)
        return node_feat, source_feat

    def forward(self, input, gt):
        gt_batch, gt_adj, gt_chinnal = gt.size()
        gt = gt.view(-1, gt_chinnal)
        gt = self.gt(gt)
        gt = gt.view(gt_batch, gt_adj, -1)
        x = self.get_graph_feature(input, self.node_num)  # batch_size,  chinnal,length = input.size()
        atten = self.get_weigth_fuction(x)
        out_node_feat, out_source_feat = self.agg_node_fuction(x, atten)
        batch_size, length, adj_num, chinnal = out_node_feat.size()
        out_node_feat = out_node_feat.view(batch_size * length, adj_num, chinnal).contiguous()
        out_source_feat = out_node_feat.view(batch_size * length, 1, chinnal).contiguous()
        out_node_feat = out_node_feat.permute(0, 2, 1)
        out_source_feat = out_source_feat.permute(0, 2, 1)
        output_node_feat = self.agg(out_node_feat)
        node_feat = out_node_feat.view(batch_size, length, -1)[:, 0, :]
        gt = gt[:, 0, :].contiguous()
        scorce = feature_measure(gt=gt, node_feat=node_feat)
        out_source_feat = self.source(out_source_feat)
        output = output_node_feat + out_source_feat
        output = output.view(batch_size, length, -1).contiguous()
        return output, scorce

class Graph_module_net_loss(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features,mid_features, out_features,type_num,childs_num,multi_head=1,dropout=0.1, atten_dropout=0.1,group=4,bias=True):
        super(Graph_module_net_loss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.type_num=type_num
        self.multi_head=multi_head
        self.childs_num=childs_num
        self.dropout = dropout
        self.atten=MultiHeadedAttention_4(h=self.multi_head,d_model=in_features,dropout=atten_dropout)
        self.agg1 = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=mid_features,
                                            kernel_size=self.type_num, stride=self.type_num, groups=group,
                                            bias=bias),
                                  nn.ReLU(inplace=True))
        self.gt1=nn.Sequential(nn.Linear(in_features, mid_features),
                               nn.ReLU(inplace=True))
        self.source1 = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=mid_features,
                                            kernel_size=1, stride=1, groups=group,
                                            bias=bias),
                                  nn.ReLU(inplace=True))
        self.agg2 = nn.Sequential(nn.Conv1d(in_channels=mid_features,out_channels=out_features,
                                           kernel_size=self.type_num,stride=self.type_num,groups=group,bias=bias),
                                 nn.ReLU(inplace=True))
        self.source2 = nn.Sequential(nn.Conv1d(in_channels=mid_features, out_channels=out_features,
                                            kernel_size=1, stride=1, groups=group, bias=bias),
                                  nn.ReLU(inplace=True))
        self.gt2 = nn.Sequential(nn.Linear(mid_features,out_features),
                                 nn.ReLU(inplace=True))
        self.get_adj_mm()
    def get_adj_mm(self):
        childs_num=np.array(self.childs_num)
        childs_num=childs_num[:self.type_num]
        self.node_num=np.sum(childs_num)
        mask=torch.zeros((self.type_num,self.node_num))
        start=0
        end=0
        for i in range(self.type_num):
            end=end+childs_num[i]
            mask[i,start:end]=1
            start=start+childs_num[i]
        self.mask=mask
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    def get_graph_feature(self,x, type_num):
        """
        :param x:
        :param prev_x:
        :param k:
        :param idx:
        :param r: output downsampling factor (-1 for no downsampling)
        :param style: method to get graph feature
        :return:
        """
        batch_size = x.size(0)
        chinnals = x.size(2)  # if prev_x is None else prev_x.size(2)
        x = x.view(batch_size,-1, chinnals)
        node_x = x[:, 1:,:].contiguous()
        node_x = node_x.view(batch_size, -1, chinnals * type_num).contiguous()
        source_x=x[:,:node_x.size()[1],:]
        # print(idx_knn.shape)
        feature = torch.cat((node_x, source_x), dim=2)
        feature=feature.view(batch_size, -1, type_num + 1, chinnals).contiguous()
        return feature
    def get_weigth_fuction(self,x):
        batch_size, length, adj_num, chinnal = x.size()
        source_feat=x[:,:,-1,:]
        adj_feat=x[:,:,:-1,:]
        source_feat=source_feat.view(batch_size*length,1,chinnal)
        adj_feat=adj_feat.view(batch_size*length,adj_num-1,chinnal)
        atten=self.atten(source_feat,adj_feat)
        return atten
    def agg_node_fuction(self,x,atten):
        batch_size, length, adj_num, chinnal = x.size()
        atten=atten.permute(0,2,1).contiguous()
        atten=atten.view(batch_size,length,adj_num-1,self.multi_head)
        node_feat=x[:,:,:-1,:]
        source_feat = x[:, :, -1, :]
        source_feat=source_feat.unsqueeze(2)
        node_feat=node_feat.view(batch_size,length,adj_num-1,self.multi_head,-1)
        node_feat=node_feat*atten[:,:,:,:,None]
        node_feat=node_feat.view(batch_size*length,adj_num-1,chinnal)
        mask=self.mask.repeat(batch_size*length,1,1)
        mask=mask.cuda(node_feat.device)
        node_feat=torch.bmm(mask,node_feat)
        node_feat = node_feat.view(batch_size, length, self.type_num, chinnal)
        return node_feat,source_feat

    def forward(self, input,gt):
        gt_batch,gt_adj,gt_chinnal=gt.size()
        gt=gt.view(-1,gt_chinnal)
        gt1=self.gt1(gt)
        gt1_drop=F.dropout(gt1, self.dropout, training=self.training)
        gt2=self.gt2(gt1_drop)
        gt1=gt1.view(gt_batch,gt_adj,-1)
        gt2=gt2.view(gt_batch,gt_adj,-1)
        x=self.get_graph_feature(input,self.node_num) #batch_size,length, chinnal = input.size()
        atten=self.get_weigth_fuction(x)
        out_node_feat,out_source_feat=self.agg_node_fuction(x,atten)
        batch_size,length, adj_num,chinnal  = out_node_feat.size()
        out_node_feat=out_node_feat.view(batch_size*length,adj_num,chinnal).contiguous()
        out_source_feat=out_source_feat.view(batch_size*length,1,chinnal).contiguous()
        out_node_feat=out_node_feat.permute(0,2,1)
        out_source_feat=out_source_feat.permute(0,2,1)
        out_source_feat=self.source1(out_source_feat).squeeze(2)
        out_node_feat=self.agg1(out_node_feat).squeeze(2)
        node_feat=out_node_feat.view(batch_size,length,-1)[:,0,:]
        gt1=gt1[:,0,:].contiguous()
        scorce1=feature_measure(gt=gt1,node_feat=node_feat)
        output1=out_node_feat+out_source_feat
        output1 = F.dropout(output1, self.dropout, training=self.training)
        output1=output1.view(batch_size,length,-1)
        output2=self.get_graph_feature(output1,self.node_num)
        atten2=atten.view(batch_size,-1,atten.size(1),atten.size(2))
        atten2=atten2[:,:output2.size()[1],:,:]
        atten2=atten2.view(-1,atten.size(1),atten.size(2))
        out_node_feat2,out_source_feat2=self.agg_node_fuction(output2,atten2)
        batch_size2, length2, adj_num2, chinnal2 = out_node_feat2.size()
        out_node_feat2 = out_node_feat2.view(batch_size2 * length2, adj_num2, chinnal2).contiguous()
        out_source_feat2=out_source_feat2.view(batch_size2*length2,1,chinnal2)
        out_node_feat2 = out_node_feat2.permute(0, 2, 1)
        out_source_feat2=out_source_feat2.permute(0,2,1)
        out_node_feat2 = self.agg2(out_node_feat2)
        node_feat2 = out_node_feat2.view(batch_size2, length2, -1)
        gt2 = gt2[:, :node_feat2.size()[1], :]
        scorce2 = feature_measure(gt=gt2, node_feat=node_feat2)
        out_source_feat2=self.source2(out_source_feat2)
        output=out_node_feat2+out_source_feat2
        output=output.view(-1,self.out_features).contiguous()
        return output,scorce1,scorce2

class Graph_module_net_0_loss(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features,mid_features, out_features,type_num,childs_num,multi_head=8,dropout=0.1, atten_dropout=0.1,group=4,bias=True):
        super(Graph_module_net_0_loss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.type_num=type_num
        self.multi_head=multi_head
        self.childs_num=childs_num
        self.dropout = dropout
        self.atten=MultiHeadedAttention_2(h=self.multi_head,d_model=in_features,dropout=atten_dropout)
        self.agg1 = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=mid_features,
                                            kernel_size= 1, stride=1, groups=group,
                                            bias=bias),
                                  nn.ReLU(inplace=True))
        self.agg2 = nn.Sequential(nn.Conv1d(in_channels=mid_features,out_channels=out_features,
                                           kernel_size=1,stride=1,groups=group,bias=bias),
                                 nn.ReLU(inplace=True))
        self.gt1 = nn.Sequential(nn.Linear(in_features, mid_features),
                                 nn.ReLU(inplace=True))
        self.gt2 = nn.Sequential(nn.Linear(mid_features, out_features),
                                 nn.ReLU(inplace=True))
        self.get_adj_mm()
    def get_adj_mm(self):
        childs_num=np.array(self.childs_num)
        childs_num=childs_num[:self.type_num]
        self.node_num=np.sum(childs_num)
        mask=torch.zeros((self.type_num,self.node_num))
        start=0
        end=0
        for i in range(self.type_num):
            end=end+childs_num[i]
            mask[i,start:end]=1
            start=start+childs_num[i]
        self.mask=mask
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    def get_graph_feature(self,x, type_num):
        """
        :param x:
        :param prev_x:
        :param k:
        :param idx:
        :param r: output downsampling factor (-1 for no downsampling)
        :param style: method to get graph feature
        :return:
        """
        batch_size = x.size(0)
        chinnals = x.size(2)  # if prev_x is None else prev_x.size(2)
        x = x.view(batch_size,-1, chinnals)
        node_x = x[:, 1:,:].contiguous()
        node_x = node_x.view(batch_size, -1, chinnals * type_num).contiguous()
        source_x=x[:,:node_x.size()[1],:]
        # print(idx_knn.shape)
        feature = torch.cat((node_x, source_x), dim=2)
        feature=feature.view(batch_size, -1, type_num + 1, chinnals).contiguous()
        return feature
    def get_weigth_fuction(self,x):
        batch_size, length, adj_num, chinnal = x.size()
        source_feat=torch.mean(x,dim=2)
        adj_feat=x[:,:,:-1,:]
        source_feat=source_feat.view(batch_size*length,1,chinnal)
        adj_feat=adj_feat.view(batch_size*length,adj_num-1,chinnal)
        atten=self.atten(source_feat,adj_feat)
        return atten
    def agg_node_fuction(self,x,atten):
        batch_size, length, adj_num, chinnal = x.size()
        atten=atten.permute(0,2,1).contiguous()
        atten=atten.view(batch_size,length,adj_num-1,self.multi_head)
        node_feat=x[:,:,:-1,:]
        source_feat = x[:, :, -1, :]
        source_feat=source_feat.unsqueeze(2)
        node_feat=node_feat.view(batch_size,length,adj_num-1,self.multi_head,-1)
        node_feat=node_feat*atten[:,:,:,:,None]
        node_feat=node_feat.view(batch_size*length,adj_num-1,chinnal)
        node_feat=node_feat.sum(dim=1)
        node_feat = node_feat.view(batch_size, length, 1, chinnal)
        return node_feat,source_feat
    def forward(self, input,gt):
        gt_batch, gt_adj, gt_chinnal = gt.size()
        gt = gt.view(-1, gt_chinnal)
        gt1 = self.gt1(gt)
        gt1_drop = F.dropout(gt1, self.dropout, training=self.training)
        gt2 = self.gt2(gt1_drop)
        gt1 = gt1.view(gt_batch, gt_adj, -1)
        gt2 = gt2.view(gt_batch, gt_adj, -1)
        x = self.get_graph_feature(input, self.node_num)  # batch_size,length, chinnal = input.size()
        atten = self.get_weigth_fuction(x)
        out_node_feat, out_source_feat = self.agg_node_fuction(x, atten)
        batch_size, length, adj_num, chinnal = out_node_feat.size()
        out_node_feat = out_node_feat.view(batch_size * length, adj_num, chinnal).contiguous()
        out_source_feat = out_source_feat.view(batch_size * length, 1, chinnal).contiguous()
        out_node_feat = out_node_feat.permute(0, 2, 1)
        out_source_feat = out_source_feat.permute(0, 2, 1)
        out_source_feat = self.agg1(out_source_feat).squeeze(2)
        out_node_feat = self.agg1(out_node_feat).squeeze(2)
        node_feat = out_node_feat.view(batch_size, length, -1)[:, 0, :]
        gt1 = gt1[:, 0, :].contiguous()
        scorce1 = feature_measure(gt=gt1, node_feat=node_feat)
        output1 = out_node_feat + out_source_feat
        output1 = F.dropout(output1, self.dropout, training=self.training)
        output1 = output1.view(batch_size, length, -1)
        output2 = self.get_graph_feature(output1, self.node_num)
        atten2 = atten.view(batch_size, -1, atten.size(1), atten.size(2))
        atten2 = atten2[:, :output2.size()[1], :, :]
        atten2 = atten2.view(-1, atten.size(1), atten.size(2))
        out_node_feat2, out_source_feat2 = self.agg_node_fuction(output2, atten2)
        batch_size2, length2, adj_num2, chinnal2 = out_node_feat2.size()
        out_node_feat2 = out_node_feat2.view(batch_size2 * length2, adj_num2, chinnal2).contiguous()
        out_source_feat2 = out_source_feat2.view(batch_size2 * length2, 1, chinnal2)
        out_node_feat2 = out_node_feat2.permute(0, 2, 1)
        out_source_feat2 = out_source_feat2.permute(0, 2, 1)
        out_node_feat2 = self.agg2(out_node_feat2)
        node_feat2 = out_node_feat2.view(batch_size2, length2, -1)
        gt2 = gt2[:, :node_feat2.size()[1], :]
        scorce2 = feature_measure(gt=gt2, node_feat=node_feat2)
        out_source_feat2 = self.agg2(out_source_feat2)
        output = out_node_feat2 + out_source_feat2
        output = output.view(-1, self.out_features).contiguous()
        return output,scorce1,scorce2


class Graph_module_net_0_loss_three_layer(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features,mid_features, out_features,type_num,childs_num,multi_head=8,dropout=0.1, atten_dropout=0.1,group=4,bias=True):
        super(Graph_module_net_0_loss_three_layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.type_num=type_num
        self.multi_head=multi_head
        self.childs_num=childs_num
        self.dropout = dropout
        self.atten=MultiHeadedAttention_2(h=self.multi_head,d_model=in_features,dropout=atten_dropout)
        self.agg1 = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=in_features,
                                            kernel_size= 1, stride=1, groups=group,
                                            bias=bias),
                                  nn.ReLU(inplace=True))
        self.agg2 = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=mid_features,
                                            kernel_size=1, stride=1, groups=group,
                                            bias=bias),
                                  nn.ReLU(inplace=True))
        self.agg3 = nn.Sequential(nn.Conv1d(in_channels=mid_features,out_channels=out_features,
                                           kernel_size=1,stride=1,groups=group,bias=bias),
                                 nn.ReLU(inplace=True))
        self.gt1 = nn.Sequential(nn.Linear(in_features, mid_features),
                                 nn.ReLU(inplace=True))
        self.gt2 = nn.Sequential(nn.Linear(mid_features, out_features),
                                 nn.ReLU(inplace=True))
        self.get_adj_mm()
    def get_adj_mm(self):
        childs_num=np.array(self.childs_num)
        childs_num=childs_num[:self.type_num]
        self.node_num=np.sum(childs_num)
        mask=torch.zeros((self.type_num,self.node_num))
        start=0
        end=0
        for i in range(self.type_num):
            end=end+childs_num[i]
            mask[i,start:end]=1
            start=start+childs_num[i]
        self.mask=mask
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    def get_graph_feature(self,x, type_num):
        """
        :param x:
        :param prev_x:
        :param k:
        :param idx:
        :param r: output downsampling factor (-1 for no downsampling)
        :param style: method to get graph feature
        :return:
        """
        batch_size = x.size(0)
        chinnals = x.size(2)  # if prev_x is None else prev_x.size(2)
        x = x.view(batch_size,-1, chinnals)
        node_x = x[:, 1:,:].contiguous()
        node_x = node_x.view(batch_size, -1, chinnals * type_num).contiguous()
        source_x=x[:,:node_x.size()[1],:]
        # print(idx_knn.shape)
        feature = torch.cat((node_x, source_x), dim=2)
        feature=feature.view(batch_size, -1, type_num + 1, chinnals).contiguous()
        return feature
    def get_weigth_fuction(self,x):
        batch_size, length, adj_num, chinnal = x.size()
        source_feat=torch.mean(x,dim=2)
        adj_feat=x[:,:,:-1,:]
        source_feat=source_feat.view(batch_size*length,1,chinnal)
        adj_feat=adj_feat.view(batch_size*length,adj_num-1,chinnal)
        atten=self.atten(source_feat,adj_feat)
        return atten
    def agg_node_fuction(self,x,atten):
        batch_size, length, adj_num, chinnal = x.size()
        atten=atten.permute(0,2,1).contiguous()
        atten=atten.view(batch_size,length,adj_num-1,self.multi_head)
        node_feat=x[:,:,:-1,:]
        source_feat = x[:, :, -1, :]
        source_feat=source_feat.unsqueeze(2)
        node_feat=node_feat.view(batch_size,length,adj_num-1,self.multi_head,-1)
        node_feat=node_feat*atten[:,:,:,:,None]
        node_feat=node_feat.view(batch_size*length,adj_num-1,chinnal)
        node_feat=node_feat.sum(dim=1)
        node_feat = node_feat.view(batch_size, length, 1, chinnal)
        return node_feat,source_feat
    def forward(self, input,gt):
        gt_batch, gt_adj, gt_chinnal = gt.size()
        gt = gt.view(-1, gt_chinnal)
        gt1 = self.gt1(gt)
        gt1_drop = F.dropout(gt1, self.dropout, training=self.training)
        gt2 = self.gt2(gt1_drop)
        gt1 = gt1.view(gt_batch, gt_adj, -1).contiguous()
        gt2 = gt2.view(gt_batch, gt_adj, -1).contiguous()
        x = self.get_graph_feature(input, self.node_num)  # batch_size,length, chinnal = input.size()
        atten = self.get_weigth_fuction(x)
        out_node_feat, out_source_feat = self.agg_node_fuction(x, atten)
        batch_size, length, adj_num, chinnal = out_node_feat.size()
        out_node_feat = out_node_feat.view(batch_size * length, adj_num, chinnal).contiguous()
        out_source_feat = out_source_feat.view(batch_size * length, 1, chinnal).contiguous()
        out_node_feat = out_node_feat.permute(0, 2, 1)
        out_source_feat = out_source_feat.permute(0, 2, 1)
        out_source_feat = self.agg1(out_source_feat).squeeze(2)
        out_node_feat = self.agg1(out_node_feat).squeeze(2)
        output1 = out_node_feat + out_source_feat
        output1 = F.dropout(output1, self.dropout, training=self.training)
        output1 = output1.view(batch_size, length, -1)
        output2 = self.get_graph_feature(output1, self.node_num)
        atten2 = atten.view(batch_size, -1, atten.size(1), atten.size(2))
        atten2 = atten2[:, :output2.size()[1], :, :].contiguous()
        atten2 = atten2.view(-1, atten.size(1), atten.size(2))
        out_node_feat2, out_source_feat2 = self.agg_node_fuction(output2, atten2)
        batch_size2, length2, adj_num2, chinnal2 = out_node_feat2.size()
        out_node_feat2 = out_node_feat2.view(batch_size2 * length2, adj_num2, chinnal2).contiguous()
        out_source_feat2 = out_source_feat2.view(batch_size2 * length2, 1, chinnal2)
        out_node_feat2 = out_node_feat2.permute(0, 2, 1)
        out_source_feat2 = out_source_feat2.permute(0, 2, 1)
        out_node_feat2 = self.agg2(out_node_feat2)
        node_feat2 = out_node_feat2.view(batch_size2, length2, -1).contiguous()[:,0,:]
        gt1 = gt1[:, 0, :].contiguous()
        scorce1 = feature_measure(gt=gt1, node_feat=node_feat2)
        out_source_feat2 = self.agg2(out_source_feat2)
        final_output2 = out_node_feat2 + out_source_feat2
        #########################################################################################
        final_output2 = F.dropout(final_output2, self.dropout, training=self.training)
        final_output2 = final_output2.view(batch_size2, length2, -1)
        output3 = self.get_graph_feature(final_output2, self.node_num)
        atten3 = atten.view(batch_size, -1, atten.size(1), atten.size(2))
        atten3 = atten3[:, :output3.size()[1], :, :]
        atten3 = atten3.view(-1, atten.size(1), atten.size(2))
        out_node_feat3, out_source_feat3 = self.agg_node_fuction(output3, atten3)
        batch_size3, length3, adj_num3, chinnal3 = out_node_feat3.size()
        out_node_feat3 = out_node_feat3.view(batch_size3 * length3, adj_num3, chinnal3).contiguous()
        out_source_feat3 = out_source_feat3.view(batch_size3 * length3, 1, chinnal3)
        out_node_feat3 = out_node_feat3.permute(0, 2, 1)
        out_source_feat3 = out_source_feat3.permute(0, 2, 1)
        out_node_feat3 = self.agg3(out_node_feat3)
        node_feat3 = out_node_feat3.view(batch_size3, length3, -1).contiguous()
        gt2 = gt2[:, :node_feat3.size()[1], :].contiguous()
        scorce2 = feature_measure(gt=gt2, node_feat=node_feat3)
        out_source_feat3 = self.agg3(out_source_feat3)
        output = out_node_feat3 + out_source_feat3
        output = output.view(-1, self.out_features).contiguous()
        return output,scorce1,scorce2


class Graph_module_net_1_loss(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features,mid_features, out_features,type_num,childs_num,multi_head=1,dropout=0.1, atten_dropout=0.1,group=4,bias=True):
        super(Graph_module_net_1_loss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.type_num=type_num
        self.multi_head=multi_head
        self.childs_num=childs_num
        self.dropout = dropout
        self.atten=MultiHeadedAttention_4(h=self.multi_head,d_model=in_features,dropout=atten_dropout)
        self.agg1 = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=mid_features,
                                            kernel_size=self.type_num, stride=self.type_num, groups=group,
                                            bias=bias),
                                  nn.ReLU(inplace=True))
        self.gt1=nn.Sequential(nn.Linear(in_features, mid_features),
                               nn.ReLU(inplace=True))
        self.source1 = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=mid_features,
                                            kernel_size=1, stride=1, groups=group,
                                            bias=bias),
                                  nn.ReLU(inplace=True))
        self.agg2 = nn.Sequential(nn.Conv1d(in_channels=mid_features,out_channels=out_features,
                                           kernel_size=self.type_num,stride=self.type_num,groups=group,bias=bias),
                                 nn.ReLU(inplace=True))
        self.source2 = nn.Sequential(nn.Conv1d(in_channels=mid_features, out_channels=out_features,
                                            kernel_size=1, stride=1, groups=group, bias=bias),
                                  nn.ReLU(inplace=True))
        self.gt2 = nn.Sequential(nn.Linear(mid_features,out_features),
                                 nn.ReLU(inplace=True))
        self.get_adj_mm()
    def get_adj_mm(self):
        childs_num=np.array(self.childs_num)
        childs_num=childs_num[:self.type_num]
        self.node_num=np.sum(childs_num)
        mask=torch.zeros((self.type_num,self.node_num))
        start=0
        end=0
        for i in range(self.type_num):
            end=end+childs_num[i]
            mask[i,start:end]=1
            start=start+childs_num[i]
        self.mask=mask
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    def get_graph_feature(self,x, type_num):
        """
        :param x:
        :param prev_x:
        :param k:
        :param idx:
        :param r: output downsampling factor (-1 for no downsampling)
        :param style: method to get graph feature
        :return:
        """
        batch_size = x.size(0)
        chinnals = x.size(2)  # if prev_x is None else prev_x.size(2)
        x = x.view(batch_size,-1, chinnals)
        node_x = x[:, 1:,:].contiguous()
        node_x = node_x.view(batch_size, -1, chinnals * type_num).contiguous()
        source_x=x[:,:node_x.size()[1],:]
        # print(idx_knn.shape)
        feature = torch.cat((node_x, source_x), dim=2)
        feature=feature.view(batch_size, -1, type_num + 1, chinnals).contiguous()
        return feature
    def get_weigth_fuction(self,x):
        batch_size, length, adj_num, chinnal = x.size()
        source_feat=torch.mean(x,dim=2)
        adj_feat=x[:,:,:-1,:]
        source_feat=source_feat.view(batch_size*length,1,chinnal)
        adj_feat=adj_feat.view(batch_size*length,adj_num-1,chinnal)
        atten=self.atten(source_feat,adj_feat)
        return atten
    def agg_node_fuction(self,x,atten):
        batch_size, length, adj_num, chinnal = x.size()
        atten=atten.permute(0,2,1).contiguous()
        atten=atten.view(batch_size,length,adj_num-1,self.multi_head)
        node_feat=x[:,:,:-1,:]
        source_feat = x[:, :, -1, :]
        source_feat=source_feat.unsqueeze(2)
        node_feat=node_feat.view(batch_size,length,adj_num-1,self.multi_head,-1)
        node_feat=node_feat*atten[:,:,:,:,None]
        node_feat=node_feat.view(batch_size*length,adj_num-1,chinnal)
        mask=self.mask.repeat(batch_size*length,1,1)
        mask=mask.cuda(node_feat.device)
        node_feat=torch.bmm(mask,node_feat)
        node_feat = node_feat.view(batch_size, length, self.type_num, chinnal)
        return node_feat,source_feat

    def forward(self, input,gt):
        gt_batch,gt_adj,gt_chinnal=gt.size()
        gt=gt.view(-1,gt_chinnal)
        gt1=self.gt1(gt)
        gt1_drop=F.dropout(gt1, self.dropout, training=self.training)
        gt2=self.gt2(gt1_drop)
        gt1=gt1.view(gt_batch,gt_adj,-1)
        gt2=gt2.view(gt_batch,gt_adj,-1)
        x=self.get_graph_feature(input,self.node_num) #batch_size,length, chinnal = input.size()
        atten=self.get_weigth_fuction(x)
        out_node_feat,out_source_feat=self.agg_node_fuction(x,atten)
        batch_size,length, adj_num,chinnal  = out_node_feat.size()
        out_node_feat=out_node_feat.view(batch_size*length,adj_num,chinnal).contiguous()
        out_source_feat=out_source_feat.view(batch_size*length,1,chinnal).contiguous()
        out_node_feat=out_node_feat.permute(0,2,1)
        out_source_feat=out_source_feat.permute(0,2,1)
        out_source_feat=self.source1(out_source_feat).squeeze(2)
        out_node_feat=self.agg1(out_node_feat).squeeze(2)
        node_feat=out_node_feat.view(batch_size,length,-1)[:,0,:]
        gt1=gt1[:,0,:].contiguous()
        scorce1=feature_measure(gt=gt1,node_feat=node_feat)
        output1=out_node_feat+out_source_feat
        output1 = F.dropout(output1, self.dropout, training=self.training)
        output1=output1.view(batch_size,length,-1)
        output2=self.get_graph_feature(output1,self.node_num)
        atten2=atten.view(batch_size,-1,atten.size(1),atten.size(2))
        atten2=atten2[:,:output2.size()[1],:,:]
        atten2=atten2.view(-1,atten.size(1),atten.size(2))
        out_node_feat2,out_source_feat2=self.agg_node_fuction(output2,atten2)
        batch_size2, length2, adj_num2, chinnal2 = out_node_feat2.size()
        out_node_feat2 = out_node_feat2.view(batch_size2 * length2, adj_num2, chinnal2).contiguous()
        out_source_feat2=out_source_feat2.view(batch_size2*length2,1,chinnal2)
        out_node_feat2 = out_node_feat2.permute(0, 2, 1)
        out_source_feat2=out_source_feat2.permute(0,2,1)
        out_node_feat2 = self.agg2(out_node_feat2)
        node_feat2 = out_node_feat2.view(batch_size2, length2, -1)
        gt2 = gt2[:, :node_feat2.size()[1], :]
        scorce2 = feature_measure(gt=gt2, node_feat=node_feat2)
        out_source_feat2=self.source2(out_source_feat2)
        output=out_node_feat2+out_source_feat2
        output=output.view(-1,self.out_features).contiguous()
        return output,scorce1,scorce2

class Graph_module_net_2_loss(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features,mid_features, out_features,type_num,childs_num,multi_head=1,dropout=0.1, atten_dropout=0.1,group=4,bias=True):
        super(Graph_module_net_2_loss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.type_num=type_num
        self.multi_head=multi_head
        self.childs_num=childs_num
        self.dropout = dropout
        self.atten=MultiHeadedAttention_4(h=self.multi_head,d_model=in_features,dropout=atten_dropout)
        self.agg1 = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=mid_features,
                                            kernel_size=self.type_num, stride=self.type_num, groups=group,
                                            bias=bias),
                                  nn.ReLU(inplace=True))
        self.gt1=nn.Sequential(nn.Linear(in_features, mid_features),
                               nn.ReLU(inplace=True))
        self.source1 = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=mid_features,
                                            kernel_size=1, stride=1, groups=group,
                                            bias=bias),
                                  nn.ReLU(inplace=True))
        self.agg2 = nn.Sequential(nn.Conv1d(in_channels=mid_features,out_channels=out_features,
                                           kernel_size=self.type_num,stride=self.type_num,groups=group,bias=bias),
                                 nn.ReLU(inplace=True))
        self.source2 = nn.Sequential(nn.Conv1d(in_channels=mid_features, out_channels=out_features,
                                            kernel_size=1, stride=1, groups=group, bias=bias),
                                  nn.ReLU(inplace=True))
        self.gt2 = nn.Sequential(nn.Linear(mid_features,out_features),
                                 nn.ReLU(inplace=True))
        self.get_adj_mm()
    def get_adj_mm(self):
        childs_num=np.array(self.childs_num)
        childs_num=childs_num[:self.type_num]
        self.node_num=np.sum(childs_num)
        mask=torch.zeros((self.type_num,self.node_num))
        start=0
        end=0
        for i in range(self.type_num):
            end=end+childs_num[i]
            mask[i,start:end]=1
            start=start+childs_num[i]
        self.mask=mask
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    def get_graph_feature(self,x, type_num):
        """
        :param x:
        :param prev_x:
        :param k:
        :param idx:
        :param r: output downsampling factor (-1 for no downsampling)
        :param style: method to get graph feature
        :return:
        """
        batch_size = x.size(0)
        chinnals = x.size(2)  # if prev_x is None else prev_x.size(2)
        x = x.view(batch_size,-1, chinnals)
        node_x = x[:, 1:,:].contiguous()
        node_x = node_x.view(batch_size, -1, chinnals * type_num).contiguous()
        source_x=x[:,:node_x.size()[1],:]
        # print(idx_knn.shape)
        feature = torch.cat((node_x, source_x), dim=2)
        feature=feature.view(batch_size, -1, type_num + 1, chinnals).contiguous()
        return feature
    def get_weigth_fuction(self,x):
        batch_size, length, adj_num, chinnal = x.size()
        source_feat=torch.max(x,dim=2)[0]
        adj_feat=x[:,:,:-1,:]
        source_feat=source_feat.view(batch_size*length,1,chinnal)
        adj_feat=adj_feat.view(batch_size*length,adj_num-1,chinnal)
        atten=self.atten(source_feat,adj_feat)
        return atten
    def agg_node_fuction(self,x,atten):
        batch_size, length, adj_num, chinnal = x.size()
        atten=atten.permute(0,2,1).contiguous()
        atten=atten.view(batch_size,length,adj_num-1,self.multi_head)
        node_feat=x[:,:,:-1,:]
        source_feat = x[:, :, -1, :]
        source_feat=source_feat.unsqueeze(2)
        node_feat=node_feat.view(batch_size,length,adj_num-1,self.multi_head,-1)
        node_feat=node_feat*atten[:,:,:,:,None]
        node_feat=node_feat.view(batch_size*length,adj_num-1,chinnal)
        mask=self.mask.repeat(batch_size*length,1,1)
        mask=mask.cuda(node_feat.device)
        node_feat=torch.bmm(mask,node_feat)
        node_feat = node_feat.view(batch_size, length, self.type_num, chinnal)
        return node_feat,source_feat

    def forward(self, input,gt):
        gt_batch,gt_adj,gt_chinnal=gt.size()
        gt=gt.view(-1,gt_chinnal)
        gt1=self.gt1(gt)
        gt1_drop=F.dropout(gt1, self.dropout, training=self.training)
        gt2=self.gt2(gt1_drop)
        gt1=gt1.view(gt_batch,gt_adj,-1)
        gt2=gt2.view(gt_batch,gt_adj,-1)
        x=self.get_graph_feature(input,self.node_num) #batch_size,length, chinnal = input.size()
        atten=self.get_weigth_fuction(x)
        out_node_feat,out_source_feat=self.agg_node_fuction(x,atten)
        batch_size,length, adj_num,chinnal  = out_node_feat.size()
        out_node_feat=out_node_feat.view(batch_size*length,adj_num,chinnal).contiguous()
        out_source_feat=out_source_feat.view(batch_size*length,1,chinnal).contiguous()
        out_node_feat=out_node_feat.permute(0,2,1)
        out_source_feat=out_source_feat.permute(0,2,1)
        out_source_feat=self.source1(out_source_feat).squeeze(2)
        out_node_feat=self.agg1(out_node_feat).squeeze(2)
        node_feat=out_node_feat.view(batch_size,length,-1)[:,0,:]
        gt1=gt1[:,0,:].contiguous()
        scorce1=feature_measure(gt=gt1,node_feat=node_feat)
        output1=out_node_feat+out_source_feat
        output1 = F.dropout(output1, self.dropout, training=self.training)
        output1=output1.view(batch_size,length,-1)
        output2=self.get_graph_feature(output1,self.node_num)
        atten2=atten.view(batch_size,-1,atten.size(1),atten.size(2))
        atten2=atten2[:,:output2.size()[1],:,:]
        atten2=atten2.view(-1,atten.size(1),atten.size(2))
        out_node_feat2,out_source_feat2=self.agg_node_fuction(output2,atten2)
        batch_size2, length2, adj_num2, chinnal2 = out_node_feat2.size()
        out_node_feat2 = out_node_feat2.view(batch_size2 * length2, adj_num2, chinnal2).contiguous()
        out_source_feat2=out_source_feat2.view(batch_size2*length2,1,chinnal2)
        out_node_feat2 = out_node_feat2.permute(0, 2, 1)
        out_source_feat2=out_source_feat2.permute(0,2,1)
        out_node_feat2 = self.agg2(out_node_feat2)
        node_feat2 = out_node_feat2.view(batch_size2, length2, -1)
        gt2 = gt2[:, :node_feat2.size()[1], :]
        scorce2 = feature_measure(gt=gt2, node_feat=node_feat2)
        out_source_feat2=self.source2(out_source_feat2)
        output=out_node_feat2+out_source_feat2
        output=output.view(-1,self.out_features).contiguous()
        return output,scorce1,scorce2

class Graph_module_net_3_loss(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features,mid_features, out_features,type_num,childs_num,multi_head=1,dropout=0.1, atten_dropout=0.1,group=4,bias=True):
        super(Graph_module_net_3_loss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.type_num=type_num
        self.multi_head=multi_head
        self.childs_num=childs_num
        self.dropout = dropout
        self.atten=MultiHeadedAttention_4(h=self.multi_head,d_model=in_features,dropout=atten_dropout)
        self.agg1 = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=mid_features,
                                            kernel_size=self.type_num, stride=self.type_num, groups=group,
                                            bias=bias),
                                  nn.ReLU(inplace=True))
        self.gt1=nn.Sequential(nn.Linear(in_features, mid_features),
                               nn.ReLU(inplace=True))
        self.source1 = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=mid_features,
                                            kernel_size=1, stride=1, groups=group,
                                            bias=bias),
                                  nn.ReLU(inplace=True))
        self.agg2 = nn.Sequential(nn.Conv1d(in_channels=mid_features,out_channels=out_features,
                                           kernel_size=self.type_num,stride=self.type_num,groups=group,bias=bias),
                                 nn.ReLU(inplace=True))
        self.source2 = nn.Sequential(nn.Conv1d(in_channels=mid_features, out_channels=out_features,
                                            kernel_size=1, stride=1, groups=group, bias=bias),
                                  nn.ReLU(inplace=True))
        self.gt2 = nn.Sequential(nn.Linear(mid_features,out_features),
                                 nn.ReLU(inplace=True))
        self.atten_linear = nn.Linear(in_features * 3, in_features)
        self.get_adj_mm()
    def get_adj_mm(self):
        childs_num=np.array(self.childs_num)
        childs_num=childs_num[:self.type_num]
        self.node_num=np.sum(childs_num)
        mask=torch.zeros((self.type_num,self.node_num))
        start=0
        end=0
        for i in range(self.type_num):
            end=end+childs_num[i]
            mask[i,start:end]=1
            start=start+childs_num[i]
        self.mask=mask
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    def get_graph_feature(self,x, type_num):
        """
        :param x:
        :param prev_x:
        :param k:
        :param idx:
        :param r: output downsampling factor (-1 for no downsampling)
        :param style: method to get graph feature
        :return:
        """
        batch_size = x.size(0)
        chinnals = x.size(2)  # if prev_x is None else prev_x.size(2)
        x = x.view(batch_size,-1, chinnals)
        node_x = x[:, 1:,:].contiguous()
        node_x = node_x.view(batch_size, -1, chinnals * type_num).contiguous()
        source_x=x[:,:node_x.size()[1],:]
        # print(idx_knn.shape)
        feature = torch.cat((node_x, source_x), dim=2)
        feature=feature.view(batch_size, -1, type_num + 1, chinnals).contiguous()
        return feature
    def get_weigth_fuction(self,x):
        batch_size, length, adj_num, chinnal = x.size()
        adj_feat=x[:,:,:-1,:]
        source_feat_1 = torch.max(adj_feat, dim=2)[0]
        source_feat_2 = torch.mean(adj_feat, dim=2)
        source_feat_3 = x[:, :, -1, :]
        source_feat = torch.cat((source_feat_1, source_feat_2, source_feat_3), dim=2)
        source_feat = source_feat.view(-1, chinnal * 3)
        source_feat = self.atten_linear(source_feat)
        source_feat=source_feat.view(batch_size*length,1,chinnal)
        adj_feat=adj_feat.view(batch_size*length,adj_num-1,chinnal)
        atten=self.atten(source_feat,adj_feat)
        return atten
    def agg_node_fuction(self,x,atten):
        batch_size, length, adj_num, chinnal = x.size()
        atten=atten.permute(0,2,1).contiguous()
        atten=atten.view(batch_size,length,adj_num-1,self.multi_head)
        node_feat=x[:,:,:-1,:]
        source_feat = x[:, :, -1, :]
        source_feat=source_feat.unsqueeze(2)
        node_feat=node_feat.view(batch_size,length,adj_num-1,self.multi_head,-1)
        node_feat=node_feat*atten[:,:,:,:,None]
        node_feat=node_feat.view(batch_size*length,adj_num-1,chinnal)
        mask=self.mask.repeat(batch_size*length,1,1)
        mask=mask.cuda(node_feat.device)
        node_feat=torch.bmm(mask,node_feat)
        node_feat = node_feat.view(batch_size, length, self.type_num, chinnal)
        return node_feat,source_feat

    def forward(self, input,gt):
        gt_batch,gt_adj,gt_chinnal=gt.size()
        gt=gt.view(-1,gt_chinnal)
        gt1=self.gt1(gt)
        gt1_drop=F.dropout(gt1, self.dropout, training=self.training)
        gt2=self.gt2(gt1_drop)
        gt1=gt1.view(gt_batch,gt_adj,-1)
        gt2=gt2.view(gt_batch,gt_adj,-1)
        x=self.get_graph_feature(input,self.node_num) #batch_size,length, chinnal = input.size()
        atten=self.get_weigth_fuction(x)
        out_node_feat,out_source_feat=self.agg_node_fuction(x,atten)
        batch_size,length, adj_num,chinnal  = out_node_feat.size()
        out_node_feat=out_node_feat.view(batch_size*length,adj_num,chinnal).contiguous()
        out_source_feat=out_source_feat.view(batch_size*length,1,chinnal).contiguous()
        out_node_feat=out_node_feat.permute(0,2,1)
        out_source_feat=out_source_feat.permute(0,2,1)
        out_source_feat=self.source1(out_source_feat).squeeze(2)
        out_node_feat=self.agg1(out_node_feat).squeeze(2)
        node_feat=out_node_feat.view(batch_size,length,-1)[:,0,:]
        gt1=gt1[:,0,:].contiguous()
        scorce1=feature_measure(gt=gt1,node_feat=node_feat)
        output1=out_node_feat+out_source_feat
        output1 = F.dropout(output1, self.dropout, training=self.training)
        output1=output1.view(batch_size,length,-1)
        output2=self.get_graph_feature(output1,self.node_num)
        atten2=atten.view(batch_size,-1,atten.size(1),atten.size(2))
        atten2=atten2[:,:output2.size()[1],:,:]
        atten2=atten2.view(-1,atten.size(1),atten.size(2))
        out_node_feat2,out_source_feat2=self.agg_node_fuction(output2,atten2)
        batch_size2, length2, adj_num2, chinnal2 = out_node_feat2.size()
        out_node_feat2 = out_node_feat2.view(batch_size2 * length2, adj_num2, chinnal2).contiguous()
        out_source_feat2=out_source_feat2.view(batch_size2*length2,1,chinnal2)
        out_node_feat2 = out_node_feat2.permute(0, 2, 1)
        out_source_feat2=out_source_feat2.permute(0,2,1)
        out_node_feat2 = self.agg2(out_node_feat2)
        node_feat2 = out_node_feat2.view(batch_size2, length2, -1)
        gt2 = gt2[:, :node_feat2.size()[1], :]
        scorce2 = feature_measure(gt=gt2, node_feat=node_feat2)
        out_source_feat2=self.source2(out_source_feat2)
        output=out_node_feat2+out_source_feat2
        output=output.view(-1,self.out_features).contiguous()
        return output,scorce1,scorce2


if __name__ == "__main__":
    x=torch.arange(1,464)
    x=x.view(1,1,-1).contiguous().float().cuda()
    x=x.repeat(2,1024,1).permute(0,2,1)
    models=Graph_module_net_loss(in_features=1024,mid_features=512,out_features=1024,type_num=6,childs_num=[1,2,3,4,5,6],multi_head=8,dropout=0.1).cuda()
    out_put=models(x)