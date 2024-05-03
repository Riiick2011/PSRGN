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
        scores = F.cosine_similarity(query, key,dim=3)
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, scores.size()[1], 1)
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn=scores
        if dropout is not None:
            p_attn = dropout(p_attn)
        return p_attn

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
        self.attention = Attention_4()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key,roi_mask, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key))]

        # 2) Apply attention on all the projected vectors in batch.
        attn = self.attention(query, key, roi_mask,mask=mask, dropout=self.dropout,softmax=self.soft_max)

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

    def forward(self, query, key, roi_mask,mask=None):
        batch_size,num,_,chinnal = query.size()

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query=query.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        key=key.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        # 2) Apply attention on all the projected vectors in batch.
        attn = self.attention(query, key, mask=mask, dropout=None,softmax=self.soft_max)


        attn=attn.view(batch_size,-1,num,num).contiguous()


        attn=attn.permute(0,2,3,1).contiguous()
        atten_mask=torch.zeros(attn.size()).cuda()

        #print(atten_mask.size())
        values,idces=torch.topk(attn,4,dim=2)
        #print(idces.size())
        atten_mask[:,:,idces,:]=1.0
        attn = attn * roi_mask[:, :, :, None].cuda()*atten_mask.cuda()#-one_attn

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

    def forward(self, query, key, roi_mask):
        chinal = query.size()[-1]
        node_num = query.size()[-2]
        ft=torch.cat((query,key),dim=3)
        ft=ft.view(-1,chinal*2)
        # 2) Apply attention on all the projected vectors in batch.
        attn = self.linear_layer(ft)
        attn = attn.view(-1, node_num, node_num, self.h)
        attn = torch.exp(attn)
        attn = attn * roi_mask[:, :, :, None].cuda()
        attn_sum = attn.sum(dim=2)
        attn = attn / attn_sum[:, :, None, :]
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

    def forward(self, query, key, roi_mask):
        chinal = query.size()[-1]
        node_num = query.size()[-2]
        ft=query+key
        ft=ft.view(-1,chinal)
        # 2) Apply attention on all the projected vectors in batch.
        attn = self.linear_layer(ft)
        attn = attn.view(-1, node_num, node_num, self.h)
        attn = torch.exp(attn)
        # 3) "Concat" using a view and apply a final linear.
        attn = attn * roi_mask[:, :, :, None].cuda()
        attn_sum = attn.sum(dim=2)
        attn = attn / attn_sum[:, :, None, :]
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

    def forward(self, query, key, roi_mask):
        chinal = query.size()[-1]
        node_num = query.size()[-2]
        ft=query-key
        ft=ft.view(-1,chinal)
        # 2) Apply attention on all the projected vectors in batch.
        attn = self.linear_layer(ft)
        attn = attn.view(-1, node_num, node_num, self.h)
        attn = torch.exp(attn)
        # 3) "Concat" using a view and apply a final linear.
        attn = attn * roi_mask[:, :, :, None].cuda()
        attn_sum = attn.sum(dim=2)
        attn = attn / attn_sum[:, :, None, :]
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

    def forward(self, query, key, roi_mask):
        chinal=query.size()[-1]
        node_num=query.size()[-2]
        ft=query*key
        ft=ft.view(-1,chinal)
        # 2) Apply attention on all the projected vectors in batch.
        attn = self.linear_layer(ft)
        attn=attn.view(-1,node_num,node_num,self.h)
        attn=torch.exp(attn)
        attn = attn * roi_mask[:, :, :, None].cuda()
        attn_sum = attn.sum(dim=2)
        attn = attn / attn_sum[:, :, None, :]

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
        self.atten=MultiHeadedAttention_2(h=self.multi_head,d_model=in_features,dropout=atten_dropout)
        self.atten2 = MultiHeadedAttention_2(h=self.multi_head, d_model=mid_features, dropout=atten_dropout)
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
        self.atten=MultiHeadedAttention_2(h=self.multi_head,d_model=in_features,dropout=atten_dropout)
        self.atten2 = MultiHeadedAttention_2(h=self.multi_head, d_model=mid_features, dropout=atten_dropout)
        self.agg1 = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=mid_features,
                                            kernel_size= 1, stride=1, groups=group,
                                            bias=bias),
                                  nn.ReLU(inplace=True))
        self.agg2 = nn.Sequential(nn.Conv1d(in_channels=mid_features,out_channels=out_features,
                                           kernel_size=1,stride=1,groups=group,bias=bias),
                                 nn.ReLU(inplace=True))
        self.layer_nomarl=nn.LayerNorm(out_features,eps=0.000001)
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
        atten=self.atten(source_feat,adj_feat,roi_mask)
        return atten
    def get_weigth_fuction2(self,x,roi_mask):
        batch_size, num, chinnal = x.size()
        source_feat=x.unsqueeze(1).repeat(1,num,1,1)
        adj_feat=x.unsqueeze(2).repeat(1,1,num,1)
        atten=self.atten2(source_feat,adj_feat,roi_mask)
        return atten
    def forward(self, input,masks_roi,score_mask,gt_feat):
        gts=self.gt(gt_feat)

        bacth,num,chinal=input.size()
        gts = gts.view(bacth,-1,self.out_features)
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