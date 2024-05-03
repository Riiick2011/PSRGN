import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch

import math
from lib.bert_src.bert_aver import Aver1DLayer

class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key,value, mask=None, dropout=None):
        scores = torch.abs(F.cosine_similarity(query.repeat(1,1,key.size()[2],1), key,dim=3))
        mask=mask.unsqueeze(1).repeat(1,scores.size()[1],1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = torch.exp(scores)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return p_attn[:,:,:,None]*value,p_attn
class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.attention = Attention()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key,value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key,value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key,value))]

        # 2) Apply attention on all the projected vectors in batch.
        x,attn = self.attention(query, key,value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.

        return x, attn


class BERT_pool(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, input_dim, attn_heads=8, dropout=0.1, mask_prob=0.8):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.attn_heads = attn_heads
        self.input_dim = input_dim
        self.mask_prob = mask_prob
        self.dropout_rate=dropout

        clsToken = torch.zeros(1, 1, self.input_dim).float().cuda()
        clsToken.require_grad = True
        self.clsToken = nn.Parameter(clsToken)
        torch.nn.init.normal_(self.clsToken, std=self.input_dim ** -0.5)
        self.a_2 = nn.Parameter(torch.ones_like(self.clsToken))
        self.b_2 = nn.Parameter(torch.zeros_like(self.clsToken))

        # paper noted they used 4*hidden_size for ff_network_hidden_size

        # embedding for BERT, sum of positional, segment, token embeddings

        # multi-layers transformer blocks, deep network
        self.attention = MultiHeadedAttention(h=attn_heads,d_model=self.input_dim,dropout=self.dropout_rate)
        self.pool=Aver1DLayer(1)


        # nn.init.xavier_normal_(self.transformer_blocks[0].feed_forward.w_2.weight, gain = 1/(0.425) ** 0.5)
        # nn.init.xavier_normal_(self.transformer_blocks[0].feed_forward.w_1.weight, gain = 1)

    def forward(self, input_vectors, rois):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        input_vectors=input_vectors.permute(0,2,1).contiguous()
        batch_size = input_vectors.shape[0]
        max_len = input_vectors.shape[1]
        sample = None

        if self.training:
            bernolliMatrix=torch.cat((torch.tensor([1]).float().cuda(), (torch.tensor([self.mask_prob]).float().cuda()).repeat(max_len)), 0).unsqueeze(0).repeat([batch_size,1])
            self.bernolliDistributor = torch.distributions.Bernoulli(bernolliMatrix)
            sample = self.bernolliDistributor.sample()
            mask = (sample > 0).float()
        else:
            mask = torch.ones(batch_size,max_len + 1).cuda()

        # embedding the indexed sequence to sequence of vectors
        clstoken_scales = self.clsToken * self.a_2 + self.b_2
        x = torch.cat((clstoken_scales.repeat(batch_size, 1, 1), input_vectors), 1)
        value, attn=self.attention(x[:,0],x,x,mask)
        value=value.permute(0,1,3,2).contiguous().view(batch_size,-1,max_len+1)
        attn=attn.view(batch_size,-1,max_len+1)
        rois[:,1]=rois[:,1]+1
        rois[:, 2] = rois[:, 2] + 1
        pool_value=self.pool(value,rois)
        pool_value=pool_value.view(rois.size()[0],self.attn_heads,-1)
        final_value=pool_value
        final_value=final_value.view(rois.size()[0],-1)
        return final_value
if __name__ == "__main__":
    inputs=torch.rand((2,10,1024)).float().cuda()
    proposal = torch.tensor([[0,0,5]]).cuda().float()
    model=BERT_pool(input_dim=1024,attn_heads=8,dropout=0.1,mask_prob=0.8).cuda()
    out=model(inputs,proposal)
    print(out.size())