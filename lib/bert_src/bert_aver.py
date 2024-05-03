# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

import Bert1D as _aver_1d

class _Aver1D(Function):
    @staticmethod
    def forward(ctx, input, roi, feature_dim):# input =(batch_size,chinnal,T)
        ctx.save_for_backward(roi)
        ctx.feature_dim = feature_dim
        ctx.input_shape = input.size()
        output = _aver_1d.forward(
            input, roi, feature_dim
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        rois, = ctx.saved_tensors
        feature_dim = ctx.feature_dim
        bs, ch, t = ctx.input_shape
        grad_input = _aver_1d.backward(
            grad_output,
            rois,
            feature_dim,
            bs,
            ch,
            t
        )
        return grad_input, None, None, None, None

align1d = _Aver1D.apply


class Aver1DLayer(nn.Module):
    def __init__(self, feature_dim):
        super(Aver1DLayer, self).__init__()
        self.feature_dim = feature_dim

    def forward(self, input, rois):
        # print('- input shape is', input.shape)
        # print('- input mean is', input.mean())
        # print('- rois shape is', rois.shape)
        # print('- rois is on', rois.get_device())
        assert input.device==rois.device, 'Align operation requires ' + \
			'both feature and roi are on the same device! ' + \
            'Get feature on {} but roi on {}'.format(input.device,rois.device)

        out = align1d(input, rois, self.feature_dim)
        # print('- output shape is', out.shape)
        # print('- output mean is', out.mean())
        return out

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "feature_dim=" + str(self.feature_dim)
        tmpstr += "sampling_ratio=" + str(self.ratio)
        tmpstr += ")"
        return tmpstr

if __name__ == "__main__":
    layer = Aver1DLayer(1)
    # layer = torch.nn.DataParallel(layer, device_ids=[0,1])
    input = torch.tensor([[[1.,2,3,4,5,6,7,8,9,10],[11,12,13,14,15,16,17,18,19,20]]]).cuda().float()
    proposal = torch.tensor([[0,1,3],[0,4,6]]).cuda().float()
    output = layer(input, proposal)
    print("output has shape {}, with mean {}".format(output.shape, torch.mean(output)))
    print(output)

