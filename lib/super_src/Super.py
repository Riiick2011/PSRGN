# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

import Super1D as _super_1d

class _Super1D(Function):
    @staticmethod
    def forward(ctx, input,center,gama,roi,feature_dim):
        ctx.save_for_backward(input,roi,center,gama)
        ctx.feature_dim = feature_dim
        ctx.input_shape = input.size()
        output = _super_1d.forward(
                input, roi,center,gama, feature_dim
            )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input,rois,center,gama = ctx.saved_tensors
        feature_dim = ctx.feature_dim
        bs, ch, t = ctx.input_shape

        grad_input,grad_center,grad_gama = _super_1d.backward(
            grad_output,
            input,
            rois,
            center,
            gama,
            feature_dim,
            bs,
            ch,
            t
        )
        return grad_input,grad_center,grad_gama, None, None

align1d = _Super1D.apply


class Super1DLayer(nn.Module):
    def __init__(self, feature_dim,N):
        super(Super1DLayer, self).__init__()
        self.feature_dim = feature_dim
        self.center=nn.Parameter(torch.arange(1,N+1).float()/float(N+1))
        self.gamma=nn.Parameter(torch.Tensor(N).float())
        self.gamma.data.normal_(1.0/N,1)

    def forward(self, input, rois):
        # print('- input shape is', input.shape)
        # print('- input mean is', input.mean())
        # print('- rois shape is', rois.shape)
        # print('- rois is on', rois.get_device())
        assert input.device==rois.device, 'Align operation requires ' + \
			'both feature and roi are on the same device! ' + \
            'Get feature on {} but roi on {}'.format(input.device,rois.device)
        out = align1d(input,  self.center,self.gamma,rois,self.feature_dim)

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
    layer = Super1DLayer(1,3).cuda()
    # layer = torch.nn.DataParallel(layer, device_ids=[0,1])
    input = torch.tensor([[[1.,2,3,4,5,6,7,8,9,10],[11,12,13,14,15,16,17,18,19,20]]]).cuda()
    proposal = torch.tensor([[0,0,3]]).cuda().float()
    output = layer(input, proposal)
    print(output.grad)
