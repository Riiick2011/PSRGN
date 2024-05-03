import math

import numpy as np
import torch
import fastai.callback as callback
import fastai.metrics as metrics
import torch.nn.functional as F
from fastai.basic_train import Learner
from fastai.basic_train import LearnerCallback
from fastai.basic_data import DatasetType, DataBunch
from fastai.torch_core import *
from threading import Thread, Event
from time import sleep
from queue import Queue
import statistics
import torchvision.utils as vutils
from abc import ABC
from tensorboardX import SummaryWriter

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))
    return res


class Design_tensorboard(callback.Callback):
    def __init__(self,  base_dir: Path, name: str,num_batch: int = 100):
        self.base_dir= base_dir
        self.name=name
        self.num_batch = num_batch
        log_dir = base_dir / name
        self.tbwriter = SummaryWriter(str(log_dir),flush_secs=30)
        self.data = None
        self.train_loss=AverageMeter()
        self.val_loss=AverageMeter()
        self.epoch=0
        self.metrics_root = '/metrics/'
    def on_epoch_begin(self, **kwargs:Any) ->None:
        self.train_loss.reset()
        self.val_loss.reset()
    def _write_training_loss(self, iteration: int, epoch_loss: Tensor) -> None:
        "Writes training loss to Tensorboard."
        scalar_value = to_np(epoch_loss)
        tag = self.metrics_root + 'train_loss'
        self.tbwriter.add_scalar(tag=tag, scalar_value=scalar_value, global_step=iteration)

    def _write_testing_loss(self, iteration: int, epoch_loss: Tensor) -> None:
        "Writes training loss to Tensorboard."
        scalar_value = to_np(epoch_loss)
        tag = self.metrics_root + 'test_loss'
        self.tbwriter.add_scalar(tag=tag, scalar_value=scalar_value, global_step=iteration)


    def _write_scalar(self, name: str, scalar_value, iteration: int) -> None:
        "Writes single scalar value to Tensorboard."
        tag = self.metrics_root + name
        self.tbwriter.add_scalar(tag=tag, scalar_value=scalar_value, global_step=iteration)

    # TODO:  Relying on a specific hardcoded start_idx here isn't great.  Is there a better solution?

    def _write_metrics(self, iteration: int, last_metrics: MetricsList) -> None:
        "Writes training metrics to Tensorboard."
        scalar_value = last_metrics[1]
        self._write_scalar(name='AMP', scalar_value=scalar_value, iteration=iteration)

    def _write_accuracy(self, iteration: int, last_metrics: MetricsList) -> None:
        "Writes training metrics to Tensorboard."
        scalar_value = last_metrics[2]
        self._write_scalar(name='accuracy', scalar_value=scalar_value, iteration=iteration)

    def _write_val_loss(self, iteration: int, last_metrics: MetricsList) -> None:
        "Writes training metrics to Tensorboard."
        scalar_value = last_metrics[0]
        self._write_scalar(name='val_loss', scalar_value=scalar_value, iteration=iteration)

    def _write_wmp(self, iteration: int, last_metrics: MetricsList) -> None:
        "Writes training metrics to Tensorboard."
        scalar_value = last_metrics[3]
        self._write_scalar(name='wmp', scalar_value=scalar_value, iteration=iteration)


    def on_batch_end(self, last_loss: Tensor, last_target , iteration: int, train: bool ,**kwargs) -> None:
        "Callback function that writes batch end appropriate data to Tensorboard."

        batch_size=last_target[0][1].size()[0]
        if train:
            self.train_loss.update(last_loss.item(), batch_size)
    # Doing stuff here that requires gradient info, because they get zeroed out afterwards in training loop

    def on_epoch_end(self, last_metrics: MetricsList, **kwargs) -> None:
        "Callback function that writes epoch end appropriate data to Tensorboard."
        train_loss=self.train_loss.avg
        train_loss=tensor(train_loss)
        val_loss=self.val_loss.avg
        val_loss=tensor(val_loss)
        self.epoch=self.epoch+1
        self._write_training_loss(iteration=self.epoch,epoch_loss=train_loss)
        self._write_testing_loss(iteration=self.epoch,epoch_loss=val_loss)
        #self._write_metrics(iteration=self.epoch, last_metrics=last_metrics)
        self._write_val_loss(iteration=self.epoch, last_metrics=last_metrics)
        #self._write_accuracy(iteration=self.epoch, last_metrics=last_metrics)
        #self._write_wmp(iteration=self.epoch, last_metrics=last_metrics)


class Best_modelsave(LearnerCallback):
    def __init__(self,learn:Learner):
        super().__init__(learn)
        self.best_loss=1.4968494

    def on_epoch_end(self, last_metrics: MetricsList,**kwargs) -> None:
        "Callback function that writes epoch end appropriate data to Tensorboard."
        scalar_value = float(last_metrics[0])
        #scalar_value_ac = float(last_metrics[2])
        if scalar_value<self.best_loss:
            self.best_loss=scalar_value
            self.learn.save('best_model_val')

class Metric_Act_acc(callback.Callback):
    def __init__(self,adj_num):
        self.name="act_acc"
        #self.apm = APMeter()
        self.act_accuracies = AverageMeter()
        self.adj_num=adj_num

    def on_epoch_begin(self,**kwargs):
        self.act_accuracies.reset()
    def on_batch_end(self,last_output,last_target,**kwargs):
        raw_act_fc = last_output[0]
        prop_type = last_target[0][0]
        target = last_target[0][1]
        type_data = prop_type.view(-1).data  # only train have
        type_data = type_data[0: -1: self.adj_num]  # only train have
        act_indexer = ((type_data == 0) + (type_data == 2)).nonzero().squeeze()
        activity_out = raw_act_fc[act_indexer, :]
        target = target.view(-1)  # only train have
        target = target[0: -1: self.adj_num]
        activity_target = target[act_indexer]
        act_acc = accuracy(activity_out, activity_target)
        self.act_accuracies.update(act_acc[0].item(), activity_out.size(0))

    def on_epoch_end(self,last_metrics,epoch, **kwargs):

        amp_meter=self.act_accuracies.avg
        return metrics.add_metrics(last_metrics,amp_meter)

class Metric_Fg_acc(callback.Callback):
    def __init__(self,adj_num):
        self.name="fg_acc"
        #self.apm = APMeter()
        self.fg_accuracies = AverageMeter()
        self.adj_num=adj_num

    def on_epoch_begin(self,**kwargs):
        self.fg_accuracies.reset()
    def on_batch_end(self,last_output,last_target,**kwargs):
        raw_act_fc = last_output[0]
        prop_type = last_target[0][0]
        target = last_target[0][1]
        type_data = prop_type.view(-1).data  # only train have
        type_data = type_data[0: -1: self.adj_num]  # only train have
        act_indexer = ((type_data == 0) + (type_data == 2)).nonzero().squeeze()
        activity_out = raw_act_fc[act_indexer, :]
        target = target.view(-1)  # only train have
        target = target[0: -1: self.adj_num]
        activity_target = target[act_indexer]
        activity_prop_type = type_data[act_indexer]
        fg_indexer = (activity_prop_type == 0).nonzero().squeeze()
        fg_acc = accuracy(activity_out[fg_indexer, :], activity_target[fg_indexer])
        self.fg_accuracies.update(fg_acc[0].item(), len(fg_indexer))

    def on_epoch_end(self,last_metrics,epoch, **kwargs):

        amp_meter=self.fg_accuracies.avg
        return metrics.add_metrics(last_metrics,amp_meter)



class Metric_Bg_acc(callback.Callback):
    def __init__(self,adj_num):
        self.name="bg_acc"
        #self.apm = APMeter()
        self.bg_accuracies = AverageMeter()
        self.adj_num=adj_num

    def on_epoch_begin(self,**kwargs):
        self.bg_accuracies.reset()
    def on_batch_end(self,last_output,last_target,**kwargs):
        raw_act_fc = last_output[0]
        prop_type = last_target[0][0]
        target = last_target[0][1]
        type_data = prop_type.view(-1).data  # only train have
        type_data = type_data[0: -1: self.adj_num]  # only train have
        act_indexer = ((type_data == 0) + (type_data == 2)).nonzero().squeeze()
        activity_out = raw_act_fc[act_indexer, :]
        target = target.view(-1)  # only train have
        target = target[0: -1: self.adj_num]
        activity_target = target[act_indexer]
        activity_prop_type = type_data[act_indexer]
        bg_indexer = (activity_prop_type == 2).nonzero().squeeze()
        if bg_indexer.dim()==0:
            bg_indexer=bg_indexer.unsqueeze(0)
        if len(bg_indexer) > 0:
            bg_acc = accuracy(activity_out[bg_indexer, :], activity_target[bg_indexer])
            self.bg_accuracies.update(bg_acc[0].item(), len(bg_indexer))

    def on_epoch_end(self,last_metrics,epoch, **kwargs):

        amp_meter=self.bg_accuracies.avg
        return metrics.add_metrics(last_metrics,amp_meter)


class Model_save(LearnerCallback):
    def __init__(self,learn:Learner):
        super().__init__(learn)
        self.loss=0
        self.epoch=0

    def on_epoch_end(self, last_metrics: MetricsList,**kwargs) -> None:
        "Callback function that writes epoch end appropriate data to Tensorboard."
        self.loss = float(last_metrics[0])
        self.epoch=self.epoch+1
        #scalar_value_ac = float(last_metrics[2])
        if self.epoch%1==0:
            self.learn.save('epoch_%d_model_GCN'%self.epoch)