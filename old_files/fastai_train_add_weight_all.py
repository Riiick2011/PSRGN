import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import argparse
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import numpy as np
import pathlib
from matplotlib import pyplot as plot
import fastai.train as train
import fastai.basic_data as basic_data
import random
from libs.modelings.models_pool import PtTransformer_gcn_loss_type_add_weight_all as final_model
from libs.datasets.Datasets import THUMOS14Dataset,mt_collate_fn,FormerLoss,FormerLoss_com,FormerLoss_reg,FormerLoss_metirc
from fastai_callback import Design_tensorboard,Model_save
from libs.core import load_config
from pprint import pprint
gpus=[0,1]

SEED=777
#seed = 8162
print(SEED)

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

cudnn.benchmark = True
pin_memory = True


def main(args):
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")
    model_cfg=cfg["model"]
    train_cfg=cfg["train_cfg"]
    data_cfg=cfg["dataset"]
    model =final_model(model_configs=model_cfg,train_cfg=train_cfg).cuda()
    #model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    train_loader = torch.utils.data.DataLoader(
        THUMOS14Dataset(is_training=True,split=cfg["train_split"],data_cfg=data_cfg,model_cfg=model_cfg,train_cfg=train_cfg),
        batch_size=cfg["loader"]["batch_size"], shuffle=True,
        num_workers=cfg["loader"]["num_workers"], pin_memory=True, drop_last=True)  # in training we drop the last incomplete minibatch

    val_loader = torch.utils.data.DataLoader(
        THUMOS14Dataset(is_training=True,split=cfg["val_split"],data_cfg=data_cfg,model_cfg=model_cfg,train_cfg=train_cfg),
        batch_size=cfg["loader"]["batch_size"], shuffle=False,
        num_workers=cfg["loader"]["num_workers"], pin_memory=True)

    data_bouch=basic_data.DataBunch(train_dl=train_loader,valid_dl=val_loader,collate_fn=mt_collate_fn)
    learner = train.Learner(data_bouch, model)
    pgnloss=FormerLoss_metirc(train_cfg=train_cfg,data_cfg=data_cfg)
    tboard_path = pathlib.Path('/data/zy/project/Actionformer_two-stage_1/tensorboard/project_5')
    learner.loss_func = pgnloss
    learner.path = learner.path / '/data/zy/project/Actionformer_two-stage_1'
    learner.model_dir = 'models/'
    learner.wd = cfg["opt"]["weight_decay"]
    #learner.load('epoch_41_model_MLP')
    mode_style = 3  # 1
    # find lr;2 check loss picture;3/4 fit_one_cycle();5 while
    if mode_style == 1:
        learner.lr_find()
        plot.show(learner.recorder.plot())
    elif mode_style == 2:
        learner.fit_one_cycle(1, max_lr=0.000006)
        plot.show(learner.recorder.plot_losses())
    elif mode_style == 3:
        learner.callbacks = [Design_tensorboard(base_dir=tboard_path, name='run1')]
        learner.callback_fns = [Model_save]
        learner.fit_one_cycle(60, max_lr=0.0001)
    else:
        print(learner.validate())

if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
      description='Train a point-based transformer for action localization')
    parser.add_argument('config', metavar='DIR',
                        help='path to a config file')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        help='print frequency (default: 10 iterations)')
    parser.add_argument('-c', '--ckpt-freq', default=5, type=int,
                        help='checkpoint frequency (default: every 5 epochs)')
    parser.add_argument('--output', default='', type=str,
                        help='name of exp folder (default: none)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to a checkpoint (default: none)')
    args = parser.parse_args()
    main(args)

