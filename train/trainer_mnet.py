# Code adapted from GrabNet

import os
import sys

sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch

from datetime import datetime

from tools.utils import makepath, makelogger
from data.dataloader_mnet import LoadData
from models.models import MNet

from torch import nn, optim
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter


class Trainer:

    def __init__(self, cfg):

        self.dtype = torch.float64

        work_dir = cfg.work_dir
        starttime = datetime.now().replace(microsecond=0)
        makepath(work_dir, isfile=False)
        logger = makelogger(makepath(os.path.join(work_dir, 'train_mnet.log'), isfile=True)).info
        self.logger = logger

        summary_logdir = os.path.join(work_dir, 'summaries')
        self.swriter = SummaryWriter(log_dir=summary_logdir)
        logger(' - Started training MNet, experiment code %s' % (starttime))
        logger('tensorboard --logdir=%s' % summary_logdir)
        logger('Torch Version: %s\n' % torch.__version__)

        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.empty_cache()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ds_train = LoadData(cfg.dataset_dir, ds_name=cfg.ds_train)
        ds_val = LoadData(cfg.dataset_dir, ds_name=cfg.ds_val)
        print(len(ds_train))
        print(len(ds_val))
        self.ds_train = DataLoader(ds_train, batch_size=cfg.batch_size, num_workers=cfg.n_workers, shuffle=True, drop_last=True)
        self.ds_val = DataLoader(ds_val, batch_size=cfg.batch_size, num_workers=cfg.n_workers, shuffle=True, drop_last=True)

        self.mnet = MNet().to(self.device)

        self.LossL1 = torch.nn.L1Loss(reduction='mean')
        self.LossL2 = torch.nn.MSELoss(reduction='mean')

        self.try_num = cfg.try_num
        self.epochs_completed = 0

        if cfg.use_multigpu:
            self.mnet = nn.DataParallel(self.mnet)
            logger("Training on Multiple GPUs")

        vars_mnet = [var[1] for var in self.mnet.named_parameters()]
        mnet_n_params = sum(p.numel() for p in vars_mnet if p.requires_grad)
        logger('Total Trainable Parameters for MNet is %2.2f M.' % ((mnet_n_params) * 1e-6))

        self.optimizer_mnet = optim.Adam(vars_mnet, lr=cfg.base_lr, weight_decay=cfg.reg_coef)

        self.best_loss_mnet = np.inf
        self.cfg = cfg

    def train(self):

        self.mnet.train()
        torch.autograd.set_detect_anomaly(True)

        train_loss_dict_mnet = {}

        for it, data in enumerate(self.ds_train):
            data = {k: data[k].to(self.device) for k in data.keys()}

            self.optimizer_mnet.zero_grad()

            drec_mnet = self.mnet(**data)
            loss_total_mnet, cur_loss_dict_mnet = self.loss_mnet(data, drec_mnet)

            loss_total_mnet.backward()
            self.optimizer_mnet.step()

            train_loss_dict_mnet = {k: train_loss_dict_mnet.get(k, 0.0) + v.item() for k, v in cur_loss_dict_mnet.items()}
            if it % (self.cfg.save_every_it + 1) == 0:
                cur_train_loss_dict_mnet = {k: v / (it + 1) for k, v in train_loss_dict_mnet.items()}
                train_msg = self.create_loss_message(cur_train_loss_dict_mnet,
                                                    expr_ID=self.cfg.expr_ID,
                                                    epoch_num=self.epochs_completed,
                                                    model_name='MNet',
                                                    it=it,
                                                    try_num=self.try_num,
                                                    mode='train')

                self.logger(train_msg)

        train_loss_dict_mnet = {k: v / len(self.ds_train) for k, v in train_loss_dict_mnet.items()}
        return train_loss_dict_mnet

    def loss_mnet(self, data, drec, ds_name='train_data'):

        ### verts loss
        loss_verts = 90. * (1. - self.cfg.kl_coef) * self.LossL1(data['future_verts_delta'], drec['future_verts_delta'])
        ### pose loss
        loss_pose = 100. * (1. - self.cfg.kl_coef) * self.LossL1(data['future_pose_delta'], drec['future_pose_delta'])
        ### body translation loss
        loss_body_transl = 100. * (1. - self.cfg.kl_coef) * self.LossL2(data['future_transl_delta'], drec['future_transl_delta'])
        ### hand to object distance loss
        loss_dists = 90. * (1. - self.cfg.kl_coef) * self.LossL1(data['future_dists_delta'], drec['future_dists_delta'])

        loss_dict = {
            'loss_verts': loss_verts,
            'loss_pose': loss_pose,
            'loss_body_transl': loss_body_transl,
            'loss_dists': loss_dists
        }

        loss_total = torch.stack(list(loss_dict.values())).sum()
        loss_dict['loss_total'] = loss_total

        return loss_total, loss_dict

    def evaluate(self):
        self.mnet.eval()

        eval_loss_dict_mnet = {}

        dataset = self.ds_val

        with torch.no_grad():
            for data in dataset:
                data = {k: data[k].to(self.device) for k in data.keys()}
                drec_mnet = self.mnet(**data)
                loss_total_mnet, cur_loss_dict_mnet = self.loss_mnet(data, drec_mnet)
                eval_loss_dict_mnet = {k: eval_loss_dict_mnet.get(k, 0.0) + v.item() for k, v in cur_loss_dict_mnet.items()}

            eval_loss_dict_mnet = {k: v / len(dataset) for k, v in eval_loss_dict_mnet.items()}

        return eval_loss_dict_mnet

    def fit(self, n_epochs=None):
        starttime = datetime.now().replace(microsecond=0)
        if n_epochs is None:
            n_epochs = self.cfg.n_epochs

        self.logger('Started Training at %s for %d epochs' % (datetime.strftime(starttime, '%Y-%m-%d_%H:%M:%S'), n_epochs))

        prev_lr_mnet = np.inf
        self.fit_mnet = True

        lr_scheduler_mnet = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_mnet, 'min')

        for epoch_num in range(1, n_epochs + 1):
            self.logger('--- starting Epoch # %03d' % epoch_num)

            train_loss_dict_mnet = self.train()
            eval_loss_dict_mnet = self.evaluate()

            if self.fit_mnet:
                lr_scheduler_mnet.step(eval_loss_dict_mnet['loss_total'])
                cur_lr_mnet = self.optimizer_mnet.param_groups[0]['lr']

                if cur_lr_mnet != prev_lr_mnet:
                    self.logger('--- MNet learning rate changed from %.2e to %.2e ---' % (prev_lr_mnet, cur_lr_mnet))
                    prev_lr_mnet = cur_lr_mnet

                with torch.no_grad():
                    eval_msg = Trainer.create_loss_message(eval_loss_dict_mnet, expr_ID=self.cfg.expr_ID,
                                                           epoch_num=self.epochs_completed, it=len(self.ds_val),
                                                           model_name='MNet',
                                                           try_num=self.try_num, mode='evald')
                    if eval_loss_dict_mnet['loss_total'] < self.best_loss_mnet:

                        self.cfg.best_mnet = makepath(os.path.join(self.cfg.work_dir, 'snapshots',
                                                                   'TR%02d_E%03d_mnet.pt' % (
                                                                   self.try_num, self.epochs_completed)), isfile=True)
                        self.save_mnet()
                        self.logger(eval_msg + ' ** ')
                        self.best_loss_mnet = eval_loss_dict_mnet['loss_total']

                    else:
                        self.logger(eval_msg)

                    self.swriter.add_scalars('total_loss_mnet/scalars',
                                             {'train_loss_total': train_loss_dict_mnet['loss_total'],
                                              'evald_loss_total': eval_loss_dict_mnet['loss_total'], },
                                             self.epochs_completed)

            self.epochs_completed += 1

        endtime = datetime.now().replace(microsecond=0)

        self.logger('Finished Training at %s\n' % (datetime.strftime(endtime, '%Y-%m-%d_%H:%M:%S')))
        self.logger(
            'Training done in %s!\n' % (endtime - starttime))
        self.logger('Best MNet val total loss achieved: %.2e\n' % (self.best_loss_mnet))
        self.logger('Best MNet model path: %s\n' % self.cfg.best_mnet)

    def save_mnet(self):
        torch.save(self.mnet.module.state_dict()
                   if isinstance(self.mnet, torch.nn.DataParallel)
                   else self.mnet.state_dict(), self.cfg.best_mnet)

    @staticmethod
    def create_loss_message(loss_dict, expr_ID='XX', epoch_num=0,model_name='MNet', it=0, try_num=0, mode='evald'):
        ext_msg = ' | '.join(['%s = %.2e' % (k, v) for k, v in loss_dict.items() if k != 'loss_total'])
        return '[%s]_TR%02d_E%03d - It %05d - %s - %s: [T:%.2e] - [%s]' % (
            expr_ID, try_num, epoch_num, it,model_name, mode, loss_dict['loss_total'], ext_msg)
