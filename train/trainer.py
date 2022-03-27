# Code adapted from GrabNet

import os
import sys

sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch

from datetime import datetime

from tools.utils import makepath, makelogger
from data.dataloader import LoadData
from models.models import GNet

from torch import nn, optim
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter


class Trainer:

    def __init__(self, cfg):

        self.dtype = torch.float64

        work_dir = cfg.work_dir
        starttime = datetime.now().replace(microsecond=0)
        makepath(work_dir, isfile=False)
        logger = makelogger(makepath(os.path.join(work_dir, 'train.log'), isfile=True)).info
        self.logger = logger

        summary_logdir = os.path.join(work_dir, 'summaries')
        self.swriter = SummaryWriter(log_dir=summary_logdir)
        logger(' - Started training GNet, experiment code %s' % (starttime))
        logger('tensorboard --logdir=%s' % summary_logdir)
        logger('Torch Version: %s\n' % torch.__version__)

        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.empty_cache()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ds_train = LoadData(cfg.dataset_dir, ds_name=cfg.ds_train)
        ds_val = LoadData(cfg.dataset_dir, ds_name=cfg.ds_val)
        self.ds_train = DataLoader(ds_train, batch_size=cfg.batch_size, num_workers=cfg.n_workers, shuffle=True, drop_last=True)
        self.ds_val = DataLoader(ds_val, batch_size=cfg.batch_size, num_workers=cfg.n_workers, shuffle=True, drop_last=True)

        self.gnet = GNet().to(self.device)

        self.LossL1 = torch.nn.L1Loss(reduction='mean')
        self.LossL2 = torch.nn.MSELoss(reduction='mean')

        self.try_num = cfg.try_num
        self.epochs_completed = 0

        if cfg.use_multigpu:
            self.gnet = nn.DataParallel(self.gnet)
            logger("Training on Multiple GPUs")

        vars_gnet = [var[1] for var in self.gnet.named_parameters()]
        gnet_n_params = sum(p.numel() for p in vars_gnet if p.requires_grad)
        logger('Total Trainable Parameters for GNet is %2.2f M.' % ((gnet_n_params) * 1e-6))

        self.optimizer_gnet = optim.Adam(vars_gnet, lr=cfg.base_lr, weight_decay=cfg.reg_coef)

        self.best_loss_gnet = np.inf
        self.cfg = cfg

    def train(self):

        self.gnet.train()
        torch.autograd.set_detect_anomaly(True)

        train_loss_dict_gnet = {}

        for it, data in enumerate(self.ds_train):
            data = {k: data[k].to(self.device) for k in data.keys()}

            self.optimizer_gnet.zero_grad()

            drec_gnet = self.gnet(**data)
            loss_total_gnet, cur_loss_dict_gnet = self.loss_gnet(data, drec_gnet)

            loss_total_gnet.backward()
            self.optimizer_gnet.step()

            train_loss_dict_gnet = {k: train_loss_dict_gnet.get(k, 0.0) + v.item() for k, v in cur_loss_dict_gnet.items()}
            if it % (self.cfg.save_every_it + 1) == 0:
                cur_train_loss_dict_gnet = {k: v / (it + 1) for k, v in train_loss_dict_gnet.items()}
                train_msg = self.create_loss_message(cur_train_loss_dict_gnet,
                                                    expr_ID=self.cfg.expr_ID,
                                                    epoch_num=self.epochs_completed,
                                                    model_name='GNet',
                                                    it=it,
                                                    try_num=self.try_num,
                                                    mode='train')

                self.logger(train_msg)

        train_loss_dict_gnet = {k: v / len(self.ds_train) for k, v in train_loss_dict_gnet.items()}
        return train_loss_dict_gnet

    def loss_gnet(self, data, drec, ds_name='train_data'):

        ### verts loss
        loss_verts = 90. * (1. - self.cfg.kl_coef) * self.LossL1(data['verts'].flatten(start_dim=1), drec['verts'])
        ### pose loss
        loss_pose = 100. * (1. - self.cfg.kl_coef) * self.LossL2(data['fullpose_rotmat'], drec['fullpose_rotmat'])
        ### body translation loss
        loss_body_transl = 100. * (1. - self.cfg.kl_coef) * self.LossL2(data['body_transl'], drec['body_transl'])
        ### hand to object distance loss
        loss_dists_h2o = 90. * (1. - self.cfg.kl_coef) * self.LossL1(data['hand_object_dists'], drec['hand_object_dists'])
        ### KL loss
        q_z = torch.distributions.normal.Normal(drec['mean'], drec['std'])
        p_z = torch.distributions.normal.Normal(
            loc=torch.tensor(np.zeros([self.cfg.batch_size, self.cfg.latentD]), requires_grad=False).to(self.device).type(self.dtype),
            scale=torch.tensor(np.ones([self.cfg.batch_size, self.cfg.latentD]), requires_grad=False).to(self.device).type(self.dtype)
        )
        loss_kl = 30 * self.cfg.kl_coef * torch.mean(torch.sum(torch.distributions.kl.kl_divergence(q_z, p_z)))

        loss_dict = {
            'loss_kl': loss_kl,
            'loss_verts': loss_verts,
            'loss_pose': loss_pose,
            'loss_body_transl': loss_body_transl,
            'loss_dists_h2o': loss_dists_h2o
        }

        loss_total = torch.stack(list(loss_dict.values())).sum()
        loss_dict['loss_total'] = loss_total

        return loss_total, loss_dict

    def evaluate(self, ds_name='train_data'):
        self.gnet.eval()

        eval_loss_dict_gnet = {}

        dataset = self.ds_val

        with torch.no_grad():
            for data in dataset:
                data = {k: data[k].to(self.device) for k in data.keys()}
                drec_gnet = self.gnet(**data)
                loss_total_gnet, cur_loss_dict_gnet = self.loss_gnet(data, drec_gnet)
                eval_loss_dict_gnet = {k: eval_loss_dict_gnet.get(k, 0.0) + v.item() for k, v in cur_loss_dict_gnet.items()}

            eval_loss_dict_gnet = {k: v / len(dataset) for k, v in eval_loss_dict_gnet.items()}

        return eval_loss_dict_gnet

    def fit(self, n_epochs=None):
        starttime = datetime.now().replace(microsecond=0)
        if n_epochs is None:
            n_epochs = self.cfg.n_epochs

        self.logger('Started Training at %s for %d epochs' % (datetime.strftime(starttime, '%Y-%m-%d_%H:%M:%S'), n_epochs))

        prev_lr_gnet = np.inf
        self.fit_gnet = True

        lr_scheduler_gnet = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_gnet, 'min')

        for epoch_num in range(1, n_epochs + 1):
            self.logger('--- starting Epoch # %03d' % epoch_num)

            train_loss_dict_gnet = self.train()
            eval_loss_dict_gnet = self.evaluate()

            if self.fit_gnet:
                lr_scheduler_gnet.step(eval_loss_dict_gnet['loss_total'])
                cur_lr_gnet = self.optimizer_gnet.param_groups[0]['lr']

                if cur_lr_gnet != prev_lr_gnet:
                    self.logger('--- GNet learning rate changed from %.2e to %.2e ---' % (prev_lr_gnet, cur_lr_gnet))
                    prev_lr_gnet = cur_lr_gnet

                with torch.no_grad():
                    eval_msg = Trainer.create_loss_message(eval_loss_dict_gnet, expr_ID=self.cfg.expr_ID,
                                                           epoch_num=self.epochs_completed, it=len(self.ds_val),
                                                           model_name='GNet',
                                                           try_num=self.try_num, mode='evald')
                    if eval_loss_dict_gnet['loss_total'] < self.best_loss_gnet:

                        self.cfg.best_gnet = makepath(os.path.join(self.cfg.work_dir, 'snapshots',
                                                                   'TR%02d_E%03d_gnet.pt' % (
                                                                   self.try_num, self.epochs_completed)), isfile=True)
                        self.save_gnet()
                        self.logger(eval_msg + ' ** ')
                        self.best_loss_gnet = eval_loss_dict_gnet['loss_total']

                    else:
                        self.logger(eval_msg)

                    self.swriter.add_scalars('total_loss_gnet/scalars',
                                             {'train_loss_total': train_loss_dict_gnet['loss_total'],
                                              'evald_loss_total': eval_loss_dict_gnet['loss_total'], },
                                             self.epochs_completed)

            self.epochs_completed += 1

        endtime = datetime.now().replace(microsecond=0)

        self.logger('Finished Training at %s\n' % (datetime.strftime(endtime, '%Y-%m-%d_%H:%M:%S')))
        self.logger(
            'Training done in %s!\n' % (endtime - starttime))
        self.logger('Best GNet val total loss achieved: %.2e\n' % (self.best_loss_gnet))
        self.logger('Best GNet model path: %s\n' % self.cfg.best_gnet)

    def save_gnet(self):
        torch.save(self.gnet.module.state_dict()
                   if isinstance(self.gnet, torch.nn.DataParallel)
                   else self.gnet.state_dict(), self.cfg.best_gnet)

    @staticmethod
    def create_loss_message(loss_dict, expr_ID='XX', epoch_num=0,model_name='GNet', it=0, try_num=0, mode='evald'):
        ext_msg = ' | '.join(['%s = %.2e' % (k, v) for k, v in loss_dict.items() if k != 'loss_total'])
        return '[%s]_TR%02d_E%03d - It %05d - %s - %s: [T:%.2e] - [%s]' % (
            expr_ID, try_num, epoch_num, it,model_name, mode, loss_dict['loss_total'], ext_msg)
