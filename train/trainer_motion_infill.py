# Code adapted from GrabNet

import os
import sys

sys.path.append('.')
sys.path.append('..')

from datetime import datetime

from tools.utils import makepath, makelogger
from data.dataloader import LoadData
from models.models import *

from torch import nn, optim
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter


class Trainer:

    def __init__(self, cfg):

        work_dir = cfg.work_dir
        starttime = datetime.now().replace(microsecond=0)
        makepath(work_dir, isfile=False)
        logger = makelogger(makepath(os.path.join(work_dir, 'train_motion_infill.log'), isfile=True)).info
        self.logger = logger
        self.dtype = torch.float32

        summary_logdir = os.path.join(work_dir, 'summaries')
        self.swriter = SummaryWriter(log_dir=summary_logdir)
        logger(' - Started training Motion Infilling Net, experiment code %s' % (starttime))
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

        self.motionFill = MotionFill().to(self.device)

        self.LossL1 = torch.nn.L1Loss(reduction='mean')
        self.LossL2 = torch.nn.MSELoss(reduction='mean')
        self.bce_loss = torch.nn.BCEWithLogitsLoss().to(self.device)

        self.try_num = cfg.try_num
        self.epochs_completed = 0

        if cfg.use_multigpu:
            self.motionFill = nn.DataParallel(self.motionFill)
            logger("Training on Multiple GPUs")

        vars_motionFill = [var[1] for var in self.motionFill.named_parameters()]
        motionFill_n_params = sum(p.numel() for p in vars_motionFill if p.requires_grad)
        logger('Total Trainable Parameters for Motion Infilling Net is %2.2f M.' % ((motionFill_n_params) * 1e-6))

        self.optimizer_motioinFill = optim.Adam(vars_motionFill, lr=cfg.base_lr, weight_decay=cfg.reg_coef)

        self.best_loss_motionFill = np.inf
        self.cfg = cfg

    def train(self):

        self.motionFill.train()
        torch.autograd.set_detect_anomaly(True)

        train_loss_dict_motionFill = {}

        for it, data in enumerate(self.ds_train):
            data = {k: data[k].to(self.device) for k in data.keys()}

            self.optimizer_motioinFill.zero_grad()

            drec_motionFill = self.motionFill(**data)
            loss_total_motionFill, cur_loss_dict_motionFill = self.loss_motionFill(data, drec_motionFill)

            loss_total_motionFill.backward()
            self.optimizer_motioinFill.step()

            train_loss_dict_motionFill = {k: train_loss_dict_motionFill.get(k, 0.0) + v.item() for k, v in cur_loss_dict_motionFill.items()}
            if it % (self.cfg.save_every_it + 1) == 0:
                cur_train_loss_dict_motionFill = {k: v / (it + 1) for k, v in train_loss_dict_motionFill.items()}
                train_msg = self.create_loss_message(cur_train_loss_dict_motionFill,
                                                    expr_ID=self.cfg.expr_ID,
                                                    epoch_num=self.epochs_completed,
                                                    model_name='MotionInfilling',
                                                    it=it,
                                                    try_num=self.try_num,
                                                    mode='train')

                self.logger(train_msg)

        train_loss_dict_motionFill = {k: v / len(self.ds_train) for k, v in train_loss_dict_motionFill.items()}
        return train_loss_dict_motionFill

    def loss_motionFill(self, data, drec):
        bs = data['traj'].shape[0]
        ### traj loss
        loss_traj = 90. * (1. - self.cfg.kl_coef) * self.LossL1(data['traj'], drec['traj'])
        ### root velocity loss
        traj_data = data['traj'].reshape(bs, data['traj'].shape[1]//3, 3)
        traj_drec = drec['traj'].reshape(bs, drec['traj'].shape[1]//3, 3)
        v_data = traj_data[:, :-1, :] - traj_data[:, 1:, :]
        v_drec = traj_drec[:, :-1, :] - traj_drec[:, 1:, :]
        loss_root_velocity = 90. * (1. - self.cfg.kl_coef) * self.LossL1(v_data, v_drec)
        ### marker loss
        loss_marker = 200. * (1. - self.cfg.kl_coef) * self.LossL1(data['I'][:, 0, :-8, :], drec['I'][:, 0, :-8, :])
        ### marker velocity loss
        loss_m_velocity = 90. * (1. - self.cfg.kl_coef) * self.LossL1(data['I'][:, 1:, :, :], drec['I'][:, 1:, :, :])
        ### foot ground contact loss
        loss_fg_contact = 60. * (1. - self.cfg.kl_coef) * self.LossL2(data['I'][:, 0, -8:, :], drec['I'][:, 0, -8:, :])

        ### KL loss traj
        q_z_traj = torch.distributions.normal.Normal(drec['mean_traj'], drec['std_traj'])
        p_z_traj = torch.distributions.normal.Normal(
            loc=torch.tensor(np.zeros([self.cfg.batch_size, 512]), requires_grad=False).to(
                self.device).type(self.dtype),
            scale=torch.tensor(np.ones([self.cfg.batch_size, 512]), requires_grad=False).to(
                self.device).type(self.dtype)
        )
        loss_kl_traj = self.cfg.kl_coef * torch.mean(torch.sum(torch.distributions.kl.kl_divergence(q_z_traj, p_z_traj)))
        ### KL loss local
        q_z_local = torch.distributions.normal.Normal(drec['mean_local'], drec['std_local'])
        p_z_local = torch.distributions.normal.Normal(
            loc=torch.tensor(np.zeros(drec['mean_local'].shape), requires_grad=False).to(
                self.device).type(self.dtype),
            scale=torch.tensor(np.ones(drec['mean_local'].shape), requires_grad=False).to(
                self.device).type(self.dtype)
        )
        loss_kl_local = 0.1 * self.cfg.kl_coef * torch.mean(torch.sum(torch.distributions.kl.kl_divergence(q_z_local, p_z_local)))

        loss_dict = {
            'loss_traj': loss_traj,
            'loss_root_velocity': loss_root_velocity,
            'loss_marker': loss_marker,
            'loss_m_velocity': loss_m_velocity,
            'loss_fg_contact': loss_fg_contact,
            'loss_kl_traj': loss_kl_traj,
            'loss_kl_local': loss_kl_local
        }

        loss_total = torch.stack(list(loss_dict.values())).sum()
        loss_dict['loss_total'] = loss_total

        return loss_total, loss_dict

    def evaluate(self):
        self.motionFill.eval()

        eval_loss_dict_motionFill = {}

        dataset = self.ds_val

        with torch.no_grad():
            for data in dataset:
                data = {k: data[k].to(self.device) for k in data.keys()}
                drec_motionFill = self.motionFill(**data)
                loss_total_motionFill, cur_loss_dict_motionFill = self.loss_motionFill(data, drec_motionFill)
                eval_loss_dict_motionFill = {k: eval_loss_dict_motionFill.get(k, 0.0) + v.item() for k, v in cur_loss_dict_motionFill.items()}

            eval_loss_dict_motionFill = {k: v / len(dataset) for k, v in eval_loss_dict_motionFill.items()}

        return eval_loss_dict_motionFill

    def fit(self, n_epochs=None):
        starttime = datetime.now().replace(microsecond=0)
        if n_epochs is None:
            n_epochs = self.cfg.n_epochs

        self.logger('Started Training at %s for %d epochs' % (datetime.strftime(starttime, '%Y-%m-%d_%H:%M:%S'), n_epochs))

        prev_lr_motionFill = np.inf
        self.fit_motionFill = True

        lr_scheduler_motionFill = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_motioinFill, 'min')

        for epoch_num in range(1, n_epochs + 1):
            self.logger('--- starting Epoch # %03d' % epoch_num)

            train_loss_dict_motionFill = self.train()
            eval_loss_dict_motionFill = self.evaluate()

            if self.fit_motionFill:
                lr_scheduler_motionFill.step(eval_loss_dict_motionFill['loss_total'])
                cur_lr_motionFill = self.optimizer_motioinFill.param_groups[0]['lr']

                if cur_lr_motionFill != prev_lr_motionFill:
                    self.logger('--- Motion Infilling learning rate changed from %.2e to %.2e ---' % (prev_lr_motionFill, cur_lr_motionFill))
                    prev_lr_motionFill = cur_lr_motionFill

                with torch.no_grad():
                    eval_msg = Trainer.create_loss_message(eval_loss_dict_motionFill, expr_ID=self.cfg.expr_ID,
                                                           epoch_num=self.epochs_completed, it=len(self.ds_val),
                                                           model_name='Motion Infilling',
                                                           try_num=self.try_num, mode='evald')
                    if eval_loss_dict_motionFill['loss_total'] < self.best_loss_motionFill:

                        self.cfg.best_motionFill = makepath(os.path.join(self.cfg.work_dir, 'snapshots',
                                                                   'TR%02d_E%03d_motionFill.pt' % (
                                                                   self.try_num, self.epochs_completed)), isfile=True)
                        self.save_motionFill()
                        self.logger(eval_msg + ' ** ')
                        self.best_loss_motionFill = eval_loss_dict_motionFill['loss_total']

                    else:
                        self.logger(eval_msg)
                        if self.epochs_completed % 3 == 0:
                            self.save_motionFill()

                    self.swriter.add_scalars('total_loss_motionFill/scalars',
                                             {'train_loss_total': train_loss_dict_motionFill['loss_total'],
                                              'evald_loss_total': eval_loss_dict_motionFill['loss_total'], },
                                             self.epochs_completed)

            self.epochs_completed += 1

        endtime = datetime.now().replace(microsecond=0)

        self.logger('Finished Training at %s\n' % (datetime.strftime(endtime, '%Y-%m-%d_%H:%M:%S')))
        self.logger(
            'Training done in %s!\n' % (endtime - starttime))
        self.logger('Best Motion Infilling val total loss achieved: %.2e\n' % (self.best_loss_motionFill))
        self.logger('Best Motion Infilling model path: %s\n' % self.cfg.best_motionFill)

    def save_motionFill(self):
        torch.save(self.motionFill.module.state_dict()
                   if isinstance(self.motionFill, torch.nn.DataParallel)
                   else self.motionFill.state_dict(), self.cfg.best_motionFill)

    @staticmethod
    def create_loss_message(loss_dict, expr_ID='XX', epoch_num=0,model_name='Motion Infilling', it=0, try_num=0, mode='evald'):
        ext_msg = ' | '.join(['%s = %.2e' % (k, v) for k, v in loss_dict.items() if k != 'loss_total'])
        return '[%s]_TR%02d_E%03d - It %05d - %s - %s: [T:%.2e] - [%s]' % (
            expr_ID, try_num, epoch_num, it,model_name, mode, loss_dict['loss_total'], ext_msg)
