import sys
sys.path.append('.')
sys.path.append('..')
import os
import argparse
from tools.cfg_parser import Config
from train.trainer import Trainer


if __name__=='__main__':

    parser = argparse.ArgumentParser(description='GNet-Training')

    parser.add_argument('--work-dir', required=True, type=str,
                        help='Saving path')

    parser.add_argument('--data-path', required=True, type=str,
                        help='The path to the folder that contains GNet data')

    parser.add_argument('--expr-ID', default='V00', type=str,
                        help='Training ID')

    parser.add_argument('--batch-size', default=256, type=int,
                        help='Training batch size')

    parser.add_argument('--n-workers', default=8, type=int,
                        help='Number of PyTorch dataloader workers')

    parser.add_argument('--lr', default=5e-4, type=float,
                        help='Training learning rate')

    parser.add_argument('--kl-coef', default=5e-3, type=float,
                        help='KL divergence coefficent for GNet training')

    parser.add_argument('--use-multigpu', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='If to use multiple GPUs for training')

    args = parser.parse_args()

    work_dir = args.work_dir
    data_path = args.data_path
    expr_ID = args.expr_ID
    n_workers = args.n_workers
    use_multigpu = args.use_multigpu
    batch_size = args.batch_size
    base_lr = args.lr
    kl_coef = args.kl_coef

    cfg = {
        'work_dir': work_dir,
        'ds_train': 'train_data',
        'ds_val': 'val_data',
        'batch_size': batch_size,
        'n_workers': n_workers,
        'use_multigpu': use_multigpu,
        'kl_coef': kl_coef,
        'dataset_dir': data_path,
        'expr_ID': expr_ID,
        'base_lr': base_lr,
        'n_epochs': 100,
        'latentD': 16,
        'save_every_it': 10,
        'best_gnet': None,
        'try_num': 0,
        'reg_coef': 5e-4
    }

    cfg = Config(**cfg)
    trainer = Trainer(cfg=cfg)

    trainer.fit()