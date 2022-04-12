import numpy as np
import torch
import os, glob
import cv2
import smplx
import argparse
from torch.utils.data import DataLoader

from tools.meshviewer import Mesh, MeshViewer, colors
from tools.utils import to_cpu
from tools.utils import euler
from tools.utils import rotmat2aa
from tools.cfg_parser import Config
from data.dataloader import LoadData
from models.models import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_npz(sequence, allow_pickle=True):
    seq_data = np.load(sequence, allow_pickle=allow_pickle)
    data = {}
    for k in seq_data.files:
        data[k] = seq_data[k]
    return data


def load_torch(dataset_dir='datasets_parsed_motion_infill', ds_name='train_data_all', batch_size=32):
    ds = LoadData(dataset_dir, ds_name=ds_name)
    ds = DataLoader(ds, batch_size=batch_size, num_workers=8, shuffle=True, drop_last=True)
    return ds


def get_interpolated_roots(N, startFrame, FrameT):
    res = np.zeros((N + 1, 3))
    unit_offset = (FrameT - startFrame) / (N - 1)
    for i in range(N + 1):
        res[i] = startFrame + i * unit_offset
    return res


def render_img(cfg):
    motionFill = MotionFill().to(device)
    motionFill.load_state_dict(torch.load(cfg.model_path, map_location=torch.device('cpu')))
    motionFill.eval()

    mv = MeshViewer(offscreen=True)

    # set the camera pose
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = euler([80, 15, 0], 'xzx')
    camera_pose[:3, 3] = np.array([0.8, -1.8, 1.5])
    mv.update_camera_pose(camera_pose)

    '''
    ds = load_torch()
    batch_size = 32
    data = next(iter(ds))
    dist_I = torch.distributions.normal.Normal(
        loc=torch.tensor(np.zeros([batch_size, 256, 10, 5]), requires_grad=False),
        scale=torch.tensor(np.ones([batch_size, 256, 10, 5]), requires_grad=False)
    )
    Zs_I = dist_I.rsample().float()
    I = data['I']
    I_predict = motionFill.localMotionNet.decode(Zs_I, data['I_cond'])
    # I_predict = (motionFill(**data))['I']
    _, _, P, T = I_predict.shape
    L = 99*3
    I0 = (np.transpose(to_cpu(I_predict), (0, 3, 2, 1)))[0]
    marker_location = I0[:, :, 0][:, :L]
    # marker_location = I_predict[0, 0, :, :].reshape(T, P)[:, :L]
    # marker_location = to_cpu(marker_location.reshape(T, 99, 3))

    dist_traj = torch.distributions.normal.Normal(
        loc=torch.tensor(np.zeros([batch_size, 512]), requires_grad=False),
        scale=torch.tensor(np.ones([batch_size, 512]), requires_grad=False)
    )
    Zs_traj = dist_traj.rsample().float()
    traj_predict = motionFill.trajNet.decode(Zs_traj, data['traj_cond']) + data['traj_cond']
    '''


    data = parse_npz(cfg.data_path)
    T = data['verts_markers'].shape[0]

    interp_traj = get_interpolated_roots(T, data['roots'][0], data['roots'][T-1])
    traj_cond = np.expand_dims(interp_traj.reshape((T+1)*3), axis=0)
    traj_cond = torch.tensor(traj_cond).float()
    dist_traj = torch.distributions.normal.Normal(
            loc=torch.tensor(np.zeros([1, 512]), requires_grad=False),
            scale=torch.tensor(np.ones([1, 512]), requires_grad=False)
        )
    Zs_traj = dist_traj.rsample().float()
    traj_predict = motionFill.trajNet.decode(Zs_traj, traj_cond)
    traj_predict = traj_predict.reshape(T+1, 3)
    velocity_predict = traj_predict[1:, :] - traj_predict[:-1, :]

    I1 = np.concatenate([data['verts_markers'].reshape(T, 99*3), data['fg_contact']], axis=1)
    P = I1.shape[1]
    I2to4 = np.tile(data['velocity'], (1, P)).reshape(150, P, 3)
    I = np.zeros([T, P, 4])
    I[:,:,0] = I1
    I[:, :, 1:] = I2to4
    I = np.transpose(I, (2, 1, 0))
    I = torch.tensor(np.expand_dims(I, axis=0)).float()

    I_cond = np.zeros([T, P, 4])
    L = len(data['verts_markers'][0].flatten())
    I_cond[0, :L, 0] = data['verts_markers'][0].flatten()
    I_cond[-1, :L, 0] = data['verts_markers'][-1].flatten()
    I2to4_cond = np.tile(velocity_predict.detach().numpy(), (1, P)).reshape(T, P, 3)
    I_cond[:, :, 1:] = I2to4_cond
    # I_cond[:, :, 1:] = I2to4
    I_cond = np.transpose(I_cond, (2, 1, 0))
    I_cond = torch.tensor(np.expand_dims(I_cond, axis=0)).float()
    dist_I = torch.distributions.normal.Normal(
            loc=torch.tensor(np.zeros([1, 256, 10, 5]), requires_grad=False),
            scale=torch.tensor(np.ones([1, 256, 10, 5]), requires_grad=False)
        )
    Zs_I = dist_I.rsample().float()
    I_predict = motionFill.decode(Zs_I, I_cond)
    I_prime = np.transpose(to_cpu(I_predict), (0, 3, 2, 1))
    I0 = np.transpose(to_cpu(I), (0, 3, 2, 1))
    marker_location = I_prime[0, :, :, 0].reshape(T, P)[:, :L]


    LossL1 = torch.nn.L1Loss(reduction='mean')
    loss_marker = 200. * (1. - 1e-2) * LossL1(I[:, 0, :-8, :], I_predict[:, 0, :-8, :])
    print(loss_marker)

    ### traj loss
    # loss_traj = 90. * (1. - 1e-2) * LossL1(data['roots'], traj_predict)
    # print(loss_traj)

    # I_forward, _, _ = motionFill.localMotionNet(I, data['I_cond'])
    # loss_marker = 200. * (1. - 1e-2) * LossL1(I[:, 0, :-8, :], I_forward[:, 0, :-8, :])
    # print(loss_marker)

    suj_id = 's1'
    if not os.path.exists(os.path.join(cfg.renderings, suj_id)):
        os.makedirs(os.path.join(cfg.renderings, suj_id))

    for i in range(T):
        s_mesh = Mesh(vertices=marker_location[i], faces=None, vc=colors['white'])

        mv.set_static_meshes([s_mesh])

        color, depth = mv.viewer.render(mv.scene)
        img = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        img_save_path = os.path.join(cfg.renderings, suj_id)
        img_name = 'cup_' + str(i) + '.jpg'
        cv2.imwrite(os.path.join(img_save_path, img_name), img)

    # mv.close_viewer()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Render GNet Poses')
    parser.add_argument('--model-path', required=True, type=str,
                        help='Path to the trained model')
    parser.add_argument('--data-path', type=str,
                        help='Path to the data to be tested')
    parser.add_argument('--renderings', default='renderings_motion_infill', type=str,
                        help='Path to the directory saving the renderings')

    args = parser.parse_args()
    data_path = args.data_path
    model_path = args.model_path
    renderings = args.renderings

    cfg = {
        'tool_meshes': 'datasets/',
        'smplx_path': 'models',
        'model_path': model_path,
        'vtemp': 'tools/subject_meshes/male/s1.ply',
        'gender': 'male',
        'data_path': data_path,
        'renderings': renderings
    }

    cfg = Config(**cfg)

    if not os.path.exists(cfg.renderings):
        os.makedirs(cfg.renderings)

    render_img(cfg)
