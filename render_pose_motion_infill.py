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
from tools.utils import CRot2rotmat
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


    ds = load_torch(ds_name=cfg.ds_name)
    batch_size = 32
    data = next(iter(ds))
    dist_I = torch.distributions.normal.Normal(
        loc=torch.tensor(np.zeros([batch_size, 256, 11, 5]), requires_grad=False),
        scale=torch.tensor(np.ones([batch_size, 256, 11, 5]), requires_grad=False)
    )
    Zs_I = dist_I.rsample().float()
    I = data['I']
    I_predict = motionFill.localMotionNet.decode(Zs_I, data['I_cond'])
    # I_predict = (motionFill(**data))['I']
    _, _, P, T = I_predict.shape
    L = 55*6
    I0 = (np.transpose(to_cpu(I_predict), (0, 3, 2, 1)))[0]
    fullpose_6D = I0[:, :, 0][:, :L]
    # fullpose_6D = ((np.transpose(to_cpu(I), (0, 3, 2, 1)))[2])[:, :, 0][:, :L]
    fullpose_rotmat = torch.zeros((T, 55, 3, 3))
    for i in range(T):
        fullpose_rotmat[i] = CRot2rotmat(torch.tensor(fullpose_6D[[i]]))
    fullpose_rotmat = fullpose_rotmat.reshape(T, 1, 55, 9)
    fullpose = rotmat2aa(fullpose_rotmat).reshape(T, 165)
    # fullpose_6D = I_predict[0, 0, :, :].reshape(T, P)[:, :L]
    # fullpose_6D = to_cpu(fullpose_6D.reshape(T, 55, 6))

    dist_traj = torch.distributions.normal.Normal(
        loc=torch.tensor(np.zeros([batch_size, 512]), requires_grad=False),
        scale=torch.tensor(np.ones([batch_size, 512]), requires_grad=False)
    )
    Zs_traj = dist_traj.rsample().float()
    traj_predict = motionFill.trajNet.decode(Zs_traj, data['traj_cond']) + data['traj_cond']
    traj = traj_predict[0].reshape(T+1, 3)
    velocity = traj[1:T+1] - traj[:T]
    traj = traj[:-1,:]

    sbj_parms = {
        'global_orient': fullpose[:, :3].float(),
        'body_pose': fullpose[:, 3:66].float(),
        'jaw_pose': fullpose[:, 66:69].float(),
        'leye_pose': fullpose[:, 69:72].float(),
        'reye_pose': fullpose[:, 72:75].float(),
        'left_hand_pose': fullpose[:, 75:120].float(),
        'right_hand_pose': fullpose[:, 120:165].float(),
        'transl': (data['traj'][0]).reshape(T+1, 3)[:-1,:]
        # 'transl': traj
    }

    LossL1 = torch.nn.L1Loss(reduction='mean')
    loss_marker = 200. * (1. - 1e-2) * LossL1(I[:, 0, :-8, :], I_predict[:, 0, :-8, :])

    ### traj loss
    # loss_traj = 90. * (1. - 1e-2) * LossL1(data['roots'], traj_predict)
    # print(loss_traj)

    # I_forward, _, _ = motionFill.localMotionNet(I, data['I_cond'])
    # loss_marker = 200. * (1. - 1e-2) * LossL1(I[:, 0, :-8, :], I_forward[:, 0, :-8, :])
    # print(loss_marker)

    sbj_mesh = os.path.join(cfg.tool_meshes, cfg.vtemp)
    sbj_vtemp = np.array(Mesh(filename=sbj_mesh).vertices)

    sbj_m = smplx.create(model_path=cfg.smplx_path,
                         model_type='smplx',
                         gender=cfg.gender,
                         use_pca=False,
                         # num_pca_comps=test_data['n_comps'],
                         v_template=sbj_vtemp,
                         batch_size=T)
    verts_sbj = to_cpu(sbj_m(**sbj_parms).vertices)

    suj_id = 's1'
    if not os.path.exists(os.path.join(cfg.renderings, suj_id)):
        os.makedirs(os.path.join(cfg.renderings, suj_id))

    for i in range(T):
        s_mesh = Mesh(vertices=verts_sbj[i], faces=sbj_m.faces, vc=colors['pink'], smooth=True)

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
    parser.add_argument('--renderings', default='renderings_pose_motion_infill', type=str,
                        help='Path to the directory saving the renderings')
    parser.add_argument('--ds-name', default='train_data_all', type=str,
                        help='File name of the dataset')

    args = parser.parse_args()
    data_path = args.data_path
    model_path = args.model_path
    renderings = args.renderings
    ds_name = args.ds_name

    cfg = {
        'tool_meshes': 'toolMeshes',
        'smplx_path': 'smplx_models',
        'model_path': model_path,
        'vtemp': 'tools/subject_meshes/male/s1.ply',
        'gender': 'male',
        'data_path': data_path,
        'renderings': renderings,
        'ds_name': ds_name
    }

    cfg = Config(**cfg)

    if not os.path.exists(cfg.renderings):
        os.makedirs(cfg.renderings)

    render_img(cfg)
