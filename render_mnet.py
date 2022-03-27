import numpy as np
import torch
import os, glob
import cv2
import smplx
import argparse
from tqdm import tqdm
import time

from models.models import *

from tools.objectmodel import ObjectModel
from tools.meshviewer import Mesh, MeshViewer, colors
from tools.utils import params2torch
from tools.utils import to_cpu
from tools.utils import euler
from tools.utils import rotmat2aa
from tools.cfg_parser import Config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def reshape_seq_data(seq_data):
    for k1 in ['body']:
        for k2 in seq_data[k1]['params']:
            seq_data[k1]['params'][k2] = seq_data[k1]['params'][k2].reshape(1, len(seq_data[k1]['params'][k2]))
    return seq_data

def parse_npz(sequence, allow_pickle=True):
    seq_data = np.load(sequence, allow_pickle=allow_pickle)
    data = {}
    for k in seq_data.files:
        data[k] = torch.tensor(np.expand_dims(seq_data[k], axis=0))
    return data


def render_img(cfg):
    mnet = MotionNet(np.random.RandomState(), use_cuda=False)
    mnet.load_state_dict(torch.load(cfg.model_path, map_location=torch.device('cpu')))
    mnet.eval()

    mv = MeshViewer(offscreen=True)

    # set the camera pose
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = euler([80, 15, 0], 'xzx')
    camera_pose[:3, 3] = np.array([0.8, -1.8, 1.5])
    mv.update_camera_pose(camera_pose)

    data = parse_npz(cfg.data_path)
    T = 180
    fullpose = None
    transl = None

    for i in range(T):
        print(i)
        if fullpose is None:
            fullpose = data['past_pose_rotmat']
            data['past_pose_rotmat'] = data['future_pose']
            data['past_transl'] = data['future_transl']
            data['cur_verts'] = data['future_verts']
            data['cur_dists_to_goal'] = data['future_dists']
        else:
            fullpose = torch.cat([fullpose, data['past_pose_rotmat']], dim=0)
        if transl is None:
            transl = data['past_transl']
        else:
            transl = torch.cat([transl, data['past_transl']], dim=0)

        results = mnet(**data)
        data['past_pose_rotmat'] = results['future_pose']
        data['past_transl'] = results['future_transl']
        data['cur_verts'] = results['future_verts']
        data['cur_dists_to_goal'] = results['future_dists']

    sbj_mesh = os.path.join(cfg.tool_meshes, cfg.vtemp)
    sbj_vtemp = np.array(Mesh(filename=sbj_mesh).vertices)

    sbj_m = smplx.create(model_path=cfg.smplx_path,
                         model_type='smplx',
                         gender=cfg.gender,
                         use_pca=False,
                         # num_pca_comps=test_data['n_comps'],
                         v_template=sbj_vtemp,
                         batch_size=T)

    fullpose = rotmat2aa(fullpose)
    fullpose = fullpose.reshape(T, 165)
    sbj_parms = {
        'global_orient': fullpose[:, :3].float(),
        'body_pose': fullpose[:, 3:66].float(),
        'jaw_pose': fullpose[:, 66:69].float(),
        'leye_pose': fullpose[:, 69:72].float(),
        'reye_pose': fullpose[:, 72:75].float(),
        'left_hand_pose': fullpose[:, 75:120].float(),
        'right_hand_pose': fullpose[:, 120:165].float(),
        'transl': transl.reshape(T, 3)
    }
    verts_sbj = to_cpu(sbj_m(**sbj_parms).vertices)

    obj_mesh = os.path.join(cfg.tool_meshes, cfg.object_mesh)
    obj_mesh = Mesh(filename=obj_mesh)
    obj_vtemp = np.array(obj_mesh.vertices)
    obj_m = ObjectModel(v_template=obj_vtemp,
                        batch_size=T)
    obj_params = {}
    obj_orient = np.array([-0.05220879,  0.10871532, -1.25274907])
    obj_params['global_orient'] = np.tile(np.expand_dims(obj_orient, axis=0), (T, 1))
    print(data['obj_goal_transl'])
    obj_params['transl'] = np.tile(data['obj_goal_transl'], (T, 1))
    obj_parms = params2torch(obj_params)
    print(obj_parms['global_orient'].shape)
    print(obj_parms['transl'].shape)
    verts_obj = to_cpu(obj_m(**obj_parms).vertices)

    suj_id = 's8'
    if not os.path.exists(os.path.join(cfg.renderings, suj_id)):
        os.makedirs(os.path.join(cfg.renderings, suj_id))

    for i in range(T):
        s_mesh = Mesh(vertices=verts_sbj[i], faces=sbj_m.faces, vc=colors['pink'], smooth=True)
        o_mesh = Mesh(vertices=verts_obj[i], faces=obj_mesh.faces, vc=colors['yellow'])

        mv.set_static_meshes([o_mesh, s_mesh])
        #time.sleep(1)

        color, depth = mv.viewer.render(mv.scene)
        img = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        img_save_path = os.path.join(cfg.renderings, suj_id)
        img_name = 'cup_' + str(i) + '.jpg'
        cv2.imwrite(os.path.join(img_save_path, img_name), img)

    # mv.close_viewer()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Render GNet Poses')

    parser.add_argument('--model-path', required=True, type=str,
                        help='Path to the saved GNet model')
    parser.add_argument('--data-path', required=True, type=str,
                        help='Path to the data to be tested')
    parser.add_argument('--renderings', default='renderings_mnet', type=str,
                        help='Path to the directory saving the renderings')

    args = parser.parse_args()

    model_path = args.model_path
    data_path = args.data_path
    renderings = args.renderings

    cfg = {
        'tool_meshes': 'toolMeshes',
        'smplx_path': 'smplx_models',
        'vtemp': 'tools/subject_meshes/male/s8.ply',
        'object_mesh': 'tools/object_meshes/contact_meshes/cup.ply',
        'gender': 'male',
        'model_path': model_path,
        'data_path': data_path,
        'renderings': renderings
    }

    cfg = Config(**cfg)

    if not os.path.exists(cfg.renderings):
        os.makedirs(cfg.renderings)

    render_img(cfg)