import numpy as np
import torch
import os, glob
import cv2
import smplx
import argparse
from tqdm import tqdm

from models.models import GNet

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
        if k in ['bps_dists', 'obj_transl']:
            data[k] = seq_data[k]
        else:
            data[k] = seq_data[k].item()
    return data


def render_img(cfg):
    gnet = GNet()
    gnet.load_state_dict(torch.load(cfg.model_path))
    gnet.eval()

    mv = MeshViewer(offscreen=True)

    # set the camera pose
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = euler([80, 15, 0], 'xzx')
    camera_pose[:3, 3] = np.array([0.8, -1.8, 1.5])
    mv.update_camera_pose(camera_pose)

    all_seqs = glob.glob((os.path.join(cfg.data_path, '*/cubem*.npz')))

    for sequence in tqdm(all_seqs):

        test_data = parse_npz(sequence)

        bps_dists = torch.tensor(test_data['bps_dists'].reshape(1, 1024))
        object_transl = torch.tensor(test_data['obj_transl'].reshape(1, 3))
        dist = torch.distributions.normal.Normal(
                loc=torch.tensor(np.zeros([1, 16]), requires_grad=False),
                scale=torch.tensor(np.ones([1, 16]), requires_grad=False)
            )
        Zs = dist.rsample().float()

        results = gnet.decode(Zs, bps_dists, object_transl)
        fullpose = rotmat2aa(results['fullpose_rotmat'])
        fullpose = fullpose.reshape(1, 165)

        sbj_mesh = os.path.join(cfg.tool_meshes, test_data['body']['vtemp'])
        sbj_vtemp = np.array(Mesh(filename=sbj_mesh).vertices)

        sbj_m = smplx.create(model_path=cfg.smplx_path,
                             model_type='smplx',
                             gender=test_data['gender'],
                             use_pca=False,
                             # num_pca_comps=test_data['n_comps'],
                             v_template=sbj_vtemp,
                             batch_size=1)

        sbj_parms = {
            'global_orient': fullpose[:,:3].float(),
            'body_pose': fullpose[:,3:66].float(),
            'jaw_pose': fullpose[:,66:69].float(),
            'leye_pose': fullpose[:,69:72].float(),
            'reye_pose': fullpose[:,72:75].float(),
            'left_hand_pose': fullpose[:,75:120].float(),
            'right_hand_pose': fullpose[:,120:165].float(),
            'transl': results['body_transl'].reshape(1, 3) + object_transl
        }
        # sbj_parms['left_hand_pose'] = torch.einsum('bi,ji->bj', [sbj_parms['left_hand_pose'], sbj_m.left_hand_components])
        # sbj_parms['right_hand_pose'] = torch.einsum('bi,ji->bj', [sbj_parms['right_hand_pose'], sbj_m.right_hand_components])
        verts_sbj = to_cpu(sbj_m(**sbj_parms).vertices)

        obj_mesh = os.path.join(cfg.tool_meshes, test_data['object']['object_mesh'])
        obj_mesh = Mesh(filename=obj_mesh)
        obj_vtemp = np.array(obj_mesh.vertices)
        obj_m = ObjectModel(v_template=obj_vtemp,
                            batch_size=1)
        test_data['object']['params']['transl'] += to_cpu(object_transl)
        obj_parms = params2torch(test_data['object']['params'])
        verts_obj = to_cpu(obj_m(**obj_parms).vertices)

        s_mesh = Mesh(vertices=verts_sbj[0], faces=sbj_m.faces, vc=colors['pink'], smooth=True)
        o_mesh = Mesh(vertices=verts_obj[0], faces=obj_mesh.faces, vc=colors['yellow'])

        mv.set_static_meshes([o_mesh, s_mesh])

        if not os.path.exists(os.path.join(cfg.renderings, test_data['sbj_id'])):
            os.makedirs(os.path.join(cfg.renderings, test_data['sbj_id']))

        color, depth = mv.viewer.render(mv.scene)
        img = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        img_save_path = os.path.join(cfg.renderings, test_data['sbj_id'])
        img_name = (sequence.split('/')[-1]).split('.')[-2] + '.jpg'
        cv2.imwrite(os.path.join(img_save_path, img_name), img)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Render GNet Poses')

    parser.add_argument('--model-path', required=True, type=str,
                        help='Path to the saved GNet model')
    parser.add_argument('--data-path', required=True, type=str,
                        help='Path to the test data file')
    parser.add_argument('--renderings', default='renderings', type=str,
                        help='Path to the directory saving the renderings')

    args = parser.parse_args()

    model_path = args.model_path
    data_path = args.data_path
    renderings = args.renderings

    cfg = {
        'tool_meshes': 'toolMeshes',
        'smplx_path': 'smplx_models',
        'model_path': model_path,
        'data_path': data_path,
        'renderings': renderings
    }

    cfg = Config(**cfg)

    if not os.path.exists(cfg.renderings):
        os.makedirs(cfg.renderings)

    render_img(cfg)