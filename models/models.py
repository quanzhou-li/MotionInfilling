# Adapted from GrabNet

import sys

sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch

from torch import nn
from torch.nn import functional as F
from tools.utils import CRot2rotmat

class ResBlock(nn.Module):

    def __init__(self,
                 Fin,
                 Fout,
                 n_neurons=512):
        super(ResBlock, self).__init__()
        # Feature dimension of input and output
        self.Fin = Fin
        self.Fout = Fout

        self.fc1 = nn.Linear(Fin, n_neurons)
        self.bn1 = nn.BatchNorm1d(n_neurons)

        self.fc2 = nn.Linear(n_neurons, Fout)
        self.bn2 = nn.BatchNorm1d(Fout)

        if Fin != Fout:
            self.fc3 = nn.Linear(Fin, Fout)

        self.ll = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, final_nl=True):
        Xin = x if self.Fin == self.Fout else self.ll(self.fc3(x))

        Xout = self.fc1(x)
        Xout = self.bn1(Xout)
        Xout = self.ll(Xout)

        Xout = self.fc2(Xout)
        Xout = self.bn2(Xout)
        Xout = Xin + Xout

        if final_nl:
            return self.ll(Xout)
        return Xout


class GNet(nn.Module):
    def __init__(self,
                 in_condition=1024+3,
                 in_params=1632,
                 n_neurons=512,
                 latentD=16,
                 **kwargs):
        super(GNet, self).__init__()
        self.latentD = latentD

        self.enc_bn0 = nn.BatchNorm1d(in_condition)
        self.enc_bn1 = nn.BatchNorm1d(in_condition + in_params)
        self.enc_rb1 = ResBlock(in_condition + in_params, n_neurons)
        self.enc_rb2 = ResBlock(in_condition + in_params + n_neurons, n_neurons)

        self.enc_mu = nn.Linear(n_neurons, latentD)
        self.enc_var = nn.Linear(n_neurons, latentD)
        self.do = nn.Dropout(p=.1, inplace=False)

        self.dec_bn1 = nn.BatchNorm1d(in_condition)
        self.dec_rb1 = ResBlock(latentD + in_condition, n_neurons)
        self.dec_rb2 = ResBlock(n_neurons + latentD + in_condition, n_neurons)

        self.dec_pose = nn.Linear(n_neurons, 55 * 6)  # Theta
        self.dec_trans = nn.Linear(n_neurons, 3)  # body translation
        self.dec_ver = nn.Linear(n_neurons, 400 * 3)  # vertices locations
        self.dec_dis = nn.Linear(n_neurons, 99)  # hand-object distances

    def encode(self, fullpose_rotmat, body_transl, verts, hand_object_dists, bps_dists, object_transl):
        '''
        :param fullpose_rotmat: N * 1 * 55 * 9
        :param body_transl: N * 3
        :param verts: N * 400 * 3
        :param dists: N * 99
        :param bps_dists: N * 1024
        :param object_transl: N * 3
        :return:
        '''
        bs = fullpose_rotmat.shape[0]

        # Get 6D rotation representation of fullpose_rotmat
        fullpose_6D = (fullpose_rotmat.reshape(bs, 1, 55, 3, 3))[:,:,:,:,:2]
        fullpose_6D = fullpose_6D.reshape(bs, 55*6)

        X = torch.cat([fullpose_6D, body_transl, verts.flatten(start_dim=1), hand_object_dists, bps_dists, object_transl], dim=1)

        X0 = self.enc_bn1(X)
        X = self.enc_rb1(X0)
        X = self.enc_rb2(torch.cat([X0, X], dim=1))

        return torch.distributions.normal.Normal(self.enc_mu(X), F.softplus(self.enc_var(X)))

    def decode(self, Zin, bps_dists, object_transl):
        bs = Zin.shape[0]

        condition = self.dec_bn1(torch.cat([bps_dists, object_transl], dim=1))

        X0 = torch.cat([Zin, condition], dim=1)
        X = self.dec_rb1(X0)
        X = self.dec_rb2(torch.cat([X0, X], dim=1))

        fullpose_6D = self.dec_pose(X)
        body_transl = self.dec_trans(X)
        verts = self.dec_ver(X)
        hand_object_dists = self.dec_dis(X)

        fullpose_rotmat = CRot2rotmat(fullpose_6D).reshape(bs, 1, 55, 9)

        return {'fullpose_rotmat': fullpose_rotmat, 'body_transl': body_transl,
                'verts': verts, 'hand_object_dists': hand_object_dists}

    def forward(self, fullpose_rotmat, body_transl, verts, hand_object_dists, bps_dists, object_transl, **kwargs):
        z = self.encode(fullpose_rotmat, body_transl, verts, hand_object_dists, bps_dists, object_transl)
        z_s = z.rsample()

        params = self.decode(z_s, bps_dists, object_transl)
        results = {'mean': z.mean, 'std': z.scale}
        results.update(params)

        return results


class TrajFill(nn.Module):
    def __init__(self,
                 Fin=2*(151*3),
                 Fout=151*3,
                 featureD=151*4,
                 n_hidden=151*4,
                 latentD=512,
                 **kwargs):
        super(TrajFill, self).__init__()

        self.rb1 = ResBlock(Fin, featureD, n_neurons=n_hidden)
        self.rb2 = ResBlock(featureD, featureD, n_neurons=n_hidden)
        self.enc_mu = nn.Linear(featureD, latentD)
        self.enc_var = nn.Linear(featureD, latentD)
        self.rb3 = ResBlock(latentD+Fout, Fout, n_neurons=Fout)
        self.rb4 = ResBlock(Fout, Fout, n_neurons=Fout)

    def encode(self, traj, traj_cond):
        X = torch.cat([traj_cond, traj], dim=1)
        X = self.rb1(X)
        X = self.rb2(X)
        return torch.distributions.normal.Normal(self.enc_mu(X), F.softplus(self.enc_var(X)))

    def decode(self, Zs, traj_cond):
        X = torch.cat([Zs, traj_cond], dim=1)
        X = self.rb3(X)
        X = self.rb4(X)
        return X

    def forward(self, traj, traj_cond, **kwargs):
        Z = self.encode(traj, traj_cond)
        z_s = Z.rsample()

        diff = self.decode(z_s, traj_cond)
        return traj_cond + diff, Z.mean, Z.scale



class EncBlock(nn.Module):
    def __init__(self, nin, nout, downsample=True, kernel=3):
        super(EncBlock, self).__init__()
        self.downsample = downsample
        padding = kernel // 2

        self.main = nn.Sequential(
            nn.Conv2d(in_channels=nin, out_channels=nout, kernel_size=kernel, stride=1, padding=padding),
            nn.BatchNorm2d(nout, track_running_stats=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=nout, out_channels=nout, kernel_size=kernel, stride=1, padding=padding),
            nn.BatchNorm2d(nout, track_running_stats=False),
            nn.LeakyReLU(0.2),
        )

        if self.downsample:
            self.pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.pooling = nn.MaxPool2d(kernel_size=(3,3), stride=(2, 1), padding=1)

    def forward(self, input):
        output = self.main(input)
        output = self.pooling(output)
        return output


class DecBlock(nn.Module):
    def __init__(self, nin, nout, upsample=True, kernel=3):
        super(DecBlock, self).__init__()
        self.upsample = upsample

        padding = kernel // 2
        if upsample:
            self.deconv1 = nn.ConvTranspose2d(in_channels=nin, out_channels=nout, kernel_size=kernel, stride=2, padding=padding)
        else:
            self.deconv1 = nn.ConvTranspose2d(in_channels=nin, out_channels=nout, kernel_size=kernel, stride=(2, 1), padding=padding)
        self.deconv2 = nn.ConvTranspose2d(in_channels=nout, out_channels=nout, kernel_size=kernel, stride=1, padding=padding)
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.BN1 = nn.BatchNorm2d(nout, track_running_stats=False)
        self.BN2 = nn.BatchNorm2d(nout, track_running_stats=False)

    def forward(self, input, out_size):
        output = self.deconv1(input, output_size=out_size)
        output = self.leaky_relu(self.BN1(output))
        output = self.leaky_relu(self.BN2(self.deconv2(output)))
        return output


class DecBlock_output(nn.Module):
    def __init__(self, nin, nout, upsample=True, kernel=3):
        super(DecBlock_output, self).__init__()
        self.upsample = upsample
        padding = kernel // 2

        if upsample:
            self.deconv1 = nn.ConvTranspose2d(in_channels=nin, out_channels=nout, kernel_size=kernel, stride=2, padding=padding)
        else:
            self.deconv1 = nn.ConvTranspose2d(in_channels=nin, out_channels=nout, kernel_size=kernel, stride=(2, 1), padding=padding)
        self.deconv2 = nn.ConvTranspose2d(in_channels=nout, out_channels=nout, kernel_size=kernel, stride=1, padding=padding)
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.BN = nn.BatchNorm2d(nout)

    def forward(self, input, out_size):
        output = self.deconv1(input, output_size=out_size)
        output = self.leaky_relu(self.BN(output))
        output = self.deconv2(output)
        return output



class LocalMotionFill(nn.Module):
    def __init__(self, downsample=True, in_channel=4, out_channel=4, kernel=3, latentD=1024, **kwargs):
        super(LocalMotionFill, self).__init__()
        self.enc_blc1 = EncBlock(nin=in_channel, nout=32, downsample=downsample, kernel=kernel)
        self.enc_blc2 = EncBlock(nin=32, nout=64, downsample=downsample, kernel=kernel)
        self.enc_blc3 = EncBlock(nin=64, nout=128, downsample=downsample, kernel=kernel)
        self.enc_blc4 = EncBlock(nin=128, nout=256, downsample=downsample, kernel=kernel)
        self.enc_blc5 = EncBlock(nin=256, nout=256, downsample=downsample, kernel=kernel)

        self.dec_blc1 = DecBlock(nin=256, nout=256, upsample=downsample, kernel=kernel)
        self.dec_blc2 = DecBlock(nin=256, nout=128, upsample=downsample, kernel=kernel)
        self.dec_blc3 = DecBlock(nin=128, nout=64, upsample=downsample, kernel=kernel)
        self.dec_blc4 = DecBlock(nin=64, nout=32, upsample=downsample, kernel=kernel)
        self.dec_blc5 = DecBlock_output(nin=32, nout=out_channel, upsample=downsample, kernel=kernel)

        self.conv_mu = nn.Conv2d(512, 256, 3, 1, 1)
        self.conv_var = nn.Conv2d(512, 256, 3, 1, 1)
        self.conv_dec = nn.Conv2d(512, 256, 3, 1, 1)

        self.BN = nn.BatchNorm2d(256, track_running_stats=False)

    def CNN_feature(self, X):
        X1 = self.enc_blc1(X)
        X2 = self.enc_blc2(X1)
        X3 = self.enc_blc3(X2)
        X4 = self.enc_blc4(X3)
        X5 = self.enc_blc5(X4)
        return X5

    def encode(self, I, I_cond, **kwargs):
        X5 = self.CNN_feature(I)
        X_cond = self.CNN_feature(I_cond)

        feature = torch.cat([X5, X_cond], dim=1)
        mean = self.conv_mu(feature)
        std = self.conv_var(feature)

        Z = torch.distributions.normal.Normal(mean, F.softplus(std))
        return Z

    def decode(self, Zs, I_cond, **kwargs):
        X1 = self.enc_blc1(I_cond)
        X2 = self.enc_blc2(X1)
        X3 = self.enc_blc3(X2)
        X4 = self.enc_blc4(X3)
        X5 = self.enc_blc5(X4)
        X = torch.cat([X5, Zs], dim=1)
        X = self.BN(self.conv_dec(X))

        x_up4 = self.dec_blc1(X, X4.size())
        x_up3 = self.dec_blc2(x_up4, X3.size())
        x_up2 = self.dec_blc3(x_up3, X2.size())
        x_up1 = self.dec_blc4(x_up2, X1.size())
        output = self.dec_blc5(x_up1, I_cond.size())

        return output


    def forward(self, I, I_cond, **kwargs):
        Z = self.encode(I, I_cond)
        z_s = Z.rsample()

        output = self.decode(z_s, I_cond)
        return output, Z.mean, Z.scale


class MotionFill(nn.Module):
    def __init__(self, **kwargs):
        super(MotionFill, self).__init__()
        self.trajNet = TrajFill()
        self.localMotionNet = LocalMotionFill()

    def decode(self, Zs, I_cond, **kwargs):
        return self.localMotionNet.decode(Zs, I_cond)

    def forward(self, I, I_cond, traj, traj_cond, **kwargs):
        predict_traj, mean_traj, std_traj = self.trajNet(traj, traj_cond)
        predict_I, mean_local, std_local = self.localMotionNet(I, I_cond)

        return {
            'traj': predict_traj,
            'I': predict_I,
            'mean_traj': mean_traj,
            'std_traj': std_traj,
            'mean_local': mean_local,
            'std_local': std_local
        }

