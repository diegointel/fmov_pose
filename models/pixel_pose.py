from models.barf_embedder import get_embedder as get_barf_embedder
from models.embedder import get_embedder, FourierEmbedding, OriginalFourierEmbedding
from models.batch_lie_group_helper import batch_make_c2w
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


# The following was borrowed from https://github.com/papagina/RotationContinuity/blob/master/sanity_test/code/tools.py
def get_rotation_matrix_from_ortho6d(ortho6d):
    x_raw = ortho6d[..., 0:3]  # shape becomes (B, N, 3)
    y_raw = ortho6d[..., 3:6]  # shape becomes (B, N, 3)

    x = F.normalize(x_raw, p=2, dim=-1)  # shape becomes (B, N, 3)
    z = torch.cross(x, y_raw, dim=-1)
    z = F.normalize(z, p=2, dim=-1)
    y = torch.cross(z, x, dim=-1)  # shape becomes (B, N, 3)

    x = x.unsqueeze(-1)  # shape becomes (B, N, 3, 1)
    y = y.unsqueeze(-1)  # shape becomes (B, N, 3, 1)
    z = z.unsqueeze(-1)  # shape becomes (B, N, 3, 1)
    matrix = torch.cat((x, y, z), -1)  # shape becomes (B, N, 3, 3)
    return matrix


# the architecture of the network is borrowed from NeRFtrinsic Four: https://github.com/HannahHaensen/nerftrinsic_four/blob/main/models/poses.py
class PixelPose(nn.Module):
    def __init__(
        self,
        init_c2w=None,
        x_multiers=10,
        t_multiers=10,
        rot_type="angle",
        internal_init=None,
        output_init="zero",
        fixed_1st_cam=False,
    ):
        """
        :param init_c2w: (N, 4, 4) torch tensor
        :param x_multiers: encoding camera coordinate points, hyperparamer referred to implementation of d-nerf
        :param t_multiers: encoding time, hyperparamer referred to implementation of d-nerf
        """
        super(PixelPose, self).__init__()
        if init_c2w is not None:
            self.init_c2w = nn.Parameter(init_c2w, requires_grad=False)
        else:
            self.init_c2w = None
        self.embed_fn_t, input_ch_t = get_embedder(multires=t_multiers, input_dims=1)
        self.embed_fn_x, input_ch_x = get_embedder(multires=x_multiers, input_dims=3)

        self.lin1 = nn.Linear(input_ch_t + input_ch_x, 64)
        self.gelu1 = nn.GELU()
        self.lin2 = nn.Linear(64, 64)
        self.gelu2 = nn.GELU()
        if rot_type == "ord":
            self.lin3 = nn.Linear(64, 9)
        elif rot_type == "angle":
            self.lin3 = nn.Linear(64, 6)
        else:
            raise NotImplementedError

        if output_init == "zero" and rot_type == "angle":
            # init weights so it has identity matrix rotation and zero translation
            nn.init.zeros_(self.lin3.weight)
            nn.init.zeros_(self.lin3.bias)

        self.rot_type = rot_type

        self.fixed_1st_cam = fixed_1st_cam

    def forward(self, input_pts, cam_id):
        """
        :param input_pts: (B, 3) or (B, N, 3) torch tensor for camera coordinate points
        :param cam_id: current camera
        """
        if self.fixed_1st_cam and cam_id == 0:
            if len(input_pts.shape) == 2:
                return self.init_c2w[cam_id][:3, :4].expand(input_pts.shape[0], -1, -1)
            elif len(input_pts.shape) == 3:
                return self.init_c2w[cam_id][:3, :4].expand(
                    input_pts.shape[0], input_pts.shape[1], -1, -1
                )
            else:
                raise NotImplementedError

        t = cam_id
        if len(input_pts.shape) == 2:
            t = t[None, ...].expand(input_pts.shape[0], -1)
        elif len(input_pts.shape) == 3:
            t = t[None, None, ...].expand(input_pts.shape[0], input_pts.shape[1], -1)
        else:
            raise NotImplementedError

        t_features = self.embed_fn_t(t)
        input_pts_features = self.embed_fn_x(input_pts)
        features = torch.cat([input_pts_features, t_features], dim=-1)

        pred = self.lin1(features)
        pred = self.gelu1(pred)
        pred = self.lin2(pred)
        pred = self.gelu2(pred)
        pred = self.lin3(pred)

        if self.rot_type == "ord":
            r = get_rotation_matrix_from_ortho6d(pred[..., :6])
            t = pred[..., 6:]
            c2w = torch.cat([r, t.unsqueeze(-1)], dim=-1)  # (B, 3, 4)
        elif self.rot_type == "angle":
            c2w = batch_make_c2w(
                pred[..., :3].reshape(-1, 3), pred[..., 3:].reshape(-1, 3)
            )  # (B, 3, 4)

        if self.init_c2w is not None:
            c2w = c2w @ self.init_c2w[cam_id].expand(c2w.shape[0], -1, -1)

        if len(input_pts.shape) == 2:
            c2w = c2w.reshape(input_pts.shape[0], 3, 4)
        elif len(input_pts.shape) == 3:
            c2w = c2w.reshape(input_pts.shape[0], input_pts.shape[1], 3, 4)
        else:
            raise NotImplementedError
        return c2w


class DeepPixelPose(nn.Module):
    def __init__(
        self,
        D=8,
        W=256,
        skips=[4],
        init_c2w=None,
        x_multiers=10,
        t_multiers=10,
        rot_type="angle",
        output_init="small_weight",
        fixed_1st_cam=False,
        img_id_delta=0,
        cam_id_encoding="position",
        fourier_embed_dim=128,
        disable_pts=False,
        barf=False,
    ):
        super().__init__()
        print("DeepPixelPose------------------")
        if img_id_delta > 0:
            print("DeepPixelPose img_id_delta", img_id_delta)
        self.D = D
        self.W = W

        if init_c2w is not None:
            if img_id_delta > 0:
                init_c2w = init_c2w[0][None, :, :].repeat(
                    init_c2w.shape[0] + img_id_delta, 1, 1
                )
            self.init_c2w = nn.Parameter(init_c2w, requires_grad=False)
        else:
            self.init_c2w = None

        self.cam_id_encoding = cam_id_encoding
        if cam_id_encoding == "original_fourier":
            print("Camera ID Original Fourier Embedding...", "dim:", 512)
            self.embed_fn_t = OriginalFourierEmbedding()
            input_ch_t = 512
        elif cam_id_encoding == "fourier":
            print("Camera ID Fourier Embedding...", "dim:", fourier_embed_dim)
            self.embed_fn_t = FourierEmbedding(
                num_cam=init_c2w.shape[0], embed_dim=fourier_embed_dim
            )
            input_ch_t = fourier_embed_dim * 2
        elif cam_id_encoding == "position":
            print("Camera ID Position Encoding...")
            self.embed_fn_t, input_ch_t = get_embedder(
                multires=t_multiers, input_dims=1
            )
        elif cam_id_encoding == "embedding":
            print("Camera ID Embedding...")
            self.embeddings = nn.Embedding(init_c2w.shape[0], 128)
            self.embeddings.weight.data.normal_(0, 1)
            self.embed_fn_t = lambda x: self.embeddings(x.long())
            input_ch_t = 128
            self.embeddings.requires_grad_(False)
        else:
            raise NotImplementedError
        self.barf = barf
        if barf:
            print("Deep Pixel Pose MLP Using BARF Embedder...")
            # self.progress = torch.nn.Parameter(torch.tensor(0.)) # use Parameter so it could be checkpointed
            self.embed_fn_x, input_ch_x = get_barf_embedder(
                multires=x_multiers, input_dims=3
            )

        else:
            self.embed_fn_x, input_ch_x = get_embedder(
                multires=x_multiers, input_dims=3
            )

        self.skips = skips

        input_ch = input_ch_x + input_ch_t
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)]
            + [
                nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W)
                for i in range(D - 1)
            ]
        )

        if rot_type == "angle":
            self.output_linear = nn.Linear(W, 6)
            if output_init == "zero":
                print("All Zero Initialization...")
                # init weights so it has identity matrix rotation and zero translation
                nn.init.zeros_(self.output_linear.weight)
                nn.init.zeros_(self.output_linear.bias)
            elif output_init == "direct":
                print("Bias Direct Initialization...")
                # In this case only translation prediction can decide the position of ray origin
                # while rotation decides view directions
                init_rotation = torch.zeros(3)
                init_translation = init_c2w[0, :3, 3].reshape(3)
                init_c2w = torch.cat([init_rotation, init_translation])
                nn.init.zeros_(self.output_linear.weight)
                with torch.no_grad():
                    self.output_linear.bias.copy_(init_c2w)
            elif output_init == "small_weight":
                print("Small Weight Initialization...")
                # init weights so it has identity matrix rotation and zero translation
                nn.init.normal_(self.output_linear.weight, mean=0, std=0.01)
                nn.init.zeros_(self.output_linear.bias)
            elif output_init == "no_init":
                print("No Initialization...")
            else:
                raise NotImplementedError
        elif rot_type == "ord":
            self.output_linear = nn.Linear(W, 9)
            raise NotImplementedError

        self.fixed_1st_cam = fixed_1st_cam
        self.rot_type = rot_type

        # the following is for debugging
        self.detach_translation = False
        self.detach_rotation = False
        self.output_init = output_init

        self.img_id_delta = img_id_delta
        self.disable_pts = disable_pts

        if self.disable_pts:
            print("Disable Points...")

    def forward(self, input_pts, cam_id):
        cam_id = cam_id + self.img_id_delta
        if self.fixed_1st_cam and cam_id == 0 and not self.disable_pts:
            if len(input_pts.shape) == 2:
                return self.init_c2w[cam_id][:3, :4].expand(input_pts.shape[0], -1, -1)
            elif len(input_pts.shape) == 3:
                return self.init_c2w[cam_id][:3, :4].expand(
                    input_pts.shape[0], input_pts.shape[1], -1, -1
                )
            else:
                raise NotImplementedError

        if self.cam_id_encoding == "fourier":
            t = cam_id
        else:
            t = (
                cam_id / self.init_c2w.shape[0]
            )  # to ensure number stability since input is including
        t_features = self.embed_fn_t(t.unsqueeze(0)).squeeze(0)

        if not self.disable_pts:
            if len(input_pts.shape) == 2:
                t_features = t_features[None, ...].expand(input_pts.shape[0], -1)
            elif len(input_pts.shape) == 3:
                t_features = t_features[None, None, ...].expand(
                    input_pts.shape[0], input_pts.shape[1], -1
                )
            else:
                raise NotImplementedError
            if self.barf:
                if not hasattr(self, "progress"):
                    self.progress = torch.nn.Parameter(
                        torch.tensor(0.0)
                    )  # use Parameter so it could be checkpointed
                input_pts_features = self.embed_fn_x(input_pts, self.progress)
            else:
                input_pts_features = self.embed_fn_x(input_pts)
        else:
            t_features = t_features.unsqueeze(0)
            pts = torch.tensor([0, 0, 0], dtype=torch.float32, device=t_features.device)
            input_pts_features = self.embed_fn_x(pts.unsqueeze(0))
        features = torch.cat([input_pts_features, t_features], dim=-1)

        h = features

        for i, _ in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([features, h], -1)

        pred = self.output_linear(h)

        if self.output_init == "direct":
            pred[..., :3] = (
                F.tanh(pred[..., :3]) * np.pi / 18
            )  # we assume the maxmum degree is 10
        if self.rot_type == "angle":
            rotation = pred[..., :3].reshape(-1, 3)
            translation = pred[..., 3:].reshape(-1, 3)
            if self.detach_translation:
                translation = translation.detach()
            if self.detach_rotation:
                rotation = rotation.detach()
            c2w = batch_make_c2w(rotation, translation)  # (B, 3, 4)
        else:
            raise NotImplementedError

        if self.init_c2w is not None and self.output_init != "direct":
            c2w = c2w @ self.init_c2w[cam_id].expand(c2w.shape[0], -1, -1)

        if not self.disable_pts:
            if len(input_pts.shape) == 2:
                c2w = c2w.reshape(input_pts.shape[0], 3, 4)
            elif len(input_pts.shape) == 3:
                c2w = c2w.reshape(input_pts.shape[0], input_pts.shape[1], 3, 4)
            else:
                raise NotImplementedError
        else:
            c2w = c2w[0, ...]
        return c2w

    def disable_grad(self):
        # disable lin1, lin2, lin3
        for param in self.pts_linears.parameters():
            param.requires_grad = False
        for param in self.output_linear.parameters():
            param.requires_grad = False

    def enable_grad(self):
        # enable lin1, lin2, lin3
        for param in self.pts_linears.parameters():
            param.requires_grad = True
        for param in self.output_linear.parameters():
            param.requires_grad = True


class SegDeepPixelPose(nn.Module):
    def __init__(self, num_cams, segment_img_num, init_c2w=None):
        super().__init__()
        print("SegDeepPixelPose------------------")
        self.num_cams = num_cams
        self.pose_mlps = []
        self.segment_img_num = segment_img_num
        pose_mlp_num = init_c2w.shape[0] // segment_img_num
        if init_c2w.shape[0] % segment_img_num != 0:
            pose_mlp_num += 1
        for _ in range(pose_mlp_num):
            self.pose_mlps.append(
                DeepPixelPose(init_c2w=init_c2w.clone(), disable_pts=True)
            )
        self.pose_mlps = nn.ModuleList(self.pose_mlps)
        self.initialized_flag = nn.Parameter(
            torch.tensor([True] + [False] * (pose_mlp_num - 1)), requires_grad=False
        )
        self.progress = nn.Parameter(torch.zeros(pose_mlp_num), requires_grad=False)

    def forward(self, cam_id):
        pose_mlp_index = cam_id.item() // self.segment_img_num
        if not self.initialized_flag[pose_mlp_index]:
            self.initialized_flag[pose_mlp_index] = True
            with torch.no_grad():
                last_cam_id = pose_mlp_index * self.segment_img_num - 1
                last_pose = self.pose_mlps[pose_mlp_index - 1](
                    None, torch.tensor(last_cam_id)
                )
                last_pose4x4 = torch.eye(4)
                last_pose4x4[:3] = last_pose[:3]
                last_pose4x4 = last_pose4x4.clone().repeat(self.num_cams, 1, 1)
                # don't fucking create a new nn.Parameter
                self.pose_mlps[pose_mlp_index].init_c2w.data.copy_(last_pose4x4)
        return self.pose_mlps[pose_mlp_index](None, cam_id)

    def step_progress(self, pose_mlp_index):
        self.progress[pose_mlp_index] += 1
        return self.progress[pose_mlp_index]
