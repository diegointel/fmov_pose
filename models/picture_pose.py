import torch
import torch.nn as nn
import numpy as np
from models.batch_lie_group_helper import batch_make_c2w

# set default device as cuda if available
if torch.cuda.is_available():
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

# the architecture of the network is borrowed from NeRFtrinsic Four: https://github.com/HannahHaensen/nerftrinsic_four/blob/main/models/poses.py


class LearnPoseGF(nn.Module):
    def __init__(
        self,
        num_cams,
        init_c2w=None,
        pose_encoding=False,
        embedding_scale=10,
        emphasize_rot=False,
        small_rot=False,
    ):
        """
        :param num_cams: number of camera poses
        :param init_c2w: (N, 4, 4) torch tensor
        :param pose_encoding True/False, positional encoding or gaussian fourer
        :param embedding_scale hyperparamer, can also be adapted
        """
        super(LearnPoseGF, self).__init__()
        # print("using LearnPoseGF")
        self.emphasize_rot = emphasize_rot
        self.num_cams = num_cams
        self.embedding_size = 128
        self.all_points = torch.tensor([(i) for i in range(num_cams)])
        if init_c2w is not None:
            self.init_c2w = nn.Parameter(init_c2w, requires_grad=False)
        else:
            self.init_c2w = None
        self.lin1 = nn.Linear(self.embedding_size * 2, 64)
        self.gelu1 = nn.GELU()
        self.lin2 = nn.Linear(64, 64)
        self.gelu2 = nn.GELU()
        if not self.emphasize_rot:
            self.lin3 = nn.Linear(64, 6)
            nn.init.normal_(self.lin3.weight, mean=0, std=0.01)
            nn.init.zeros_(self.lin3.bias)
        else:
            self.lin3_rot = nn.Linear(64, 3)
            nn.init.normal_(self.lin3_rot.weight, mean=0, std=0.01)
            nn.init.zeros_(self.lin3_rot.bias)

            self.lin3_trans = nn.Linear(64, 3)
            nn.init.zeros_(self.lin3_trans.weight)
            nn.init.zeros_(self.lin3_trans.bias)
            for param in self.lin3_trans.parameters():
                param.requires_grad = False

            self.lin3_scale = nn.Linear(64, 1)
            nn.init.normal_(self.lin3_scale.weight, mean=0, std=0.01)
            nn.init.ones_(self.lin3_scale.bias)

        self.embedding_scale = embedding_scale

        if pose_encoding:
            # print("AXIS")
            posenc_mres = 5
            self.b = 2.0 ** np.linspace(0, posenc_mres, self.embedding_size // 2) - 1.0
            self.b = self.b[:, np.newaxis]
            self.b = np.concatenate([self.b, np.roll(self.b, 1, axis=-1)], 0) + 0
            self.b = torch.tensor(self.b).float()
            self.a = torch.ones_like(self.b[:, 0])
        else:
            # print("FOURIER")
            self.b = np.random.normal(
                loc=0.0, scale=self.embedding_scale, size=[self.embedding_size, 1]
            )  # * self.embedding_scale
            self.b = torch.tensor(self.b).float()
            self.a = torch.ones_like(self.b[:, 0])

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.b = self.b
        self.b = torch.nn.Parameter(self.b, requires_grad=False).to(self.device)
        self.a = self.a.to(self.device)
        self.all_points = self.all_points.to(self.device)
        self.debug_timmer = 0
        self.small_rot = small_rot
        if self.small_rot:
            print("small rotation")

    def finish_warmup(self):
        pass
        # if self.emphasize_rot:
        #     for param in self.lin3_trans.parameters():
        #         param.requires_grad = True
        #     nn.init.normal_(self.lin3_trans.weight, mean=0, std=0.01)
        #     nn.init.zeros_(self.lin3_trans.bias)

    def disable_trans(self):
        for param in self.lin3_scale.parameters():
            param.requires_grad = False

    def enable_trans(self):
        for param in self.lin3_scale.parameters():
            param.requires_grad = True

    def disable_grad(self):
        # disable lin1, lin2, lin3
        for param in self.lin1.parameters():
            param.requires_grad = False
        for param in self.lin2.parameters():
            param.requires_grad = False
        if not self.emphasize_rot:
            for param in self.lin3.parameters():
                param.requires_grad = False
        else:
            for param in self.lin3_rot.parameters():
                param.requires_grad = False
            for param in self.lin3_trans.parameters():
                param.requires_grad = False
            for param in self.lin3_scale.parameters():
                param.requires_grad = False

    def enable_grad(self):
        # enable lin1, lin2, lin3
        for param in self.lin1.parameters():
            param.requires_grad = True
        for param in self.lin2.parameters():
            param.requires_grad = True
        if not self.emphasize_rot:
            for param in self.lin3.parameters():
                param.requires_grad = True
        else:
            for param in self.lin3_rot.parameters():
                param.requires_grad = True
            for param in self.lin3_trans.parameters():
                param.requires_grad = True
            for param in self.lin3_scale.parameters():
                param.requires_grad = True

    def forward(self, cam_id):
        """
        :param cam_id: current camera
        """
        cam_id = self.all_points[cam_id]
        cam_id = cam_id.unsqueeze(0)

        fourier_features = torch.cat(
            [
                self.a * torch.sin((2.0 * np.pi * cam_id) @ self.b.T),
                self.a * torch.cos((2.0 * np.pi * cam_id) @ self.b.T),
            ],
            axis=-1,
        ) / torch.linalg.norm(self.a)
        pred = self.lin1(fourier_features)
        pred = self.gelu1(pred)
        pred = self.lin2(pred)
        pred = self.gelu2(pred)
        if len(pred.shape) == 1:
            pred = pred[None, :]
        if not self.emphasize_rot:
            pred = self.lin3(pred)
            if self.small_rot:
                # we can limit it to 30 degrees
                pred_rot = pred[:, :3] * np.pi / 6
            else:
                pred_rot = pred[:, :3] * np.pi
            pred_trans = pred[:, 3:]
        else:
            if self.small_rot:
                pred_rot = self.lin3_rot(pred) * np.pi / 6
            else:
                pred_rot = self.lin3_rot(pred) * np.pi
            pred_trans = self.lin3_trans(pred)
            pred_scale = self.lin3_scale(pred)

        c2w = batch_make_c2w(pred_rot, pred_trans).squeeze(0)  # (4, 4)
        if self.init_c2w is not None:
            if self.emphasize_rot:
                t = self.init_c2w[cam_id][0][:3, 3] * pred_scale[0]
            else:
                t = self.init_c2w[cam_id][0][:3, 3]
            tmp = torch.eye(4)
            tmp[:3, 3] = t
            tmp[:3, :3] = self.init_c2w[cam_id][0][:3, :3]
            c2w = c2w @ tmp
        return c2w


class SegLearnPose(nn.Module):
    def __init__(
        self,
        num_cams,
        segment_img_num,
        init_c2w=None,
        pose_encoding=False,
        embedding_scale=10,
        emphasize_rot=False,
        small_rot=False,
    ):
        super(SegLearnPose, self).__init__()
        self.num_cams = num_cams
        self.pose_mlps = []
        self.segment_img_num = segment_img_num
        pose_mlp_num = init_c2w.shape[0] // segment_img_num
        if init_c2w.shape[0] % segment_img_num != 0:
            pose_mlp_num += 1
        for _ in range(pose_mlp_num):
            self.pose_mlps.append(
                LearnPoseGF(
                    num_cams,
                    init_c2w.clone(),
                    pose_encoding,
                    embedding_scale,
                    emphasize_rot,
                    small_rot,
                )
            )
        self.pose_mlps = nn.ModuleList(self.pose_mlps)
        self.initialized_flag = nn.Parameter(
            torch.tensor([True] + [False] * (pose_mlp_num - 1)), requires_grad=False
        )
        self.progress = nn.Parameter(torch.zeros(pose_mlp_num), requires_grad=False)
        pass

    def forward(self, cam_id):
        pose_mlp_index = cam_id.item() // self.segment_img_num
        if not self.initialized_flag[pose_mlp_index]:
            self.initialized_flag[pose_mlp_index] = True
            with torch.no_grad():
                last_cam_id = pose_mlp_index * self.segment_img_num - 1
                last_pose = self.pose_mlps[pose_mlp_index - 1](last_cam_id)
                last_pose4x4 = torch.eye(4)
                last_pose4x4[:3] = last_pose[:3]
                last_pose4x4 = last_pose4x4.clone().repeat(self.num_cams, 1, 1)
                # don't fucking create a new nn.Parameter
                self.pose_mlps[pose_mlp_index].init_c2w.data.copy_(last_pose4x4)
        return self.pose_mlps[pose_mlp_index](cam_id)

    def set_pose(self, cam_id, pose, force_update=False):
        pose_mlp_index = cam_id // self.segment_img_num
        if not self.initialized_flag[pose_mlp_index] or force_update:
            self.initialized_flag[pose_mlp_index] = True
            with torch.no_grad():
                pose = pose.clone().repeat(self.num_cams, 1, 1)
                # don't fucking create a new nn.Parameter
                self.pose_mlps[pose_mlp_index].init_c2w.data.copy_(pose)

    def step_progress(self, pose_mlp_index):
        self.progress[pose_mlp_index] += 1
        return self.progress[pose_mlp_index]
