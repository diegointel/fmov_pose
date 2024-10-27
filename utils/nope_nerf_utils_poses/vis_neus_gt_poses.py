from vis_cam_traj import vis_single_pose
import numpy as np
import cv2 as cv
import os

import torch
import tempfile
import imageio
from tqdm import tqdm


# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()  # this rotation can rotate camera axis to world axis
    pose[:3, 3] = (t[:3] / t[3])[
        :, 0
    ]  # after translation, the t can represent the camera center in world coordinate

    return intrinsics, pose


def vis_gt_poses(npz_path, increasing=True):
    camera_dict = np.load(npz_path)
    n_images = 0
    for k in camera_dict.keys():
        if "scale_mat_" in k and "inv" not in k:
            n_images += 1
    poses = []
    images = []
    prev_scale_mat = None
    with tempfile.TemporaryDirectory() as tmpdirname:
        for idx in tqdm(range(n_images)):
            world_mat = camera_dict["world_mat_{}".format(idx)]
            scale_mat = camera_dict["scale_mat_{}".format(idx)]
            P = world_mat @ scale_mat
            intrinsics, pose = load_K_Rt_from_P(None, P[:3, :4])
            fx, fy = intrinsics[0, 0], intrinsics[1, 1]
            H, W = int(intrinsics[1, 2] * 2), int(intrinsics[0, 2] * 2)
            poses.append(pose)
            if increasing:
                cur_poses = torch.from_numpy(np.stack(poses, axis=0))
                vis_single_pose(
                    cur_poses,
                    H,
                    W,
                    fx,
                    fy,
                    save_path=os.path.join(tmpdirname, "tmp.png"),
                )
                images.append(imageio.imread(os.path.join(tmpdirname, "tmp.png")))
            if prev_scale_mat is not None:
                # compare the scale matrix
                assert np.allclose(
                    prev_scale_mat, scale_mat
                ), "scale matrix is not consistent"
            prev_scale_mat = scale_mat
    if increasing:
        imageio.mimsave(
            os.path.join(os.path.dirname(npz_path), "gt_poses.gif"), images, fps=5
        )
        imageio.mimsave(
            os.path.join(os.path.dirname(npz_path), "gt_poses.mp4"), images, fps=5
        )
    else:
        poses = torch.from_numpy(np.stack(poses, axis=0))
        npz_name = os.path.basename(npz_path).replace(".npz", ".png")
        vis_single_pose(
            poses,
            H,
            W,
            fx,
            fy,
            save_path=os.path.join(os.path.dirname(npz_path), npz_name),
        )
    pass
