import numpy as np
import os
import torch
from models.dataset import load_K_Rt_from_P
from utils.nope_nerf_utils_poses.align_traj import align_ate_c2b_use_a2b
from utils.nope_nerf_utils_poses.vis_cam_traj import vis_poses, vis_simple_traj
from utils.nope_nerf_utils_poses.comp_ate import compute_ATE, compute_rpe
from utils.get_norm_matrix import get_normalization
import cv2


def align_poses(
    ori_cam_path,
    mesh_path,
    pred_poses,
    Ks,
    transform_matrixs,
    exp_dir,
    img_names,
    iter,
    case,
    H=480,
    W=640,
    save_dataset=True,
    normalize_trans=True,
    tgt_dir=None,
    save_meta=True,
    global_mask_dir=None,
):
    eval_ids = set()
    if ori_cam_path is not None:
        # get original resolution pose
        camera_dict = np.load(ori_cam_path)
        ori_K = None
        ori_gt_poses = []
        for i in range(len(img_names)):
            tag = f"scale_mat_{img_names[i]}"
            if tag not in camera_dict:
                continue
            scale_mat = camera_dict[f"scale_mat_{img_names[i]}"]
            world_mat = camera_dict[f"world_mat_{img_names[i]}"]
            # the following function is actually applying the inverse of scale_mat on translation of c2w pose only
            P = world_mat @ scale_mat
            intrinsics, pose = load_K_Rt_from_P(None, P[:3, :4])
            if ori_K is None:
                ori_K = intrinsics
            ori_gt_poses.append(pose)
            eval_ids.add(i)
        ori_gt_poses = torch.from_numpy(np.stack(ori_gt_poses)).cpu()
    else:
        camera_dict = {}
        # this is for ML dataset
        ori_K = Ks[0]

    # load mesh
    import trimesh

    mesh = trimesh.load(mesh_path)
    pts = mesh.vertices

    global_poses = []
    eval_global_poses = []
    for i in range(len(img_names)):
        ratio = 0
        new_pose = np.linalg.inv(pred_poses[i])
        new_K = Ks[i][:3, :3]
        transform_matrix = transform_matrixs[i]
        patience = 30
        while True:
            # we sample 1000 points from the mesh
            world_pts_3d = pts[np.random.choice(pts.shape[0], 1000, replace=False)]
            # project the sampled pts to the image plane
            cam_pts_3d = world_pts_3d @ new_pose[:3, :3].T + new_pose[:3, 3]
            pixel_pts = (new_K @ cam_pts_3d.T).T
            pixel_pts = pixel_pts[:, :2] / pixel_pts[:, 2:]
            ratio = (
                np.sum(
                    (pixel_pts[:, 0] > 0)
                    & (pixel_pts[:, 0] < W)
                    & (pixel_pts[:, 1] > 0)
                    & (pixel_pts[:, 1] < H)
                )
                / 1000
            )
            if ratio < 0.3:
                print(f"ratio: {ratio}, continue")
                patience -= 1
                if patience == 0:
                    print(f"patience: {patience}, break")
                    if i in eval_ids:
                        eval_global_poses.append(eval_global_poses[-1])
                    break
                else:
                    exit()
                    continue
            else:
                pixels_pts_hom = np.concatenate([pixel_pts, np.ones((1000, 1))], axis=1)
                # transform the pixel pts to the original resolution
                pixels_pts_hom = (np.linalg.inv(transform_matrix) @ pixels_pts_hom.T).T
                ori_pixels_pts = pixels_pts_hom[:, :2] / pixels_pts_hom[:, 2:]
                retval, rotation_pred, translation_pred, inliers = cv2.solvePnPRansac(
                    world_pts_3d,
                    ori_pixels_pts,
                    ori_K[:3, :3],
                    None,
                    flags=cv2.SOLVEPNP_EPNP,
                    reprojectionError=3,
                    iterationsCount=100,
                )
                rotation_pred = cv2.Rodrigues(rotation_pred)[0].reshape(3, 3)
                obj_pose = np.eye(4)
                obj_pose[:3, :3] = rotation_pred
                obj_pose[:3, 3] = translation_pred.reshape(3)
                global_poses.append(np.linalg.inv(obj_pose))
                if i in eval_ids:
                    eval_global_poses.append(np.linalg.inv(obj_pose))
                break
    if save_dataset:
        if save_meta:
            # tgt_dir = f"./global_uniform32_data/{case}"
            tgt_dir = f"./global_reset_data/{case}"
            os.makedirs(tgt_dir, exist_ok=True)
            # we want to create a new dataset for global refinement, camera_dict, mask, image
            case = case.split("_")[0]
            if ori_cam_path is not None:
                img_dir = f"./data/HO3Dv2/{case}/image"
                mask_dir = f"./data/HO3Dv2/{case}/mask_obj"
            else:
                img_dir = f"./data/ML/{case}/image"
                mask_dir = f"./data/ML/{case}/mask_obj"
            os.makedirs(os.path.join(tgt_dir, "image"), exist_ok=True)
            os.makedirs(os.path.join(tgt_dir, "mask_obj"), exist_ok=True)
        noise_camera_dict = {}
        for i in range(len(img_names)):
            if save_meta:
                img_name = img_names[i]
                img = cv2.imread(os.path.join(img_dir, f"{img_name}.jpg"))
                mask = cv2.imread(
                    os.path.join(mask_dir, f"{img_name}.png"), cv2.IMREAD_GRAYSCALE
                )
                cv2.imwrite(os.path.join(tgt_dir, "image", f"{img_name}.jpg"), img)
                cv2.imwrite(os.path.join(tgt_dir, "mask_obj", f"{img_name}.png"), mask)
            noise_camera_dict["world_mat_%d" % i] = ori_K @ np.linalg.inv(
                global_poses[i]
            )
            if not normalize_trans:
                noise_camera_dict["scale_mat_%d" % i] = np.eye(4)
        print("noise_camera_dict.keys()", noise_camera_dict.keys())
        os.makedirs(tgt_dir, exist_ok=True)
        np.savez(os.path.join(tgt_dir, "cameras_sphere.npz"), **noise_camera_dict)
        try:
            if normalize_trans:
                # the following will normalize the tranlation based on predicted poses
                get_normalization(tgt_dir, False, masks_dir=global_mask_dir)
        except Exception as e:
            print("get_normalization failed----------------------------")
            print(e)
            for i in range(len(img_names)):
                noise_camera_dict["scale_mat_%d" % i] = np.eye(4)
            np.savez(os.path.join(tgt_dir, "cameras_sphere.npz"), **noise_camera_dict)
        # rename os.path.join(tgt_dir, "cameras_sphere.npz") to os.path.join(tgt_dir, "noise_cameras_sphere.npz")
        os.rename(
            os.path.join(tgt_dir, "cameras_sphere.npz"),
            os.path.join(tgt_dir, "noise_cameras_sphere.npz"),
        )
        np.savez(os.path.join(tgt_dir, "cameras_sphere.npz"), **camera_dict)
        if ori_cam_path is None:
            return
    else:
        # we want to save noise camera dict
        noise_camera_dict = {}
        for i in range(len(img_names)):
            noise_camera_dict["world_mat_%d" % i] = ori_K @ np.linalg.inv(
                global_poses[i]
            )
        np.savez(os.path.join(exp_dir, "noise_cameras_sphere.npz"), **noise_camera_dict)

    global_poses = np.stack(global_poses)
    np.save(
        os.path.join(exp_dir, f"global_poses_{len(img_names)}_{iter}.npy"), global_poses
    )

    # align global poses with ori_gt_poses for visualization :)
    eval_global_poses = torch.from_numpy(np.stack(eval_global_poses))
    eval_global_poses = align_ate_c2b_use_a2b(eval_global_poses, ori_gt_poses)
    ate = compute_ATE(ori_gt_poses.cpu().numpy(), eval_global_poses.cpu().numpy())
    rpe_trans, rpe_rot = compute_rpe(
        ori_gt_poses.cpu().numpy(), eval_global_poses.cpu().numpy()
    )
    fx, fy = ori_K[0, 0], ori_K[1, 1]
    print(f"ATE: {ate}, rpe_trans: {rpe_trans}, rpe_rot: {rpe_rot}")
    vis_poses(
        eval_global_poses,
        ori_gt_poses,
        H,
        W,
        fx,
        fy,
        save_path=os.path.join(
            exp_dir,
            f"global_alignment{len(img_names)}_{iter}_ate={ate}_rpet={rpe_trans}_rpe_r={rpe_rot}_rpe_degree={np.rad2deg(rpe_rot)}.png",
        ),
    )
    vis_simple_traj(
        eval_global_poses,
        ori_gt_poses,
        save_path=os.path.join(exp_dir, f"global_traj{len(img_names)}_{iter}.png"),
    )


def align_poses_wo_virtual(
    ori_cam_path,
    mesh_path,
    pred_poses,
    Ks,
    transform_matrixs,
    exp_dir,
    img_names,
    iter,
    case,
    H=480,
    W=640,
    save_dataset=True,
    normalize_trans=True,
    tgt_dir=None,
    save_meta=True,
    global_mask_dir=None,
):
    eval_ids = set()
    if ori_cam_path is not None:
        # get original resolution pose
        camera_dict = np.load(ori_cam_path)
        ori_K = None
        ori_gt_poses = []
        for i in range(len(img_names)):
            tag = f"scale_mat_{img_names[i]}"
            if tag not in camera_dict:
                continue
            scale_mat = camera_dict[f"scale_mat_{img_names[i]}"]
            world_mat = camera_dict[f"world_mat_{img_names[i]}"]
            # the following function is actually applying the inverse of scale_mat on translation of c2w pose only
            P = world_mat @ scale_mat
            intrinsics, pose = load_K_Rt_from_P(None, P[:3, :4])
            if ori_K is None:
                ori_K = intrinsics
            ori_gt_poses.append(pose)
            eval_ids.add(i)
        ori_gt_poses = torch.from_numpy(np.stack(ori_gt_poses)).cpu()
    else:
        camera_dict = {}
        # this is for ML dataset
        ori_K = Ks[0]

    # collect poses
    global_poses = []
    eval_global_poses = []
    for i in range(len(img_names)):
        global_poses.append(pred_poses[i])
        if i in eval_ids:
            eval_global_poses.append(pred_poses[i])

    # save camera dicts
    noise_camera_dict = {}
    for i in range(len(img_names)):
        noise_camera_dict["world_mat_%d" % i] = ori_K @ np.linalg.inv(global_poses[i])
        if not normalize_trans:
            noise_camera_dict["scale_mat_%d" % i] = np.eye(4)
    os.makedirs(tgt_dir, exist_ok=True)
    np.savez(os.path.join(tgt_dir, "cameras_sphere.npz"), **noise_camera_dict)
    if normalize_trans:
        # the following will normalize the tranlation based on predicted poses
        get_normalization(tgt_dir, False, masks_dir=global_mask_dir)
    os.rename(
        os.path.join(tgt_dir, "cameras_sphere.npz"),
        os.path.join(tgt_dir, "noise_cameras_sphere.npz"),
    )
    np.savez(os.path.join(tgt_dir, "cameras_sphere.npz"), **camera_dict)
    if ori_cam_path is None:
        # we don't have gt poses
        return

    # align global poses with ori_gt_poses for visualization :)
    eval_global_poses = torch.from_numpy(np.stack(eval_global_poses))
    eval_global_poses = align_ate_c2b_use_a2b(eval_global_poses, ori_gt_poses)
    ate = compute_ATE(ori_gt_poses.cpu().numpy(), eval_global_poses.cpu().numpy())
    rpe_trans, rpe_rot = compute_rpe(
        ori_gt_poses.cpu().numpy(), eval_global_poses.cpu().numpy()
    )
    fx, fy = ori_K[0, 0], ori_K[1, 1]
    print(f"ATE: {ate}, rpe_trans: {rpe_trans}, rpe_rot: {rpe_rot}")
    vis_poses(
        eval_global_poses,
        ori_gt_poses,
        H,
        W,
        fx,
        fy,
        save_path=os.path.join(
            exp_dir,
            f"global_alignment{len(img_names)}_{iter}_ate={ate}_rpet={rpe_trans}_rpe_r={rpe_rot}_rpe_degree={np.rad2deg(rpe_rot)}.png",
        ),
    )
    vis_simple_traj(
        eval_global_poses,
        ori_gt_poses,
        save_path=os.path.join(exp_dir, f"global_traj{len(img_names)}_{iter}.png"),
    )
