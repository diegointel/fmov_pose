# we need crop/scale the image into the same size and assume we have the same camera
# remeber to record the relationship between original reolution and new resolution
import cv2
import numpy as np
import torch
import os
import argparse


# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
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


def get_crop_M(mask, patch_width=480, patch_height=480, patch_border=5):
    mask_ys, mask_xs = np.where(mask > 0)
    if len(mask_ys) < 3:
        return None
    raw_patch_cx = (mask_xs.max() + mask_xs.min()) / 2
    raw_patch_cy = (mask_ys.max() + mask_ys.min()) / 2
    raw_patch_w = mask_xs.max() - mask_xs.min() + 2 * patch_border
    raw_patch_h = mask_ys.max() - mask_ys.min() + 2 * patch_border
    scale = min(patch_width / raw_patch_w, patch_height / raw_patch_h)
    pleft = patch_width / 2 - raw_patch_cx * scale
    ptop = patch_height / 2 - raw_patch_cy * scale
    trans_M = np.array(
        [[scale, 0.0, pleft], [0.0, scale, ptop], [0.0, 0.0, 1.0]]
    )  # transformation matrix
    return trans_M.astype(np.float32)


def get_crop_M_ori(mask):
    patch_height, patch_width = mask.shape[:2]
    mask_ys, mask_xs = np.where(mask > 0)
    if len(mask_ys) < 3:
        return None
    raw_patch_cx = (mask_xs.max() + mask_xs.min()) / 2
    raw_patch_cy = (mask_ys.max() + mask_ys.min()) / 2
    # move the center to  (patch_width/2, patch_height/2)
    pleft = patch_width / 2 - raw_patch_cx
    ptop = patch_height / 2 - raw_patch_cy
    trans_M = np.array(
        [[1.0, 0.0, pleft], [0.0, 1.0, ptop], [0.0, 0.0, 1.0]]
    )  # transformation matrix
    return trans_M.astype(np.float32)


def origin_to_new(coords, transform_matrix):
    # Compute the inverse of the transformation matrix
    # Convert the 2D coordinates to 3D vectors
    coords_homogeneous = np.concatenate(
        [coords, np.ones((coords.shape[0], 1))], axis=-1
    )
    # Multiply the vectors by the inverse transformation matrix
    coords_transformed = (transform_matrix @ coords_homogeneous.T).T
    # Convert back to 2D
    coords_new = coords_transformed[:, :2]
    return coords_new


def get_match(match_file):
    if os.path.exists(match_file):
        xs1, ys1, xs2, ys2 = [], [], [], []
        for line in open(match_file).readlines():
            line = line.replace("\n", "").split("\t")
            xs1.append(int(float(line[0])))
            ys1.append(int(float(line[1])))
            xs2.append(int(float(line[2])))
            ys2.append(int(float(line[3])))
        return (np.array(xs1), np.array(ys1), np.array(xs2), np.array(ys2))
    else:
        return None


def solve_pose_by_pnp(
    points_2d: torch.Tensor, points_3d: torch.Tensor, internel_k: torch.Tensor, **kwargs
):
    """
    Args:
        points_2d (Tensor): xy coordinates of 2d points, shape (N, 2)
        points_3d (Tenosr): xyz coordinates of 3d points, shape (N, 3)
        internel_k (Tensor): camera intrinsic, shape (3, 3)
        kwargs (dict):
    """
    if points_2d.size(0) < 4:
        return None, None, False
    if kwargs.get("solve_pose_mode", "ransacpnp") == "ransacpnp":
        ransacpnp_parameter = kwargs.get("solve_pose_param", {})
        reprojectionError = ransacpnp_parameter.get("reprojectionerror", 3.0)
        iterationscount = ransacpnp_parameter.get("iterationscount", 100)
        retval, rotation_pred, translation_pred, inliers = cv2.solvePnPRansac(
            points_3d.cpu().numpy(),
            points_2d.cpu().numpy(),
            internel_k.cpu().numpy(),
            None,
            flags=cv2.SOLVEPNP_EPNP,
            reprojectionError=reprojectionError,
            iterationsCount=iterationscount,
        )
        rotation_pred = cv2.Rodrigues(rotation_pred)[0].reshape(3, 3)
    else:
        raise RuntimeError(f"Not supported pnp solver :{kwargs.get('solve_pose_mode')}")
    if retval:
        translation_pred = translation_pred.reshape(-1)
        if np.isnan(rotation_pred.sum()) or np.isnan(translation_pred.sum()):
            retval = False
    return rotation_pred, translation_pred, retval


# python virtual_cam_preprocess.py --root ./data_to_test_virtual_cam --ori --has_gt
if __name__ == "__main__":
    print("Virtual Camera Preprocess")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default="./data_to_test_virtual_cam",
        help="root of the data",
    )
    parser.add_argument(
        "--has_gt",
        default=False,
        action="store_true",
        help="if we have GT depth and GT pose annotation",
    )
    parser.add_argument(
        "--ori",
        default=False,
        action="store_true",
        help="if we want to maintain the original resolution but only shift the object center to the center of the image",
    )

    parser.add_argument(
        "--crop_resolution",
        type=int,
        default=480,
        help="the resolution we want to crop the image",
    )
    parser.add_argument(
        "--patch_border", type=int, default=5, help="the border of the patch"
    )
    args = parser.parse_args()

    root = args.root
    has_gt = args.has_gt
    ori = args.ori
    crop_resolution = args.crop_resolution
    patch_border = args.patch_border

    for seq in os.listdir(root):
        if f"_{crop_resolution}" in seq or "_ori" in seq:
            continue
        print("processing", seq, "...")
        data_dir = os.path.join(root, seq)
        if ori:
            new_data_dir = data_dir + "_ori"
        else:
            new_data_dir = data_dir + f"_{crop_resolution}"
        if patch_border != 5:
            new_data_dir += f"_{patch_border}"
        import os

        os.makedirs(new_data_dir, exist_ok=True)

        image_dir = os.path.join(data_dir, "image")
        mask_dir = os.path.join(data_dir, "mask_obj")
        # we need GT depth to get the 3D points so as to get the GT poses during virtual camera phase
        depth_dir = os.path.join(data_dir, "depth")

        if has_gt:
            camera_file = os.path.join(data_dir, "cameras_sphere.npz")
        else:
            camera_file = None

        new_image_dir = os.path.join(new_data_dir, "image")
        new_mask_dir = os.path.join(new_data_dir, "mask_obj")
        os.makedirs(new_image_dir, exist_ok=True)
        os.makedirs(new_mask_dir, exist_ok=True)

        # read images
        images = []
        for file in sorted(os.listdir(image_dir)):
            images.append(cv2.imread(os.path.join(image_dir, file)))
        image_names = sorted(os.listdir(image_dir))
        # remove suffix
        image_names = [name.split(".")[0] for name in image_names]

        # get frame_to_id
        frame_to_id = {}
        for i, name in enumerate(image_names):
            frame_to_id[name] = i

        masks = []
        for file in sorted(os.listdir(mask_dir)):
            masks.append(cv2.imread(os.path.join(mask_dir, file), cv2.IMREAD_GRAYSCALE))
        depths = []
        for file in sorted(os.listdir(depth_dir)):
            if "png" in file:
                depths.append(
                    cv2.imread(os.path.join(depth_dir, file), cv2.IMREAD_UNCHANGED)
                )
            else:
                depths.append(np.load(os.path.join(depth_dir, file)))
        scales = []
        transform_matrixs = []

        for i in range(len(images)):
            if ori:
                transform_matrix = get_crop_M_ori(masks[i])
                shape = (masks[i].shape[1], masks[i].shape[0])
            else:
                transform_matrix = get_crop_M(masks[i], patch_border=patch_border)
                shape = (480, 480)
            new_img = cv2.warpAffine(
                images[i], transform_matrix[:2], shape, flags=cv2.INTER_NEAREST
            )
            new_mask = cv2.warpAffine(
                masks[i], transform_matrix[:2], shape, flags=cv2.INTER_NEAREST
            )

            scales.append(transform_matrix[0, 0])
            transform_matrixs.append(transform_matrix)

            cv2.imwrite(
                os.path.join(new_data_dir, "image", f"{image_names[i]}.jpg"), new_img
            )
            cv2.imwrite(
                os.path.join(new_data_dir, "mask_obj", f"{image_names[i]}.jpg.png"),
                new_mask,
            )
            # print(f"Processing {i}/{len(images)}")
        print("mean scales", np.mean(scales))
        mean_scale = np.mean(scales)

        if not has_gt:
            # we don't have GT poses
            camera_dict = {}
        else:
            # we can calculate the GT poses during virtual camera phase
            camera_dict = np.load(camera_file)
        HO3D_K = None

        gt_poses = []
        intrinsics_all = []
        new_K = np.eye(3)
        new_K[:2, 2] = [240, 240]
        new_camera_dict = {}
        world_pts_total = []
        world_colors_total = []
        reproj_errors = []

        # find image name set
        avai_ann_frame = set()
        for k in camera_dict.keys():
            if "world_mat" in k:
                frame_name = k.split("_")[2]
                avai_ann_frame.add(frame_name)
        i = None
        for frame in avai_ann_frame:
            world_mat = camera_dict["world_mat_%s" % frame].astype(np.float32)
            scale_mat = camera_dict["scale_mat_%s" % frame].astype(np.float32)
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)  # pose is camera to world
            if HO3D_K is None:
                HO3D_K = intrinsics[:3, :3]
                if ori:
                    new_K = HO3D_K
                else:
                    new_K[0, 0] = intrinsics[0, 0] * mean_scale
                    new_K[1, 1] = intrinsics[1, 1] * mean_scale
            gt_depth = depths[frame_to_id[frame]] * 1 / scale_mat[2, 2]

            # we want to project points to world
            ys, xs = np.where(masks[frame_to_id[frame]] > 0)
            cam_pts = np.stack([xs, ys, np.ones_like(xs)], axis=-1)  # N, 3
            cam_pts = cam_pts * gt_depth[ys, xs, None]  # N, 3
            cam_pts = (
                np.linalg.inv(HO3D_K) @ cam_pts.T
            ).T  # (3, 3) @ (3, N) -> (3, N) -> (N, 3)
            cam_pts_hom = np.concatenate(
                [cam_pts, np.ones((len(cam_pts), 1))], axis=-1
            )  # N, 4
            world_pts = (pose @ cam_pts_hom.T).T  # N, 4
            color = images[frame_to_id[frame]][ys, xs]

            world_norm = np.linalg.norm(world_pts[:, :3], axis=-1)
            # find the maximum world_pts norm
            # print("pts max_norm before", world_norm.max())
            valid_mask = world_norm < 1
            world_pts = world_pts[valid_mask]
            color = color[valid_mask]
            # for verification
            # world_pts_total.append(world_pts)
            # world_colors_total.append(color)

            # transform original points to new points
            new_pts_2D = origin_to_new(
                np.stack([xs, ys], axis=-1), transform_matrixs[frame_to_id[frame]]
            )
            new_pts_2D = new_pts_2D[valid_mask]
            # perform PnP to get pose
            rotation_pred, translation_pred, retval = solve_pose_by_pnp(
                torch.tensor(new_pts_2D).float(),
                torch.tensor(world_pts[:, :3]).float(),
                torch.tensor(new_K).float(),
            )
            Rt = np.concatenate([rotation_pred, translation_pred[:, None]], axis=-1)
            # project with estimated pose
            est_pts_2D = (new_K @ (Rt @ world_pts.T)).T
            est_pts_2D = est_pts_2D[:, :2] / est_pts_2D[:, 2:]
            # reprojection error
            reproj_error = np.mean(np.linalg.norm(new_pts_2D - est_pts_2D, axis=-1))
            # print("reproj_error", reproj_error)
            reproj_errors.append(reproj_error)
            new_K_4x4 = np.eye(4)
            new_K_4x4[:3, :3] = new_K
            Rt = np.concatenate([Rt, np.array([[0, 0, 0, 1]])], axis=0)
            gt_poses.append(np.linalg.inv(Rt))
            new_camera_dict[f"world_mat_{frame}"] = new_K_4x4 @ Rt
            new_camera_dict[f"scale_mat_{frame}"] = np.eye(4)
            # print("retval", retval)
        # if i == 10:
        #     break
        print("reproj_error mean, std", np.mean(reproj_errors), np.std(reproj_errors))
        # save new camera dict
        np.savez(os.path.join(new_data_dir, "cameras_sphere.npz"), **new_camera_dict)
        # save transform matrix
        transform_matrix_dict = {}
        for i in range(len(transform_matrixs)):
            transform_matrix_dict[image_names[i]] = transform_matrixs[i]
        np.save(
            os.path.join(new_data_dir, "transform_matrixs.npy"), transform_matrix_dict
        )
