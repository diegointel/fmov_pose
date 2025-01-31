import torch
import cv2 as cv
import numpy as np
import os
from glob import glob
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
from plyfile import PlyData, PlyElement

np.random.seed(2024)
torch.manual_seed(2024)


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


def save_point_cloud(points, colors, path):
    vertices = np.zeros(
        len(points),
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    vertices["x"], vertices["y"], vertices["z"] = (
        points[:, 0],
        points[:, 1],
        points[:, 2],
    )
    vertices["red"], vertices["green"], vertices["blue"] = (
        colors[:, 2],
        colors[:, 1],
        colors[:, 0],
    )
    ply = PlyData([PlyElement.describe(vertices, "vertex")], text=True)
    ply.write(path)


def shrink_mask(mask, shrink_ratio=0.9):
    # Convert the boolean mask to a uint8 mask
    mask_uint8 = mask.astype(np.uint8) * 255
    # Calculate the size of the structuring element
    selem_size = int((1 - np.sqrt(shrink_ratio)) * np.sqrt(mask.size) / 2)
    # Get a structuring element for erosion
    selem = cv.getStructuringElement(cv.MORPH_ELLIPSE, (selem_size, selem_size))
    # Erode the mask
    eroded_mask_uint8 = cv.erode(mask_uint8, selem)
    # Convert the eroded mask back to a boolean mask
    eroded_mask = eroded_mask_uint8.astype(bool)
    return eroded_mask


def another_epe(rotation_matrix):
    import numpy as np
    from scipy.spatial.transform import Rotation as R

    # assuming rotation_matrix is your rotation matrix
    rotation = R.from_matrix(rotation_matrix)
    quaternion = rotation.as_quat()

    # compute the rotation angle from the quaternion
    rotation_angle = 2 * np.arccos(quaternion[3])

    # convert to degrees
    rotation_angle_degrees = np.degrees(rotation_angle)

    return rotation_angle_degrees


def get_center_radius(vertices):
    bbox_max = np.max(vertices, axis=0)
    bbox_min = np.min(vertices, axis=0)
    center = (bbox_max + bbox_min) * 0.5
    radius = np.linalg.norm(vertices - center, ord=2, axis=-1).max()
    return center, radius


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


def load_unit_K_Rt(K, Rt):
    R_cam_to_world = Rt[:3, :3].transpose()
    t_cam_in_world = -R_cam_to_world @ Rt[:3, 3]
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R_cam_to_world
    pose[:3, 3] = t_cam_in_world
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K
    return intrinsics, pose


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


class Dataset:
    def __init__(self, conf, exp_dir=None):
        super(Dataset, self).__init__()
        print("Load data: Begin")
        self.device = torch.device("cuda")
        self.conf = conf

        self.data_dir = conf.get_string("data_dir")
        self.render_cameras_name = conf.get_string("render_cameras_name")
        self.object_cameras_name = conf.get_string("object_cameras_name")

        self.camera_outside_sphere = conf.get_bool(
            "camera_outside_sphere", default=True
        )
        self.scale_mat_scale = conf.get_float("scale_mat_scale", default=1.1)
        if exp_dir is not None:
            # this is for rebooting global training :)
            camera_dir = exp_dir
        else:
            camera_dir = self.data_dir

        if not conf.get_bool("unknown_camera", default=False):
            if ".npz" in self.render_cameras_name:
                camera_dict = np.load(
                    os.path.join(camera_dir, self.render_cameras_name)
                )
            else:
                camera_dict = np.load(
                    os.path.join(camera_dir, self.render_cameras_name),
                    allow_pickle=True,
                ).item()
        self.images_lis = sorted(glob(os.path.join(self.data_dir, "image/*")))
        assert len(self.images_lis) > 0, "no images found!!!"

        self.masks_lis = sorted(glob(os.path.join(self.data_dir, "mask_obj/*")))
        assert len(self.masks_lis) > 0, "no masks found!!!"

        self.n_images = len(self.images_lis)
        self.images_np = (
            np.stack([cv.imread(im_name) for im_name in self.images_lis]) / 256.0
        )
        self.masks_np = (
            np.stack([cv.imread(im_name) for im_name in self.masks_lis]) / 256.0
        )

        if self.conf.get("wo_mask", default=False):
            for i in range(self.n_images):
                # put all pixels that mask = 0 to zero
                self.images_np[i][self.masks_np[i] < 0.5] = 0
            print("without mask supervision!!!")

        # all the flow/rotation prior loading information is image name based!!!
        self.start_idx = conf.get_int("start_idx", default=0)
        self.end_idx = conf.get_int("end_idx", default=self.n_images)
        self.frame_to_index = {}
        self.index_to_frame = {}
        self.image_names_set = set()  # this is for the case that we may delete some frames since it does not contain annotations :)
        for idx, im_name in enumerate(self.images_lis[self.start_idx : self.end_idx]):
            im_name = os.path.basename(im_name).split(".")[0]
            self.frame_to_index[im_name] = idx
            self.index_to_frame[idx] = im_name
            self.image_names_set.add(im_name)

        self.H, self.W = self.images_np.shape[1], self.images_np.shape[2]

        self.gt_poses = []
        self.intrinsics_all = []
        self.pose_all = []
        self.avai_ann_frame = []
        ml_camera_intrinsics = conf.get("ml_camera_intrinsics", default="")
        if ml_camera_intrinsics != "":
            K = np.zeros((3, 3))
            with open(ml_camera_intrinsics, "r") as f:
                lines = f.readlines()
                # every line is 3 number
                for i in range(3):
                    K[i, :] = list(map(float, lines[i].split()))
            for i in range(self.n_images):
                intrinsics = np.eye(4, dtype=np.float32)
                intrinsics[:3, :3] = K
                self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
                pose = np.eye(4, dtype=np.float32)
                pose[:3, :3] = np.eye(3)
                pose[:3, 3] = np.array([0, 0, 0])
                self.pose_all.append(torch.from_numpy(pose).float())
            self.scale_mats_np = [
                np.eye(4, dtype=np.float32) for _ in range(self.n_images)
            ]
            pass
        elif conf.get_bool("unknown_camera", default=False):
            print("using identity camera pose...")
            K = np.load(os.path.join(self.data_dir, "K.npy"))
            for idx in range(self.n_images):
                K4x4 = np.eye(4, dtype=np.float32)
                K4x4[:3, :3] = K
                self.intrinsics_all.append(torch.from_numpy(K4x4).float())
                pose = np.eye(4, dtype=np.float32)
                pose[:3, :3] = np.eye(3)
                pose[:3, 3] = np.array([0, 0, 0])
                self.pose_all.append(torch.from_numpy(pose).float())
            self.scale_mats_np = [
                np.eye(4, dtype=np.float32) for _ in range(self.n_images)
            ]
        elif conf.get_bool("partial_ann", default=False):
            self.scale_mats_np = [
                np.eye(4, dtype=np.float32) for _ in range(self.n_images)
            ]
            print("scale_mats_np", len(self.scale_mats_np))
            print("using partial annotation camera pose...")
            intrinsics = None
            for k in self.frame_to_index.keys():
                # for corner case that 1st frame would not have annotation
                if f"world_mat_{k}" in camera_dict:
                    world_mat = camera_dict[f"world_mat_{k}"].astype(np.float32)
                    scale_mat = camera_dict[f"scale_mat_{k}"].astype(np.float32)
                    P = world_mat @ scale_mat
                    P = P[:3, :4]
                    intrinsics, pose = load_K_Rt_from_P(None, P)
                    break

            for k in self.frame_to_index.keys():
                # this loop ensure that it is in order
                if f"world_mat_{k}" in camera_dict:
                    world_mat = camera_dict[f"world_mat_{k}"].astype(np.float32)
                    scale_mat = camera_dict[f"scale_mat_{k}"].astype(np.float32)
                    P = world_mat @ scale_mat
                    P = P[:3, :4]
                    intrinsics, pose = load_K_Rt_from_P(None, P)
                    self.gt_poses.append(torch.from_numpy(pose))
                    self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
                    self.pose_all.append(torch.from_numpy(pose).float())
                    self.avai_ann_frame.append(self.frame_to_index[k])  # for evaluation
                elif intrinsics is not None:
                    self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            pass
        else:
            raise NotImplementedError

        self.use_crop_init = conf.get_bool("use_crop_init", default=False)
        self.crop_scale = 1
        if self.use_crop_init:
            noise_camera_dict = np.load(
                os.path.join(camera_dir, "noise_cameras_sphere.npz")
            )
            self.crop_poses = []
            if len(self.gt_poses) == 0:
                use_noise_intrinsic = True
            else:
                use_noise_intrinsic = False
            for i in range(self.n_images):
                scale_mat = noise_camera_dict[f"scale_mat_{i}"]
                self.crop_scale = scale_mat[0, 0]
                world_mat = noise_camera_dict[f"world_mat_{i}"]
                P = world_mat @ scale_mat
                P = P[:3, :4]
                intrinsics, pose = load_K_Rt_from_P(None, P)
                self.crop_poses.append(torch.from_numpy(pose).float())
                if use_noise_intrinsic:
                    self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.crop_poses = torch.stack(self.crop_poses).to(
                self.device
            )  # [n_images, 4, 4]

        # load crop transform matrixes
        self.crop = conf.get_bool("crop", default=False)
        if self.crop:
            self.crop_transforms = np.load(
                os.path.join(self.data_dir, "transform_matrixs.npy"), allow_pickle=True
            ).item()

        # we want to load loftr flow
        self.loftr_interval_flow_dir = self.conf.get(
            "loftr_interval_flow_dir", default=None
        )
        self.filter_match_outliers = self.conf.get_bool(
            "filter_match_outliers", default=False
        )
        if self.loftr_interval_flow_dir is not None:
            self.loftr_interval_flows = {}
            self.flow_pairs = {}
            seq_name = self.data_dir.split("/")[-2]
            seq_name = seq_name.split("_")[0]
            print("loading loftr interval flow...", seq_name)
            seq_flow_dir = os.path.join(self.loftr_interval_flow_dir, seq_name)
            for file in os.listdir(seq_flow_dir):
                # it is txt file like before
                frame_name1, frame_name2 = file.split("_")[
                    :2
                ]  # this is file name, please do not modify!!!!
                if (
                    frame_name1 not in self.image_names_set
                    or frame_name2 not in self.image_names_set
                ):
                    continue
                file_path = os.path.join(seq_flow_dir, file)
                xs1, ys1, xs2, ys2 = [], [], [], []
                for line in open(file_path).readlines():
                    line = line.replace("\n", "").split("\t")
                    xs1.append(float(line[0]))
                    ys1.append(float(line[1]))
                    xs2.append(float(line[2]))
                    ys2.append(float(line[3]))
                xys1 = np.stack([np.array(xs1), np.array(ys1)], axis=-1)
                xys2 = np.stack([np.array(xs2), np.array(ys2)], axis=-1)
                if self.filter_match_outliers:
                    # filter based on distance via 3-sigma
                    dists = np.linalg.norm(xys1 - xys2, axis=-1)
                    mean_dist = np.mean(dists)
                    std_dist = np.std(dists)
                    valid_mask = np.abs(dists - mean_dist) < 3 * std_dist
                    xys1 = xys1[valid_mask]
                    xys2 = xys2[valid_mask]
                if self.crop:
                    trans1 = self.crop_transforms[frame_name1]
                    trans2 = self.crop_transforms[frame_name2]
                    xys1 = origin_to_new(xys1, trans1)
                    xys2 = origin_to_new(xys2, trans2)
                # filter based on image border
                valid_mask = (
                    (xys1[:, 0] >= 0)
                    & (xys1[:, 0] < self.W)
                    & (xys1[:, 1] >= 0)
                    & (xys1[:, 1] < self.H)
                )
                valid_mask &= (
                    (xys2[:, 0] >= 0)
                    & (xys2[:, 0] < self.W)
                    & (xys2[:, 1] >= 0)
                    & (xys2[:, 1] < self.H)
                )
                xs1, ys1 = xys1[valid_mask][:, 0], xys1[valid_mask][:, 1]
                xs2, ys2 = xys2[valid_mask][:, 0], xys2[valid_mask][:, 1]
                # filter based on mask
                valid_mask = (
                    self.masks_np[self.frame_to_index[frame_name1]][..., 0][
                        (ys1.astype(int), xs1.astype(int))
                    ]
                    > 0.5
                )
                valid_mask &= (
                    self.masks_np[self.frame_to_index[frame_name2]][..., 0][
                        (ys2.astype(int), xs2.astype(int))
                    ]
                    > 0.5
                )
                xs1, ys1 = xs1[valid_mask], ys1[valid_mask]
                xs2, ys2 = xs2[valid_mask], ys2[valid_mask]
                tag = frame_name1 + "_" + frame_name2

                if tag not in self.loftr_interval_flows:
                    self.loftr_interval_flows[tag] = (
                        np.array(xs1),
                        np.array(ys1),
                        np.array(xs2),
                        np.array(ys2),
                    )
                tag = frame_name2 + "_" + frame_name1
                if tag not in self.loftr_interval_flows:
                    self.loftr_interval_flows[tag] = (
                        np.array(xs2),
                        np.array(ys2),
                        np.array(xs1),
                        np.array(ys1),
                    )
                if frame_name1 not in self.flow_pairs:
                    self.flow_pairs[frame_name1] = set()
                self.flow_pairs[frame_name1].add(frame_name2)
                if frame_name2 not in self.flow_pairs:
                    self.flow_pairs[frame_name2] = set()
                self.flow_pairs[frame_name2].add(frame_name1)

        # actually we can do initialization with mask
        self.mask_init = conf.get_bool("mask_init", default=False)
        if self.mask_init:
            # we want to find the frame with maximum area
            max_mask_index = 0
            # the following will be perfect but it will introduce a lot of workload, we temporally # try the first frame :)
            # max_mask_area = 0
            # for i in range(self.n_images):
            #     mask_sum = self.masks_np[i].sum()
            #     if mask_sum > max_mask_area:
            #         max_mask_area = mask_sum
            #         max_mask_index = i
            ys, xs = np.where(self.masks_np[max_mask_index][:, :, 0] > 0.5)
            # we want to project the mask to camera space
            K = self.intrinsics_all[max_mask_index][:3, :3]
            homogeneous_pixels = np.stack([xs, ys, np.ones_like(xs)], axis=-1)
            homogeneous_pixels = homogeneous_pixels.reshape(-1, 3)
            K_inv = np.linalg.inv(K)
            camera_points = (K_inv @ homogeneous_pixels.T).T
            camera_points = camera_points / camera_points[:, 2:]
            self.max_mask_pose = np.eye(4, dtype=np.float32)
            if self.crop:
                xy_radius = np.linalg.norm(camera_points[:, :2], ord=2, axis=-1).max()
                # we want to scale it to 0.9
                self.max_mask_pose[:3, 3] = np.array([0, 0, -0.9 / xy_radius])
            else:
                cam_x_min, cam_x_max, cam_y_min, cam_y_max = (
                    camera_points[:, 0].min(),
                    camera_points[:, 0].max(),
                    camera_points[:, 1].min(),
                    camera_points[:, 1].max(),
                )
                print(
                    "cam_x_min, cam_x_max, cam_y_min, cam_y_max",
                    cam_x_min,
                    cam_x_max,
                    cam_y_min,
                    cam_y_max,
                )
                cam_center = np.array(
                    [(cam_x_min + cam_x_max) / 2, (cam_y_min + cam_y_max) / 2]
                )
                print("cam_center", cam_center)
                print(camera_points[:, :2], cam_center)
                print(
                    (camera_points[:, :2] - cam_center[np.newaxis, ...]).max(),
                    (camera_points[:, :2] - cam_center[np.newaxis, ...]).min(),
                )
                xy_radius = np.linalg.norm(
                    (camera_points[:, :2] - cam_center[np.newaxis, ...]), axis=-1
                ).max()
                print("xy_radius", xy_radius)
                # we want to scale it to 0.9
                # the depth is 0.9 / xy_radius
                self.max_mask_pose[:3, 3] = np.array(
                    [cam_center[0], cam_center[1], 1]
                ) * (-0.9 / xy_radius)
                # verify the correctness
                camera_points = camera_points * (0.9 / xy_radius)
                world_points = camera_points + self.max_mask_pose[:3, 3]
                print(
                    "world_points norm min, max, mean",
                    np.linalg.norm(world_points, axis=-1).min(),
                    np.linalg.norm(world_points, axis=-1).max(),
                    np.linalg.norm(world_points, axis=-1).mean(),
                )
            self.max_mask_pose = torch.from_numpy(self.max_mask_pose).cuda().float()
            self.max_mask_index = max_mask_index
            pass

        self.images = torch.from_numpy(
            self.images_np.astype(np.float32)
        ).cpu()  # [n_images, H, W, 3]
        self.masks = torch.from_numpy(
            self.masks_np.astype(np.float32)
        ).cpu()  # [n_images, H, W, 3]
        self.intrinsics_all = torch.stack(self.intrinsics_all).to(
            self.device
        )  # [n_images, 4, 4]
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)  # [n_images, 4, 4]
        self.focal = self.intrinsics_all[0][0, 0]
        if len(self.pose_all) > 0:
            self.pose_all = torch.stack(self.pose_all).to(
                self.device
            )  # [n_images, 4, 4]
        if len(self.gt_poses) > 0:
            self.gt_poses = torch.stack(self.gt_poses).to(
                self.device
            )  # [n_images, 4, 4]

        print("------start_idx", self.start_idx)
        self.images = self.images[self.start_idx : self.end_idx]
        self.masks = self.masks[self.start_idx : self.end_idx]
        self.intrinsics_all = self.intrinsics_all[self.start_idx : self.end_idx]
        self.intrinsics_all_inv = self.intrinsics_all_inv[self.start_idx : self.end_idx]
        if len(self.gt_poses) > 0:
            self.pose_all = self.pose_all[self.start_idx : self.end_idx]
            self.gt_poses = self.gt_poses[self.start_idx : self.end_idx]
        self.n_images = self.images.shape[0]
        self.images_lis = self.images_lis[self.start_idx : self.end_idx]

        # In the paper, we didn't mention depth; however, during our research, we found that, current system can be boosted a lot by using depth.
        # If you are interested, you can load the depth on your own and slightly modify the code.
        self.use_mono_depth = conf.get_bool("use_mono_depth", default=False)

        if self.use_mono_depth:
            self.mono_depths = self.mono_depths[self.start_idx : self.end_idx]

        self.image_pixels = self.H * self.W

        object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        object_bbox_max = np.array([1.01, 1.01, 1.01, 1.0])
        # Object scale mat: region of interest to **extract mesh**
        object_scale_mat = np.eye(4, dtype=np.float32)

        object_bbox_min = (
            np.linalg.inv(self.scale_mats_np[0])
            @ object_scale_mat
            @ object_bbox_min[:, None]
        )
        object_bbox_max = (
            np.linalg.inv(self.scale_mats_np[0])
            @ object_scale_mat
            @ object_bbox_max[:, None]
        )
        self.object_bbox_min = object_bbox_min[:3, 0]
        self.object_bbox_max = object_bbox_max[:3, 0]

        print("Load data: End")

    def gen_rays_at(self, img_idx, resolution_level=1, pose=None, with_mask=False):
        """
        Generate rays at world space from one camera.
        """
        if pose is None:
            # use GT pose
            pose = self.pose_all[img_idx]
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack(
            [pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1
        )  # W, H, 3
        p = torch.matmul(
            self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None]
        ).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.matmul(
            pose[None, None, :3, :3], rays_v[:, :, :, None]
        ).squeeze()  # W, H, 3
        rays_o = pose[None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        mask = self.masks[img_idx][(pixels_y.long(), pixels_x.long())]  # W, H, 3
        if with_mask:
            return (
                rays_o.transpose(0, 1),
                rays_v.transpose(0, 1),
                mask[..., 0].transpose(0, 1),
            )
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def get_rays_based_on_mask(self, img_idx, pose, resolution_level=1):
        """
        Generate rays at world space from one camera based on mask.
        """
        if not self.crop:
            mask = self.masks_np[img_idx][:, :, 0]
        else:
            mask_dir = os.path.join(self.data_dir.replace("_ori", ""), "mask_obj")
            mask = (
                cv.imread(
                    os.path.join(mask_dir, self.index_to_frame[img_idx] + ".png"),
                    cv.IMREAD_UNCHANGED,
                )
                / 255
            )
            if len(mask.shape) == 3:
                mask = mask[..., 0]
        ys, xs = np.where(mask > 0.5)
        # pixels_x = torch.from_numpy(xs).cuda()
        # pixels_y = torch.from_numpy(ys).cuda()
        ys_min, ys_max = max(ys.min() - 5, 0), min(ys.max() + 5, self.H - 1)
        xs_min, xs_max = max(xs.min() - 5, 0), min(xs.max() + 5, self.W - 1)
        x_step = (xs_max - xs_min) // resolution_level
        y_step = (ys_max - ys_min) // resolution_level

        # we need shift it based on transform matrix
        if self.crop:
            # then the current xs, ys is in original resolution
            tranform_matrix = self.crop_transforms[self.index_to_frame[img_idx]]
            x_shift = tranform_matrix[0, 2]
            y_shift = tranform_matrix[1, 2]
            xs_min = xs_min + x_shift
            xs_max = xs_max + x_shift
            ys_min = ys_min + y_shift
            ys_max = ys_max + y_shift
            xs = xs + x_shift
            ys = ys + y_shift

        # create meshgrid
        tx = torch.linspace(xs_min, xs_max, x_step).long()
        ty = torch.linspace(ys_min, ys_max, y_step).long()
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        pixels_x = pixels_x.reshape(-1)
        pixels_y = pixels_y.reshape(-1)
        xs, ys = pixels_x.cpu().numpy(), pixels_y.cpu().numpy()
        mask = self.masks[img_idx][(pixels_y, pixels_x)]
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()
        p = torch.matmul(
            self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]
        ).squeeze()
        p_norm = torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)
        rays_v = p / p_norm
        rays_v = torch.matmul(pose[None, :3, :3], rays_v[:, :, None]).squeeze()
        rays_o = pose[None, :3, 3].expand(rays_v.shape)
        return rays_o, rays_v, ys, xs, p_norm

    def gen_random_rays_at(
        self, img_idx, batch_size, pose, mask_guided_sampling=False, patch_size=30
    ):
        """
        Generate random rays at world space from one camera.
        """
        self.images = self.images.to('cuda:0')
        self.masks = self.masks.to('cuda:0')

        if mask_guided_sampling and np.random.rand() < 0.7:
            mask = self.masks_np[img_idx][:, :, 0]
            ys, xs = np.where(mask > 0.5)
            ys_min, ys_max = (
                max(ys.min() - patch_size, 0),
                min(ys.max() + patch_size, self.H),
            )
            xs_min, xs_max = (
                max(xs.min() - patch_size, 0),
                min(xs.max() + patch_size, self.W),
            )
        else:
            ys_min, ys_max, xs_min, xs_max = 0, self.H, 0, self.W
        pixels_x = torch.randint(low=xs_min, high=xs_max, size=[batch_size])
        pixels_y = torch.randint(low=ys_min, high=ys_max, size=[batch_size])
        color = self.images[img_idx][(pixels_y, pixels_x)]  # batch_size, 3
        mask = self.masks[img_idx][(pixels_y, pixels_x)]  # batch_size, 3
        p = torch.stack(
            [pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1
        ).float()  # batch_size, 3
        p = torch.matmul(
            self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]
        ).squeeze()  # batch_size, 3
        p_norm = torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)
        rays_v = p / p_norm  # batch_size, 3
        rays_v = torch.matmul(
            pose[None, :3, :3], rays_v[:, :, None]
        ).squeeze()  # batch_size, 3
        rays_o = pose[None, :3, 3].expand(rays_v.shape)  # batch_size, 3
        if self.use_mono_depth:
            depth = (
                self.mono_depths[img_idx][(pixels_y, pixels_x)][..., None].cuda()
                * p_norm.cuda()
            )
        else:
            depth = None
        return torch.cat(
            [rays_o, rays_v, color, mask[:, :1]], dim=-1
        ).cuda(), depth

    def gen_random_ray_pairs_at(
        self, img_id_corr, batch_size, pose_network, current_img_num, interval=1
    ):
        """
        Generate random ray pairs at world space based on flow.
        pose_all is only for barf
        """
        # search corresponding image based on interval flow
        img_name_corr = self.index_to_frame[img_id_corr.item()]
        if img_name_corr not in self.flow_pairs:
            return None, None, None, None, None
        pairs = self.flow_pairs[img_name_corr]
        pairs_idx = [self.frame_to_index[img_name] for img_name in pairs]
        # pairs_idx < current_img_num and abs(pairs_idx - img_id_corr) < interval
        pairs_idx = [
            idx
            for idx in pairs_idx
            if idx < current_img_num and abs(idx - img_id_corr) <= interval
        ]
        if len(pairs_idx) == 0:
            return None, None, None, None, None
        img_id = torch.tensor(np.random.choice(pairs_idx)).long().cuda()

        # get flow
        xs1, ys1, xs2, ys2 = self.loftr_interval_flows[
            img_name_corr + "_" + self.index_to_frame[img_id.item()]
        ]

        # we need to sample the points from xs1
        indexs = np.random.choice(len(xs1), batch_size, replace=True)
        pixels_x_corr = torch.from_numpy(xs1[indexs]).cuda()
        pixels_y_corr = torch.from_numpy(ys1[indexs]).cuda()
        pixels_x = torch.from_numpy(xs2[indexs]).cuda()
        pixels_y = torch.from_numpy(ys2[indexs]).cuda()

        # color
        color_corr = self.images[img_id_corr][
            (pixels_y_corr.long(), pixels_x_corr.long())
        ]  # batch_size, 3
        color = self.images[img_id][(pixels_y.long(), pixels_x.long())]  # batch_size, 3

        # view directions
        p_corr = torch.stack(
            [pixels_x_corr, pixels_y_corr, torch.ones_like(pixels_y_corr)], dim=-1
        ).float()
        p_corr = torch.matmul(
            self.intrinsics_all_inv[img_id_corr, None, :3, :3], p_corr[:, :, None]
        ).squeeze()  # batch_size, 3
        p_corr_norm = torch.linalg.norm(p_corr, ord=2, dim=-1, keepdim=True)
        rays_v_corr = p_corr / p_corr_norm  # batch_size, 3
        pose_corr = pose_network(img_id_corr)
        rays_v_corr = torch.matmul(
            pose_corr[None, :3, :3], rays_v_corr[:, :, None]
        ).squeeze()  # batch_size, 3
        rays_o_corr = pose_corr[None, :3, 3].expand(rays_v_corr.shape)  # batch_size, 3
        if self.use_mono_depth:
            depth_corr = (
                self.mono_depths[img_id_corr][
                    (pixels_y_corr.long(), pixels_x_corr.long())
                ][..., None].cuda()
                * p_corr_norm.cuda()
            )
        else:
            depth_corr = torch.zeros(batch_size)

        p = torch.stack(
            [pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1
        ).float()  # batch_size, 3
        p = torch.matmul(
            self.intrinsics_all_inv[img_id, None, :3, :3], p[:, :, None]
        ).squeeze()  # batch_size, 3
        p_norm = torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)
        rays_v = p / p_norm  # batch_size, 3
        pose = pose_network(img_id)
        rays_v = torch.matmul(
            pose[None, :3, :3], rays_v[:, :, None]
        ).squeeze()  # batch_size, 3
        rays_o = pose[None, :3, 3].expand(rays_v.shape)  # batch_size, 3

        if self.use_mono_depth:
            depth = (
                self.mono_depths[img_id][(pixels_y.long(), pixels_x.long())][
                    ..., None
                ].cuda()
                * p_norm.cuda()
            )
        else:
            depth = torch.zeros(batch_size)

        # since we already preprocess it, so the flows are already inside the mask!
        mask = torch.ones(batch_size, 1)
        mask_corr = torch.ones(batch_size, 1)

        # start concatenation
        rays_o = torch.cat([rays_o_corr, rays_o], dim=0)
        rays_v = torch.cat([rays_v_corr, rays_v], dim=0)
        color = torch.cat([color_corr, color], dim=0).cuda()
        mask = torch.cat([mask_corr, mask], dim=0)
        depth = torch.cat([depth_corr, depth], dim=0)

        pixels_xy_corr = torch.stack([pixels_x_corr, pixels_y_corr], dim=-1)
        pixels_xy = torch.stack([pixels_x, pixels_y], dim=-1)

        return (
            torch.cat([rays_o, rays_v, color, mask[:, :1].cuda()], dim=-1),
            pixels_xy,
            pixels_xy_corr,
            img_id,
            depth,
        )

    def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1, pose_all=None):
        """
        Interpolate pose between two cameras.
        """
        if pose_all is None:
            pose_all = self.pose_all
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack(
            [pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1
        )  # W, H, 3
        p = torch.matmul(
            self.intrinsics_all_inv[0, None, None, :3, :3], p[:, :, :, None]
        ).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        trans = pose_all[idx_0, :3, 3] * (1.0 - ratio) + pose_all[idx_1, :3, 3] * ratio
        pose_0 = pose_all[idx_0].detach().cpu().numpy()
        pose_1 = pose_all[idx_1].detach().cpu().numpy()
        pose_0 = np.linalg.inv(pose_0)
        pose_1 = np.linalg.inv(pose_1)
        rot_0 = pose_0[:3, :3]
        rot_1 = pose_1[:3, :3]
        rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
        key_times = [0, 1]
        slerp = Slerp(key_times, rots)
        rot = slerp(ratio)
        pose = np.diag([1.0, 1.0, 1.0, 1.0])
        pose = pose.astype(np.float32)
        pose[:3, :3] = rot.as_matrix()
        pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
        pose = np.linalg.inv(pose)
        rot = torch.from_numpy(pose[:3, :3]).cuda()
        trans = torch.from_numpy(pose[:3, 3]).cuda()
        rays_v = torch.matmul(
            rot[None, None, :3, :3], rays_v[:, :, :, None]
        ).squeeze()  # W, H, 3
        rays_o = trans[None, None, :3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def near_far_from_sphere(self, rays_o, rays_d):
        # this function "mid +-1 " would typically introduce the points outside of the unit sphere, which is not disirable
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near, far

    def near_far_from_sphere_new(self, rays_o, rays_d):
        # it does not work, we may need more advanced mechanisms
        # Center and radius of the unit sphere
        C = torch.zeros_like(rays_o)
        r = 1.0

        # Coefficients of the quadratic equation
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_d * (rays_o - C), dim=-1, keepdim=True)
        c = torch.sum((rays_o - C) ** 2, dim=-1, keepdim=True) - r**2

        # Discriminant
        discriminant = b**2 - 4 * a * c

        # If the discriminant is negative, the ray does not intersect the sphere
        no_intersection = discriminant < 0

        # If the ray intersects the sphere, calculate the intersection points
        sqrt_discriminant = torch.sqrt(
            torch.max(discriminant, torch.zeros_like(discriminant))
        )
        near = (-b - sqrt_discriminant) / (2 * a)
        far = (-b + sqrt_discriminant) / (2 * a)

        return near, far, no_intersection

    def image_at(self, idx, resolution_level):
        img = cv.imread(self.images_lis[idx])
        return (
            cv.resize(img, (self.W // resolution_level, self.H // resolution_level))
        ).clip(0, 255)
