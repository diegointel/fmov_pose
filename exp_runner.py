import os
import logging
import argparse
import numpy as np
import cv2 as cv
import trimesh
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile
from tqdm import tqdm
from pyhocon import ConfigFactory
from models.dataset import Dataset
from models.fields import RenderingNetwork, SDFNetwork, SingleVarianceNetwork, NeRF
from models.renderer import NeuSRenderer
from models.barf_fields import BarfSDFNetwork, BarfRenderingNetwork
import models.camera as camera
from models.camera import to_hom
import imageio
from utils.textured_mesh import textured_mesh
from utils.nope_nerf_utils_poses.align_traj import align_ate_c2b_use_a2b
from utils.nope_nerf_utils_poses.comp_ate import compute_ATE, compute_rpe
from utils.nope_nerf_utils_poses.vis_cam_traj import vis_poses, vis_simple_traj
from models.picture_pose import LearnPoseGF, SegLearnPose
from models.pixel_pose import SegDeepPixelPose
from utils.util import *
import traceback

np.random.seed(2024)
torch.manual_seed(2024)

from utils.align_poses import align_poses, align_poses_wo_virtual


# the following is for debugging
def get_gradients(params):
    norms = []
    for param in params:
        if param.grad is not None:
            norms.append(param.grad.abs().mean().cpu().item())
    norms = np.array(norms)
    if len(norms) == 0:
        return 0, 0, 0
    return np.min(norms), np.max(norms), np.mean(norms)


def extract_camera_poses(poses, image_names, output_csv):
    """
    Extract and save camera poses to a CSV file.

    Args:
        poses (list of torch.Tensor): List of camera poses (4x4 transformation matrices).
        image_names (list of str): List of image names corresponding to the poses.
        output_csv (str): Path to save the extracted camera poses.
    """
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image Name', 'Camera Pose'])

        for image_name, pose in zip(image_names, poses):
            pose_str = ' '.join(map(str, pose.flatten().tolist()))
            writer.writerow([image_name, pose_str])

    print(f'Camera poses saved to {output_csv}')


class Runner:
    def __init__(
        self,
        conf_path,
        mode="train",
        case="CASE_NAME",
        dataset="DTU",
        is_continue=False,
        start_at=-1,
        start_img_idx=0,
        gradient_analysis=False,
        exp_dir=None,
        has_global_conf=False,
        flow_interval=-1,
        reset_rot_degree=-1,
        image_interval=-1,
    ):
        self.case = case
        self.device = torch.device("cuda")
        self.gradient_analysis = gradient_analysis
        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace("CASE_NAME", case).replace("DATA_SET", dataset)
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        if exp_dir is not None:
            self.base_exp_dir = exp_dir
        else:
            self.base_exp_dir = self.conf["general.base_exp_dir"]
        if not has_global_conf and "global_reset_exp" not in self.base_exp_dir:
            print("Remove global conf...")
            self.base_exp_dir = self.base_exp_dir + "_wo_global_conf"
        else:
            print("Use global conf...")

        if flow_interval > 0:
            self.base_exp_dir += f"_m{flow_interval}"
            self.conf.put("train.flow_interval", flow_interval)
        if reset_rot_degree > 0:
            self.base_exp_dir += f"_r{reset_rot_degree}"
            self.conf.put("train.reset_rot_threshold", reset_rot_degree)
        if image_interval > 0:
            self.base_exp_dir += f"_i{image_interval}"
            self.conf.put("train.image_interval", image_interval)
            self.conf.put("train.max_pro_iteration", 1000 * image_interval)
            self.conf.put("train.pro_warm_up_end", 500 * image_interval)
            self.conf.put(
                "train.current_image", image_interval
            )  # this is very important!!!
        if flow_interval > 0 or reset_rot_degree > 0 or image_interval > 0:
            # save_freq = 50000
            self.conf.put("train.save_freq", 30000)
            print("updated confs--------------")
            print(self.conf)
            print("updated confs--------------")

        if start_img_idx > 0:
            self.base_exp_dir += f"_start_at_{start_img_idx}"
        os.makedirs(self.base_exp_dir, exist_ok=True)
        self.conf.put("dataset.start_idx", start_img_idx)

        self.dataset = Dataset(self.conf["dataset"], exp_dir)
        self.iter_step = 0

        # Training parameters
        self.end_iter = self.conf.get_int("train.end_iter")
        self.save_freq = self.conf.get_int("train.save_freq")
        self.report_freq = self.conf.get_int("train.report_freq")
        self.val_freq = self.conf.get_int("train.val_freq")
        self.val_mesh_freq = self.conf.get_int("train.val_mesh_freq")
        self.pose_freq = self.conf.get_int("train.pose_freq", 1000)
        self.batch_size = self.conf.get_int("train.batch_size")
        self.validate_resolution_level = self.conf.get_int(
            "train.validate_resolution_level"
        )
        self.learning_rate = self.conf.get_float("train.learning_rate")
        self.learning_rate_alpha = self.conf.get_float("train.learning_rate_alpha")
        self.use_white_bkgd = self.conf.get_bool("train.use_white_bkgd")
        self.warm_up_end = self.conf.get_float("train.warm_up_end", default=0.0)
        self.anneal_end = self.conf.get_float("train.anneal_end", default=0.0)
        self.mask_guided_sampling = self.conf.get_bool(
            "train.mask_guided_sampling", default=False
        )

        # Weights
        self.igr_weight = self.conf.get_float("train.igr_weight")
        self.mask_weight = self.conf.get_float("train.mask_weight")
        self.flow_weight = self.conf.get_float("train.flow_weight", default=0.0)
        self.unit_sphere_weight = self.conf.get_float(
            "train.unit_sphere_weight", default=0.0
        )
        self.depth_weight = self.conf.get_float("train.depth_weight", default=0.0)
        self.is_continue = is_continue
        self.mode = mode
        self.model_list = []
        self.writer = None

        # Networks
        self.progressive = self.conf.get_bool("train.progressive", default=False)
        self.image_interval = self.conf.get_int("train.image_interval", default=10)
        self.current_image = self.conf.get_int(
            "train.current_image", default=self.dataset.n_images
        )
        self.current_image = min(self.current_image, self.dataset.n_images)
        self.max_pro_iteration = self.conf.get_int("train.max_pro_iteration", default=0)
        self.pro_warm_up_end = self.conf.get_int("train.pro_warm_up_end", default=0)

        params_to_train = []
        self.nerf_outside = NeRF(**self.conf["model.nerf"]).to(self.device)
        self.deviation_network = SingleVarianceNetwork(
            **self.conf["model.variance_network"]
        ).to(self.device)
        if "model.barf" not in self.conf:
            self.conf.put("model.barf", False)
        self.mask_init = self.conf.get_bool("dataset.mask_init", default=False)
        self.mono_init = self.conf.get_bool("train.mono_init", default=False)
        self.mesh_warmup_step = self.conf.get_int("train.mesh_warmup_step", default=0)
        if self.conf["model.barf"]:
            print("Using BARF...")
            if self.conf.get("dataset.use_crop_init", False):
                # this is for 2nd phase to refine poses
                noise_poses = self.dataset.crop_poses
            else:
                # this is for 1st phase to calculate poss
                if self.mask_init:
                    # we only support the mask initialization method, which is most effective
                    noise_poses = self.dataset.max_mask_pose[None, ...].repeat(
                        self.dataset.n_images, 1, 1
                    )
                else:
                    raise NotImplementedError
            self.color_network = BarfRenderingNetwork(
                **self.conf["model.rendering_network"]
            ).to(self.device)
            self.sdf_network = BarfSDFNetwork(
                noise_poses=noise_poses,
                **self.conf["model.sdf_network"],
                n_images=self.dataset.n_images,
                barf=self.conf["model.barf"],
            ).to(self.device)
        else:
            self.color_network = RenderingNetwork(
                **self.conf["model.rendering_network"]
            ).to(self.device)
            self.sdf_network = SDFNetwork(**self.conf["model.sdf_network"]).to(
                self.device
            )

        self.pose_type = self.conf.get("model.pose_type", default="None")
        self.pose_lr = self.conf.get("train.pose_lr", default=5e-4)
        self.pose_alpha = self.conf.get("train.pose_alpha", default=0.5)
        self.current_pose_mlp_index = 0
        self.pro_iteration = 0
        if self.pose_type == "gf":
            print("Using GF...")
            self.sdf_network.se3_refine.weight.requires_grad_(
                False
            )  # we don't have gradients on this
            self.pose_network = LearnPoseGF(
                num_cams=self.dataset.n_images, init_c2w=noise_poses
            ).to(self.device)
            params_to_train += list(self.pose_network.parameters())
        elif self.pose_type == "seg":
            print("Using Seg...")
            self.sdf_network.se3_refine.weight.requires_grad_(
                False
            )  # we don't have gradients on this
            self.pixel_level = self.conf.get_bool("model.pixel_level", default=False)
            if self.pixel_level:
                print("Using pixel level...")
                self.pose_network = SegDeepPixelPose(
                    num_cams=self.dataset.n_images,
                    segment_img_num=self.image_interval,
                    init_c2w=noise_poses,
                ).to(self.device)
            else:
                print("Using picture level...")
                emphasize_rot = self.conf.get("train.emphasize_rot", False)
                small_rot = self.conf.get("train.small_rot", False)
                if emphasize_rot:
                    print("Using emphasize rotation...")
                self.pose_network = SegLearnPose(
                    num_cams=self.dataset.n_images,
                    segment_img_num=self.image_interval,
                    init_c2w=noise_poses,
                    emphasize_rot=emphasize_rot,
                    small_rot=small_rot,
                ).to(self.device)
            self.pose_optimizers = []
            for pose_mlp in self.pose_network.pose_mlps:
                self.pose_optimizers.append(
                    torch.optim.Adam(pose_mlp.parameters(), lr=self.pose_lr)
                )

        params_to_train += list(self.nerf_outside.parameters())
        params_to_train += list(self.sdf_network.parameters())
        params_to_train += list(self.deviation_network.parameters())
        params_to_train += list(self.color_network.parameters())

        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)

        self.renderer = NeuSRenderer(
            self.nerf_outside,
            self.sdf_network,
            self.deviation_network,
            self.color_network,
            **self.conf["model.neus_renderer"],
        )

        # Load checkpoint
        latest_model_name = None
        if start_at > 0:
            assert False
            latest_model_name = "ckpt_{:0>6d}.pth".format(start_at)
        else:
            if is_continue:
                model_list_raw = os.listdir(
                    os.path.join(self.base_exp_dir, "checkpoints")
                )
                model_list = []
                for model_name in model_list_raw:
                    if (
                        model_name[-3:] == "pth" and True
                    ):  # int(model_name[5:-4]) <= self.end_iter:
                        model_list.append(model_name)
                model_list.sort()
                latest_model_name = model_list[-1]

        self.disable_trans_during_warm_up = self.conf.get(
            "train.disable_trans_during_warm_up", False
        )
        if self.disable_trans_during_warm_up:
            print("Disable translation during warm up...")

        self.reset_based_on_rot = self.conf.get_bool(
            "train.reset_based_on_rot", default=False
        )
        if self.reset_based_on_rot:
            print("Reset based on rotation...")
            self.prev_pose = None
            self.reset_rot_threshold = self.conf.get_float(
                "train.reset_rot_threshold", default=60
            )

        if latest_model_name is not None:
            logging.info("Find checkpoint: {}".format(latest_model_name))
            self.load_checkpoint(latest_model_name)

        # Backup codes and configs for debug
        if self.mode[:5] == "train":
            self.file_backup()

        self.dataset.n_images = self.conf.get_int(
            "dataset.n_images", default=self.dataset.n_images
        )
        print(f"n_images: {self.dataset.n_images}")
        self.order_sample = self.conf.get_bool("train.order_sample", default=False)
        self.detach_ref = self.conf.get_bool("train.detach_ref", default=False)
        self.flow_interval = self.conf.get("train.flow_interval", default=1)
        print(
            f"order_sample: {self.order_sample}, detach_ref: {self.detach_ref}, flow_interval: {self.flow_interval}"
        )
        self.only_rotation = self.conf.get_bool("train.only_rotation", default=False)

        self.detach_flow_on_sdf = self.conf.get_bool(
            "train.detach_flow_on_sdf", default=False
        )

        self.ensure_sample_inside_sphere = self.conf.get_bool(
            "train.ensure_sample_inside_sphere", default=False
        )
        if self.ensure_sample_inside_sphere:
            print("Ensure sample inside sphere...")

        self.detach_mesh_at_warm_up = self.conf.get_bool(
            "train.detach_mesh_at_warm_up", default=False
        )
        if self.detach_mesh_at_warm_up:
            print("Detach mesh at warm up...")

        self.dynamic_pro_iterations = self.conf.get_bool(
            "train.dynamic_pro_iterations", default=False
        )
        if self.dynamic_pro_iterations:
            print("Dynamic pro iterations...")

        self.mask_guided_patch_size = self.conf.get_int(
            "train.mask_guided_patch_size", default=30
        )
        self.maintain_shape = self.conf.get_bool("train.maintain_shape", default=False)

        self.remove_prev_matches = self.conf.get_bool(
            "train.remove_prev_matches", default=True
        )
        if self.remove_prev_matches:
            print("Remove previous matches...")

    def reset_neus(self):
        self.nerf_outside = NeRF(**self.conf["model.nerf"]).to(self.device)
        self.deviation_network = SingleVarianceNetwork(
            **self.conf["model.variance_network"]
        ).to(self.device)
        noise_poses = torch.eye(4)[None, ...].repeat(self.dataset.n_images, 1, 1)
        self.sdf_network = BarfSDFNetwork(
            noise_poses=noise_poses,
            **self.conf["model.sdf_network"],
            n_images=self.dataset.n_images,
            barf=self.conf["model.barf"],
        ).to(self.device)
        self.color_network = BarfRenderingNetwork(
            **self.conf["model.rendering_network"]
        ).to(self.device)
        params_to_train = []
        params_to_train += list(self.nerf_outside.parameters())
        params_to_train += list(self.sdf_network.parameters())
        params_to_train += list(self.deviation_network.parameters())
        params_to_train += list(self.color_network.parameters())
        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)
        self.renderer = NeuSRenderer(
            self.nerf_outside,
            self.sdf_network,
            self.deviation_network,
            self.color_network,
            **self.conf["model.neus_renderer"],
        )
        self.iter_step = 0  # we need warming up neus again!
        self.mesh_warmup_step = self.conf.get_int("train.mesh_warmup_step", default=0)
        pass

    def train(self):
        # assert False, "we need warm up the neus?"
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, "logs"))
        if self.pose_type != "seg":
            self.update_learning_rate()
        res_step = self.end_iter - self.iter_step
        if self.maintain_shape:
            image_perm = self.get_current_image_perm()
            prev_image_perm = self.get_prev_image_perm()
        else:
            image_perm = self.get_image_perm()

        for iter_i in range(res_step):
            pose_all = self.dataset.pose_all
            if self.conf["model.barf"]:
                self.sdf_network.progress.data.fill_(iter_i / res_step)
                self.color_network.progress.data.fill_(iter_i / res_step)
                if self.pose_type in ["gf", "seg"]:
                    pose_all = None  # can remind that it requires the pose network
                else:
                    pose_refine = camera.lie.se3_to_SE3(
                        self.sdf_network.se3_refine.weight, only_rot=self.only_rotation
                    )
                    pose_all = camera.pose.compose(
                        [pose_refine, self.sdf_network.noise_poses[:, :3, :]]
                    )

            if (
                self.flow_weight > 0.0
                and np.random.rand() < 0.5
                and self.iter_step > self.mesh_warmup_step
            ):
                use_flow = True
            else:
                use_flow = False
            if self.remove_prev_matches:
                # the match quality is not reliable
                if np.abs(
                    image_perm[self.iter_step % len(image_perm)].item()
                    - self.current_image
                ) >= self.flow_interval or (
                    self.current_image == self.dataset.n_images
                ):
                    use_flow = False
            if use_flow:
                img_id_corr = image_perm[self.iter_step % len(image_perm)]
                data, pixels_xy, pixels_xy_corr, img_id, depth = (
                    self.dataset.gen_random_ray_pairs_at(
                        img_id_corr,
                        self.batch_size // 2,
                        pose_network=self.pose_network,
                        current_img_num=self.current_image,
                        interval=self.flow_interval,
                    )
                )
                if data is None:
                    img_id = image_perm[self.iter_step % len(image_perm)]
                    if self.pose_type in ["gf", "seg"]:
                        pose = self.pose_network(img_id_corr)[:3]
                    else:
                        pose = pose_all[img_id_corr, :3]
                    data, depth = self.dataset.gen_random_rays_at(
                        img_id,
                        self.batch_size,
                        pose=pose,
                        mask_guided_sampling=self.mask_guided_sampling
                        and self.iter_step > self.mesh_warmup_step,
                    )
                    use_flow = False
                else:
                    pass
                    # print("we get matches between", img_id_corr, img_id)
            else:
                img_id = image_perm[self.iter_step % len(image_perm)]
                if self.iter_step < self.mesh_warmup_step:
                    if self.pose_type != "gf":
                        if self.pose_type == "seg":
                            for i in range(len(self.pose_network.pose_mlps)):
                                self.pose_network.pose_mlps[i].disable_grad()
                        else:
                            self.sdf_network.se3_refine.requires_grad_(False)
                        if self.reset_based_on_rot and self.prev_pose is not None:
                            # select images from [0, self.current_image - 1]
                            img_id = torch.tensor(
                                np.random.randint(0, self.current_image)
                            ).long()
                        else:
                            img_id = torch.tensor(0).long()
                else:
                    if self.pose_type != "gf":
                        if self.mesh_warmup_step > 0:
                            self.mesh_warmup_step = 0
                            if self.pose_type == "seg":
                                for i in range(len(self.pose_network.pose_mlps)):
                                    self.pose_network.pose_mlps[i].enable_grad()
                            else:
                                self.sdf_network.se3_refine.requires_grad_(True)

                if self.pose_type in ["gf", "seg"]:
                    pose = self.pose_network(img_id)[:3]
                else:
                    pose = pose_all[img_id, :3]

                data, depth = self.dataset.gen_random_rays_at(
                    img_id,
                    self.batch_size,
                    pose=pose,
                    mask_guided_sampling=self.mask_guided_sampling
                    and self.iter_step > self.mesh_warmup_step,
                )
                img_id_corr = None

            # we additionally sample self.batch_size from previous images
            if self.maintain_shape:
                additional_img_id = prev_image_perm[
                    self.iter_step % len(prev_image_perm)
                ]
                if self.iter_step < self.mesh_warmup_step:
                    if self.pose_type != "gf":
                        if self.pose_type == "seg":
                            for i in range(len(self.pose_network.pose_mlps)):
                                self.pose_network.pose_mlps[i].disable_grad()
                        else:
                            self.sdf_network.se3_refine.requires_grad_(False)
                        additional_img_id = torch.tensor(0).long()
                else:
                    if self.pose_type != "gf":
                        if self.mesh_warmup_step > 0:
                            self.mesh_warmup_step = 0
                            if self.pose_type == "seg":
                                for i in range(len(self.pose_network.pose_mlps)):
                                    self.pose_network.pose_mlps[i].enable_grad()
                            else:
                                self.sdf_network.se3_refine.requires_grad_(True)

                if self.pose_type in ["gf", "seg"]:
                    pose = self.pose_network(additional_img_id)[:3]
                else:
                    pose = pose_all[additional_img_id, :3]

                add_data, add_depth = self.dataset.gen_random_rays_at(
                    additional_img_id,
                    self.batch_size,
                    pose=pose,
                    mask_guided_sampling=self.mask_guided_sampling
                    and self.iter_step > self.mesh_warmup_step,
                )
                data = torch.cat([data, add_data], dim=0)
                if add_depth is not None:
                    depth = torch.cat([depth, add_depth], dim=0)

            rays_o, rays_d, true_rgb, mask = (
                data[:, :3],
                data[:, 3:6],
                data[:, 6:9],
                data[:, 9:10],
            )

            near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)
            background_rgb = None
            if self.use_white_bkgd:
                background_rgb = torch.ones([1, 3])

            if self.mask_weight > 0.0:
                mask = (mask > 0.5).float()
            else:
                mask = torch.ones_like(mask)

            mask_sum = mask.sum() + 1e-5
            render_out = self.renderer.render(
                rays_o,
                rays_d,
                near,
                far,
                background_rgb=background_rgb,
                cos_anneal_ratio=self.get_cos_anneal_ratio(),
            )

            color_fine = render_out["color_fine"]
            s_val = render_out["s_val"]
            cdf_fine = render_out["cdf_fine"]
            gradient_error = render_out["gradient_error"]
            weight_max = render_out["weight_max"]
            weight_sum = render_out["weight_sum"]

            # Loss
            color_error = (color_fine - true_rgb) * mask
            color_fine_loss = (
                F.l1_loss(color_error, torch.zeros_like(color_error), reduction="sum")
                / mask_sum
            )
            psnr = 20.0 * torch.log10(
                1.0
                / (
                    ((color_fine - true_rgb) ** 2 * mask).sum() / (mask_sum * 3.0)
                ).sqrt()
            )

            eikonal_loss = gradient_error

            mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)

            # test the norms of pts
            # if self.iter_step % 100 == 0:
            #     print(f"max pts norm: {render_out['pts'].norm(dim=-1).max()}")

            if self.flow_weight > 0.0 and use_flow:
                # order img_id_corr, img_id
                pts = render_out["pts"]  # (N, 3)
                weights = render_out["weights"]
                if self.detach_flow_on_sdf:
                    weights = weights.detach()
                # project the points to the next image
                # we should sperate based on flags
                pts_N = pts.shape[0]
                weights_N = weights.shape[0]
                if self.maintain_shape:
                    pts0 = pts[: pts_N // 4]
                    pts1 = pts[pts_N // 4 : pts_N // 2]
                    weights0 = weights[: weights_N // 4]
                    weights1 = weights[weights_N // 4 : weights_N // 2]
                else:
                    pts0 = pts[: pts_N // 2]
                    pts1 = pts[pts_N // 2 :]
                    weights0 = weights[: weights_N // 2]
                    weights1 = weights[weights_N // 2 :]

                # the following is pose for img_id, which is "1"
                c2w_1 = torch.eye(4)
                if self.pose_type in ["gf", "seg"]:
                    c2w_1[:3] = self.pose_network(img_id)[:3]
                else:
                    c2w_1[:3] = pose_all[img_id]
                if self.detach_ref:
                    c2w_1 = c2w_1.detach()
                w2c_1 = torch.inverse(c2w_1)[:3][None, ...].expand(
                    pts0.shape[0], -1, -1
                )
                # first half points project points to the next image based on "1"'s w2c
                cam_pts = (w2c_1 @ to_hom(pts0).unsqueeze(-1)).squeeze(
                    -1
                )  # (N, 3, 4) @ (N, 4, 1) -> (N, 3, 1) -> (N, 3)
                K = self.dataset.intrinsics_all[img_id][:3, :3][None, ...].expand(
                    pts0.shape[0], -1, -1
                )
                pixel_pts = (K @ cam_pts.unsqueeze(-1)).squeeze(
                    -1
                )  # (N, 3, 3) @ (N, 3, 1) -> (N, 3, 1) -> (N, 3)
                pixel_pts = pixel_pts[:, :2] / pixel_pts[:, 2:]  # (N, 2)
                pixel_pts = pixel_pts.reshape(-1, weights0.shape[1], 2)
                pixels_xy = pixels_xy[:, None, :].expand(-1, weights0.shape[1], -1)
                pixel_error = ((pixel_pts - pixels_xy) * weights0[:, :, None]).sum(
                    dim=1
                )
                flow_loss0 = (
                    F.l1_loss(pixel_error, torch.zeros_like(pixel_error))
                    * self.flow_weight
                )

                # the following is pose for img_id_corr, which is "0"
                c2w_0 = torch.eye(4)
                if self.pose_type in ["gf", "seg"]:
                    c2w_0[:3] = self.pose_network(img_id_corr)[:3]
                else:
                    c2w_0[:3] = pose_all[img_id_corr]
                if self.detach_ref:
                    c2w_0 = c2w_0.detach()
                w2c_0 = torch.inverse(c2w_0)[:3][None, ...].expand(
                    pts1.shape[0], -1, -1
                )

                # project pts1 to the img_id_corr
                cam_pts = (w2c_0 @ to_hom(pts1).unsqueeze(-1)).squeeze(
                    -1
                )  # (N, 3, 4) @ (N, 4, 1) -> (N, 3, 1) -> (N, 3)
                K = self.dataset.intrinsics_all[img_id_corr][:3, :3][None, ...].expand(
                    pts1.shape[0], -1, -1
                )
                pixel_pts = (K @ cam_pts.unsqueeze(-1)).squeeze(
                    -1
                )  # (N, 3, 3) @ (N, 3, 1) -> (N, 3, 1) -> (N, 3)
                pixel_pts = pixel_pts[:, :2] / pixel_pts[:, 2:]  # (N, 2)
                pixel_pts = pixel_pts.reshape(-1, weights1.shape[1], 2)
                pixels_xy_corr = pixels_xy_corr[:, None, :].expand(
                    -1, weights1.shape[1], -1
                )
                pixel_error = ((pixel_pts - pixels_xy_corr) * weights1[:, :, None]).sum(
                    dim=1
                )
                flow_loss1 = (
                    F.l1_loss(pixel_error, torch.zeros_like(pixel_error))
                    * self.flow_weight
                )

                flow_loss = flow_loss0 + flow_loss1
                # print(f"flow_loss: {flow_loss}")
            else:
                flow_loss = 0

            if self.depth_weight > 0.0:
                depth_fine = render_out["depth_fine"]
                depth_mask = ((mask > 0.5) & (depth > 0)).reshape(-1)
                depth_fine = depth_fine[depth_mask]
                depth = depth[depth_mask]
                if depth.shape[0] == 0 or depth_fine.shape[0] == 0:
                    depth_loss = 0
                else:
                    depth_loss = F.l1_loss(depth_fine, depth) * self.depth_weight
            else:
                depth_loss = 0

            # assert depth loss is not nan
            if depth_loss != 0:
                assert not torch.isnan(depth_loss), f"depth_loss: {depth_loss}"

            if self.unit_sphere_weight > 0:
                pts = render_out["pts"]
                weights = render_out["weights"].reshape(-1, 1)
                outside_mask = (pts.norm(dim=-1) > 1.0).detach()
                weights = weights[outside_mask]
                unit_sphere_loss = (
                    F.l1_loss(weights, torch.zeros_like(weights))
                    * self.unit_sphere_weight
                )
            else:
                unit_sphere_loss = 0

            if self.gradient_analysis:
                # we want verify the balance of the loss weight
                losses = {
                    "flow_loss": flow_loss,
                    "color_fine_loss": color_fine_loss,
                    "mask_loss": mask_loss,
                    "unit_sphere_loss": unit_sphere_loss,
                    "depth_loss": depth_loss,
                    "eikonal_loss": eikonal_loss,
                }
                params = {
                    "sdf_network": self.sdf_network,
                    "pose_network": self.pose_network,
                }
                for k, v in losses.items():
                    print(k, v)
                    self.optimizer.zero_grad()
                    if v > 0:
                        v.backward(retain_graph=True)
                        for name, param in params.items():
                            min_grad, max_grad, mean_grad = get_gradients(
                                param.parameters()
                            )
                            self.writer.add_scalar(
                                f"Gradients/{k}_min_{name}", min_grad, self.iter_step
                            )
                            self.writer.add_scalar(
                                f"Gradients/{k}_max_{name}", max_grad, self.iter_step
                            )
                            self.writer.add_scalar(
                                f"Gradients/{k}_mean_{name}", mean_grad, self.iter_step
                            )
                            print(
                                f"Gradients/{k}_{name}",
                                round(min_grad, 5),
                                round(max_grad, 5),
                                round(mean_grad, 5),
                                self.iter_step,
                            )
                    else:
                        print(f"Gradients/{k}_min", 0, 0, 0, self.iter_step)
                print("--------------------")

                self.optimizer.zero_grad()

            # loss will affect all models
            loss = (
                color_fine_loss
                + eikonal_loss * self.igr_weight
                + mask_loss * self.mask_weight
                + unit_sphere_loss
                + flow_loss
                + depth_loss
            )

            # self.optimizer.zero_grad() # so we would only use depth to supervise the sdf network
            # if depth_loss > 0:
            #     # we only affect sdf network
            #     depth_loss.backward(retain_graph=True)

            pose_mlp_index_set = set([img_id.item() // self.image_interval])
            if img_id_corr is not None:
                pose_mlp_index_set.add(img_id_corr.item() // self.image_interval)

            if self.maintain_shape:
                pose_mlp_index_set.add(additional_img_id.item() // self.image_interval)

            if self.pose_type == "seg":
                for pose_mlp_index in pose_mlp_index_set:
                    self.pose_optimizers[pose_mlp_index].zero_grad()

            # flow loss will only affect pose network
            # if flow_loss > 0:
            #     flow_loss.backward(retain_graph=True)

            self.optimizer.zero_grad()  # this will remove gradients on sdf network from flow loss
            loss.backward()

            if self.detach_mesh_at_warm_up and self.iter_step > self.mesh_warmup_step:
                if (
                    self.pro_iteration < self.pro_warm_up_end
                    and self.current_pose_mlp_index in pose_mlp_index_set
                ):
                    # disable the gradients of the current pose_mlp
                    self.optimizer.zero_grad()

            self.optimizer.step()

            if self.pose_type == "seg":
                for pose_mlp_index in pose_mlp_index_set:
                    self.pose_optimizers[pose_mlp_index].step()

            self.iter_step += 1

            self.writer.add_scalar("Loss/loss", loss, self.iter_step)
            self.writer.add_scalar("Loss/color_loss", color_fine_loss, self.iter_step)
            self.writer.add_scalar("Loss/eikonal_loss", eikonal_loss, self.iter_step)
            self.writer.add_scalar("Loss/mask_loss", mask_loss, self.iter_step)
            self.writer.add_scalar("Loss/flow_loss", flow_loss, self.iter_step)
            self.writer.add_scalar("Loss/depth_loss", depth_loss, self.iter_step)
            self.writer.add_scalar(
                "Loss/unit_sphere_loss", unit_sphere_loss, self.iter_step
            )
            self.writer.add_scalar("Statistics/s_val", s_val.mean(), self.iter_step)
            self.writer.add_scalar(
                "Statistics/cdf",
                (cdf_fine[:, :1] * mask).sum() / mask_sum,
                self.iter_step,
            )
            self.writer.add_scalar(
                "Statistics/weight_max",
                (weight_max * mask).sum() / mask_sum,
                self.iter_step,
            )
            self.writer.add_scalar("Statistics/psnr", psnr, self.iter_step)

            if self.iter_step % self.report_freq == 0:
                print(self.base_exp_dir)
                print(
                    "iter:{:8>d} loss = {} lr={}".format(
                        self.iter_step, loss, self.optimizer.param_groups[0]["lr"]
                    )
                )
                # print all losses
                print(
                    f"color_fine_loss: {color_fine_loss}, eikonal_loss: {eikonal_loss}, mask_loss: {mask_loss}, flow_loss: {flow_loss}, depth_loss: {depth_loss}, unit_sphere_loss: {unit_sphere_loss}"
                )

            if self.iter_step % self.val_freq == 0:
                self.validate_image()

            if self.iter_step % self.pose_freq == 0:
                self.validate_poses()

            if (
                self.pose_type == "seg"
                and self.pro_iteration >= 0
                and self.iter_step > self.mesh_warmup_step
            ):
                self.pro_iteration += 1
                if self.pro_iteration == self.max_pro_iteration:
                    # save the depth map of current image
                    self.pro_iteration = 0
                    prev_image = self.current_image
                    self.current_image = min(
                        self.current_image + self.image_interval, self.dataset.n_images
                    )
                    if self.current_image > prev_image:
                        if self.reset_based_on_rot:

                            def rotation_error(pose_error):
                                a = pose_error[0, 0]
                                b = pose_error[1, 1]
                                c = pose_error[2, 2]
                                d = 0.5 * (a + b + c - 1.0)
                                rot_error = np.arccos(max(min(d, 1.0), -1.0))
                                return rot_error * 180 / np.pi

                            with torch.no_grad():
                                if self.prev_pose is None:
                                    self.prev_pose = self.pose_network.pose_mlps[0](
                                        torch.tensor(0).long()
                                    )[:3, :3].cpu()
                                cur_pose = self.pose_network.pose_mlps[
                                    self.current_pose_mlp_index
                                ](torch.tensor(prev_image - 1).long())[:3, :3].cpu()
                            rel_R = (
                                (cur_pose @ torch.inverse(self.prev_pose)).cpu().numpy()
                            )
                            if rotation_error(rel_R) > self.reset_rot_threshold:
                                print("reset based on rotation...")
                                self.reset_neus()
                                self.prev_pose = cur_pose
                        prev_pose_mlp_index = self.current_pose_mlp_index
                        self.current_pose_mlp_index += 1  # pose mlp index increase by 1 since image intavel frames are in a group!
                        if self.dynamic_pro_iterations:
                            self.max_pro_iteration = (
                                self.dataset.pro_iteration_at_frame[
                                    self.current_pose_mlp_index
                                ]
                            )
                            self.pro_warm_up_end = self.max_pro_iteration // 3
                            print(
                                "dynamic pro iteration",
                                f"max_pro_iteration: {self.max_pro_iteration}, pro_warm_up_end: {self.pro_warm_up_end}",
                            )
                        for i in range(prev_pose_mlp_index + 1):
                            # disable the gradients of the previous pose_mlp
                            # and only enable the gradients of the current pose_mlp
                            self.pose_network.pose_mlps[i].disable_grad()
                    else:
                        # finish feeding frames
                        self.pro_iteration = -1

                    if self.disable_trans_during_warm_up:
                        self.pose_network.pose_mlps[
                            self.current_pose_mlp_index
                        ].disable_trans()
                    print("reach max pro iteration.........")
                    print("current_image: ", self.current_image)
                    print("current_pose_mlp_index: ", self.current_pose_mlp_index)
                if self.pro_iteration == self.pro_warm_up_end:
                    print("finish warm up...")
                    if False:  # self.maintain_shape:
                        # we would fix the stable poses of previous images
                        for i in range(1, self.flow_interval):
                            if self.current_pose_mlp_index - i >= 0:
                                self.pose_network.pose_mlps[
                                    self.current_pose_mlp_index - i
                                ].enable_grad()
                    else:
                        for i in range(self.current_pose_mlp_index):
                            # enable the gradients of the previous pose_mlp
                            self.pose_network.pose_mlps[i].enable_grad()
                    # the following enables predicting the translation
                    self.pose_network.pose_mlps[
                        self.current_pose_mlp_index
                    ].finish_warmup()
                    if self.disable_trans_during_warm_up:
                        self.pose_network.pose_mlps[
                            self.current_pose_mlp_index
                        ].enable_trans()

            if self.iter_step % self.val_mesh_freq == 0:
                self.validate_mesh()

            self.update_learning_rate(pose_mlp_index_set)

            if self.iter_step % len(image_perm) == 0:
                if self.maintain_shape:
                    image_perm = self.get_current_image_perm()
                else:
                    image_perm = self.get_image_perm()

            if self.maintain_shape and self.iter_step % len(prev_image_perm) == 0:
                prev_image_perm = self.get_prev_image_perm()

            # saving checkpoint should happen at last
            if self.iter_step % self.save_freq == 0 and self.iter_step > 0:
                self.save_checkpoint()

            if "_wo_global_conf" not in self.base_exp_dir:
                # only if we have global conf, we can reboot
                if (
                    self.pro_iteration == -1
                    and self.current_image == self.dataset.n_images
                ):
                    self.validate_mesh()
                    self.save_checkpoint()  # we save the final model
                    # we are ready to reboot for global training
                    return
        poses = self.dataset.pose_all  # Assuming this is where the poses are stored
        image_names = [os.path.basename(image_path) for image_path in self.dataset.images_lis]  # Get the image file names  # Example image names
        output_csv = os.path.join(self.base_exp_dir, f'{self.case}_camera_poses.csv')
        extract_camera_poses(poses, image_names, output_csv)
        
    def get_image_perm(self):
        if self.progressive:
            if self.current_image > self.image_interval:
                # 80% for [current_image - self.image_interval, current_image - 1], 20% for [0, current_image - self.image_interval - 1]
                prev_img_num = self.current_image - self.image_interval
                prev_weight = [0.2 / (prev_img_num)] * prev_img_num
                cur_weight = [0.8 / (self.image_interval)] * self.image_interval
                weight = prev_weight + cur_weight
                # use numpy to randomly select the index
                indexes = np.random.choice(
                    self.current_image, self.current_image, p=weight
                )
                return torch.tensor(indexes)
            else:
                return torch.randperm(self.current_image)
        else:
            return torch.randperm(self.dataset.n_images)

    def get_prev_image_perm(self):
        if self.current_image > self.flow_interval:
            prev_img_num = self.current_image - self.flow_interval
            return torch.randperm(prev_img_num)
        else:
            return torch.randperm(self.current_image)

    def get_current_image_perm(self):
        # current image has 80% possibility to be selected, while the previous image(inside flow interval) has 20% possibility to be selected
        # if self.current_image > self.flow_interval:
        if self.current_image > (self.image_interval - 1) + self.flow_interval:
            # option-1: only support image_interval = 1
            # prev_img_num = self.current_image - self.flow_interval
            # if self.flow_interval == 1:
            #     return torch.tensor([prev_img_num])
            # prev_weight = [0.2 / (self.flow_interval - 1)] * (self.flow_interval - 1)
            # cur_weight = [0.8]
            # weight = prev_weight + cur_weight
            # # use numpy to randomly select the index: [0, self.flow_interval - 1] + [prev_img_num]
            # indexes = np.random.choice(self.flow_interval, self.flow_interval, p=weight) + prev_img_num
            # return torch.tensor(indexes)

            # option-2: supporting more frames
            if self.flow_interval == 1:
                # randomly select self.image_interval
                return (
                    torch.randperm(self.image_interval)
                    + self.current_image
                    - self.image_interval
                )
            prev_img_num = (
                self.current_image - (self.image_interval - 1) - self.flow_interval
            )
            prev_weight = [0.2 / (self.flow_interval - 1)] * (self.flow_interval - 1)
            cur_weight = [0.8 / (self.image_interval)] * self.image_interval
            weight = prev_weight + cur_weight
            indexes = (
                np.random.choice(len(weight), len(weight), p=weight) + prev_img_num
            )
            return torch.tensor(indexes)
        else:
            return torch.randperm(self.current_image)

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])

    def update_learning_rate(self, pose_mlp_index_set=None):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (
                self.end_iter - self.warm_up_end
            )
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (
                1 - alpha
            ) + alpha

        for g in self.optimizer.param_groups:
            g["lr"] = self.learning_rate * learning_factor

        if self.pose_type == "seg":
            for pose_mlp_index in pose_mlp_index_set:
                step = self.pose_network.step_progress(pose_mlp_index).item()
                if "_wo_global_conf" not in self.base_exp_dir:
                    if False:  # step <= self.warm_up_end:
                        learning_factor = step / self.warm_up_end
                    else:
                        progress = (
                            step / self.max_pro_iteration
                        )  # (step - self.warm_up_end) / (self.max_pro_iteration - self.warm_up_end)
                        alpha = self.pose_alpha
                        learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (
                            1 - alpha
                        ) + alpha
                else:
                    # we will schedule the learning rate based on the global progress
                    progress = step / self.end_iter
                    alpha = self.learning_rate_alpha
                    learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (
                        1 - alpha
                    ) + alpha
                pose_optimizer = self.pose_optimizers[pose_mlp_index]
                for param_group in pose_optimizer.param_groups:
                    param_group["lr"] = self.pose_lr * learning_factor

    def file_backup(self):
        dir_lis = self.conf["general.recording"]
        os.makedirs(os.path.join(self.base_exp_dir, "recording"), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, "recording", dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == ".py":
                    copyfile(
                        os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name)
                    )
        try:
            copyfile(
                self.conf_path,
                os.path.join(self.base_exp_dir, "recording", "config.conf"),
            )
        except:
            pass

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(
            os.path.join(self.base_exp_dir, "checkpoints", checkpoint_name),
            map_location=self.device,
        )
        self.nerf_outside.load_state_dict(checkpoint["nerf"])
        self.sdf_network.load_state_dict(checkpoint["sdf_network_fine"])
        self.deviation_network.load_state_dict(checkpoint["variance_network_fine"])
        self.color_network.load_state_dict(checkpoint["color_network_fine"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.iter_step = checkpoint["iter_step"]

        if "current_image" in checkpoint:
            self.current_image = checkpoint["current_image"]

        if self.pose_type in ["gf", "seg"]:
            self.pose_network.load_state_dict(checkpoint["pose_network"])

        if self.pose_type == "seg":
            self.current_pose_mlp_index = checkpoint["current_pose_mlp_index"]
            self.pro_iteration = checkpoint["pro_iteration"]

        if self.reset_based_on_rot:
            self.prev_pose = checkpoint["prev_pose"].cpu()

        print(f"current_image: {self.current_image}")
        print(f"iter_step: {self.iter_step}")
        print(f"current_pose_mlp_index: {self.current_pose_mlp_index}")
        print(f"pro_iteration: {self.pro_iteration}")
        print("total imgs", self.dataset.n_images)
        # be careful that the state_dict() would not help you save the grad property
        for i in range(self.current_pose_mlp_index):
            self.pose_network.pose_mlps[i].disable_grad()
        if self.disable_trans_during_warm_up:
            self.pose_network.pose_mlps[self.current_pose_mlp_index].disable_trans()
        logging.info("End")

    def validate_poses(self, save_pose=False, init_tag="", only_align=False):
        if self.mode != "train":
            if "global" in self.base_exp_dir:
                self.current_image = self.dataset.n_images
            if (
                self.current_image != self.dataset.n_images
                and "480" not in self.base_exp_dir
            ):
                self.current_image -= 10
        # collect all predicted poses
        if self.pose_type not in ["gf", "seg"]:
            # it is barf
            pose_all = self.sdf_network.noise_poses
            pose_refine = camera.lie.se3_to_SE3(
                self.sdf_network.se3_refine.weight, only_rot=self.only_rotation
            )
            pose_all = camera.pose.compose(
                [pose_refine, self.sdf_network.noise_poses[:, :3, :]]
            ).detach()
        else:
            pose_all = []
            with torch.no_grad():
                for i in range(self.current_image):
                    pose = self.pose_network(torch.tensor(i).cuda())
                    pose_all.append(pose[:3])
            pose_all = torch.stack(pose_all)
        pose_all4x4 = torch.eye(4)[None, ...].repeat(pose_all.shape[0], 1, 1)
        pose_all4x4[:, :3, :] = pose_all

        # collect the annotated pose
        gt_poses = []
        learned_poses = []
        if len(self.dataset.gt_poses) > 0:
            for i in range(len(self.dataset.avai_ann_frame)):
                # i correspond to gt poses and self.dataset.avai_ann_frame one by one
                frame_idx = self.dataset.avai_ann_frame[i]
                if frame_idx >= self.current_image:
                    break
                else:
                    gt_pose = self.dataset.gt_poses[i]
                    gt_poses.append(gt_pose)
                    learned_poses.append(pose_all4x4[frame_idx])

        if len(gt_poses) == 0:
            gt_poses = pose_all4x4[: self.current_image]
            learned_poses = pose_all4x4[: self.current_image]
        else:
            gt_poses = torch.stack(gt_poses)
            learned_poses = torch.stack(learned_poses)

        print("learned_poses.shape: ", learned_poses.shape)
        print("gt_poses.shape: ", gt_poses.shape)
        if only_align:
            aligns = [True]
        else:
            aligns = [False, True]
        for align in aligns:
            if len(self.dataset.gt_poses) == 0 and align:
                ate, rpe_trans, rpe_rot = float("inf"), float("inf"), float("inf")
                break
            try:
                c2ws_gt_aligned = gt_poses
                if align:
                    c2ws_est_aligned = align_ate_c2b_use_a2b(learned_poses, gt_poses)
                else:
                    c2ws_est_aligned = learned_poses
                ate = compute_ATE(
                    c2ws_gt_aligned.cpu().numpy(), c2ws_est_aligned.cpu().numpy()
                )
                rpe_trans, rpe_rot = compute_rpe(
                    c2ws_gt_aligned.cpu().numpy(), c2ws_est_aligned.cpu().numpy()
                )
                fx, fy = (
                    self.dataset.intrinsics_all[0, 0, 0],
                    self.dataset.intrinsics_all[0, 1, 1],
                )
                H, W = self.dataset.images[0].shape[:2]
                pose_dir = os.path.join(self.base_exp_dir, "poses")
                os.makedirs(pose_dir, exist_ok=True)
                img_name = (
                    "aligned_pose_{:0>6d}_{:0>6d}.png"
                    if align
                    else "raw_pose_{:0>6d}_{:0>6d}.png"
                )
                sufix = f"_{ate}_{rpe_trans}_{np.rad2deg(rpe_rot)}_mode={self.mode}{init_tag}.png"
                img_name = img_name.replace(".png", sufix)
                vis_poses(
                    c2ws_est_aligned.cpu(),
                    c2ws_gt_aligned.cpu(),
                    H,
                    W,
                    fx.cpu(),
                    fy.cpu(),
                    save_path=os.path.join(
                        self.base_exp_dir,
                        "poses",
                        img_name.format(self.current_image, self.iter_step),
                    ),
                )
                simple_pose_dir = os.path.join(self.base_exp_dir, "poses", "simple")
                os.makedirs(simple_pose_dir, exist_ok=True)
                if align:
                    vis_simple_traj(
                        c2ws_est_aligned.cpu(),
                        c2ws_gt_aligned.cpu(),
                        save_path=os.path.join(
                            simple_pose_dir,
                            img_name.format(self.current_image, self.iter_step),
                        ),
                    )
            except Exception as e:
                print("validate_pose error", e)
                ate, rpe_trans, rpe_rot = float("inf"), float("inf"), float("inf")
        print("ate: ", ate, "rpe_trans", rpe_trans, "rpe_rot", rpe_rot)
        # save pred poses and gt poses
        if save_pose:
            pose_arr_dir = os.path.join(self.base_exp_dir, "poses_arr")
            os.makedirs(pose_arr_dir, exist_ok=True)
            np.save(
                os.path.join(pose_arr_dir, f"pred_poses_{self.iter_step}.npy"),
                learned_poses.cpu().numpy(),
            )
            np.save(os.path.join(pose_arr_dir, "gt_poses.npy"), gt_poses.cpu().numpy())
            print("save poses to ", pose_arr_dir)

        return ate, rpe_trans, rpe_rot, gt_poses, learned_poses

    def save_poses(self):
        self.current_image -= 10
        self.validate_poses()
        pose_dir = os.path.join(self.base_exp_dir, "poses")
        os.makedirs(pose_dir, exist_ok=True)
        poses = []
        with torch.no_grad():
            for i in range(self.current_image):
                pose = self.pose_network(torch.tensor(i).cuda())
                pose4x4 = torch.eye(4)
                pose4x4[:3] = pose
                poses.append(pose4x4)
        poses = torch.stack(poses)
        np.save(
            os.path.join(pose_dir, f"pred_poses_{self.iter_step}.npy"),
            poses.cpu().numpy(),
        )
        # also save gt poses
        np.save(
            os.path.join(pose_dir, "gt_poses.npy"), self.dataset.gt_poses.cpu().numpy()
        )
        # we also need to save intrinsics
        np.save(
            os.path.join(pose_dir, "intrinsics.npy"),
            self.dataset.intrinsics_all.cpu().numpy(),
        )
        # we also need to save transform_matrixs
        transform_matrixs = []
        for i in range(len(poses)):
            transform_matrixs.append(
                self.dataset.crop_transforms[self.dataset.index_to_frame[i]]
            )
        transform_matrixs = np.stack(transform_matrixs, axis=0)
        np.save(os.path.join(pose_dir, "transform_matrixs.npy"), transform_matrixs)
        print("save poses to ", pose_dir)

    def save_poses_simple(self, align_dir=None, virtual=False):
        poses = {}
        if virtual:
            # find pose file
            for file in os.listdir(self.base_exp_dir):
                if "global_poses" in file and ".npy" in file:
                    virtual_pose = np.load(os.path.join(self.base_exp_dir, file))
            for i in range(virtual_pose.shape[0]):
                poses[self.dataset.index_to_frame[i]] = virtual_pose[i]
        else:
            with torch.no_grad():
                for i in range(self.current_image):
                    pose = self.pose_network(torch.tensor(i).cuda())
                    pose4x4 = torch.eye(4)
                    pose4x4[:3] = pose
                    poses[self.dataset.index_to_frame[i]] = pose4x4.cpu().numpy()
        if align_dir is not None:
            save_path = os.path.join(align_dir, f"{self.case}_poses.npy")
        else:
            save_path = os.path.join(self.base_exp_dir, f"poses_{self.iter_step}.npy")
        print("saving to ", save_path)
        np.save(save_path, poses)
        return save_path

    def save_aligned_poses(
        self,
        save_dataset=True,
        normalize_trans=True,
        tgt_dir=None,
        save_meta=True,
        global_mask_dir=None,
    ):
        if self.current_image != self.dataset.n_images:
            self.current_image -= 10
        poses = []
        img_names = []
        with torch.no_grad():
            for i in range(self.current_image):
                pose = self.pose_network(torch.tensor(i).cuda())
                pose4x4 = torch.eye(4)
                pose4x4[:3] = pose
                poses.append(pose4x4.cpu().numpy())
                img_names.append(self.dataset.index_to_frame[i])
        poses = np.stack(poses)
        Ks = self.dataset.intrinsics_all.cpu().numpy()
        if self.dataset.crop:
            transform_matrixs = []
            for i in range(len(poses)):
                transform_matrixs.append(
                    self.dataset.crop_transforms[self.dataset.index_to_frame[i]]
                )
            transform_matrixs = np.stack(transform_matrixs, axis=0)
        else:
            transform_matrixs = None
        mesh_path = os.path.join(
            self.base_exp_dir,
            "meshes",
            "{:0>8d}_{:0>8d}_{}_train.ply".format(
                self.current_image,
                self.iter_step - (self.iter_step % self.val_mesh_freq),
                64,
            ),
        )
        # ori_cam_path, img_num, mesh_path, tag, pred_poses, Ks, transform_matrixs, save_path
        case = self.case.split("_")[0]
        ml_camera_intrinsics = self.conf.get("dataset.ml_camera_intrinsics", default="")
        if ml_camera_intrinsics == "":
            ori_cam_path = f"./data/HO3Dv3/ann/{case}.npz"
        else:
            ori_cam_path = None
        if self.dataset.crop:
            align_poses(
                ori_cam_path,
                mesh_path,
                poses,
                Ks,
                transform_matrixs,
                self.base_exp_dir,
                img_names,
                self.iter_step,
                case,
                save_dataset=save_dataset,
                normalize_trans=normalize_trans,
                tgt_dir=tgt_dir,
                save_meta=save_meta,
                global_mask_dir=global_mask_dir,
            )
        else:
            align_poses_wo_virtual(
                ori_cam_path,
                mesh_path,
                poses,
                Ks,
                transform_matrixs,
                self.base_exp_dir,
                img_names,
                self.iter_step,
                case,
                save_dataset=save_dataset,
                normalize_trans=normalize_trans,
                tgt_dir=tgt_dir,
                save_meta=save_meta,
                global_mask_dir=global_mask_dir,
            )

    def save_checkpoint(self):
        checkpoint = {
            "nerf": self.nerf_outside.state_dict(),
            "sdf_network_fine": self.sdf_network.state_dict(),
            "variance_network_fine": self.deviation_network.state_dict(),
            "color_network_fine": self.color_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "iter_step": self.iter_step,
            "current_image": self.current_image,
        }
        if self.pose_type in ["gf", "seg"]:
            checkpoint["pose_network"] = self.pose_network.state_dict()

        if self.pose_type == "seg":
            checkpoint["current_pose_mlp_index"] = self.current_pose_mlp_index
            checkpoint["pro_iteration"] = self.pro_iteration

        if self.reset_based_on_rot:
            checkpoint["prev_pose"] = self.prev_pose

        os.makedirs(os.path.join(self.base_exp_dir, "checkpoints"), exist_ok=True)
        torch.save(
            checkpoint,
            os.path.join(
                self.base_exp_dir,
                "checkpoints",
                "ckpt_{:0>6d}_{:0>6d}.pth".format(self.current_image, self.iter_step),
            ),
        )

    def validate_image(self, idx=-1, resolution_level=-1, return_img=False):
        if idx < 0:
            idx = np.random.randint(self.current_image)

        print("Validate: iter: {}, camera: {}".format(self.iter_step, idx))
        pose_all = self.dataset.pose_all
        if self.conf["model.barf"]:
            pose_refine = camera.lie.se3_to_SE3(
                self.sdf_network.se3_refine.weight, only_rot=self.only_rotation
            )
            pose_all = camera.pose.compose(
                [pose_refine, self.sdf_network.noise_poses[:, :3, :]]
            )
        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        if self.pose_type in ["gf", "seg"]:
            with torch.no_grad():
                pose = self.pose_network(torch.tensor(idx).long())[:3]
        else:
            pose = pose_all[idx, :3]
        rays_o, rays_d = self.dataset.gen_rays_at(
            idx, resolution_level=resolution_level, pose=pose
        )
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        out_normal_fine = []

        print("length of rays_o: ", len(rays_o))
        for rays_o_batch, rays_d_batch in tqdm(zip(rays_o, rays_d)):
            # near, far, _ = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(
                rays_o_batch,
                rays_d_batch,
                near,
                far,
                cos_anneal_ratio=self.get_cos_anneal_ratio(),
                background_rgb=background_rgb,
            )

            def feasible(key):
                return (key in render_out) and (render_out[key] is not None)

            if feasible("color_fine"):
                out_rgb_fine.append(render_out["color_fine"].detach().cpu().numpy())
            if feasible("gradients") and feasible("weights"):
                n_samples = self.renderer.n_samples + self.renderer.n_importance
                normals = (
                    render_out["gradients"] * render_out["weights"][:, :n_samples, None]
                )
                if feasible("inside_sphere"):
                    normals = normals * render_out["inside_sphere"][..., None]
                normals = normals.sum(dim=1).detach().cpu().numpy()
                out_normal_fine.append(normals)
            del render_out

        img_fine = None
        if len(out_rgb_fine) > 0:
            img_fine = (
                np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256
            ).clip(0, 255)

        if return_img:
            return np.concatenate(
                [
                    img_fine[..., 0],
                    self.dataset.image_at(idx, resolution_level=resolution_level),
                ]
            )
        normal_img = None
        if len(out_normal_fine) > 0:
            normal_img = np.concatenate(out_normal_fine, axis=0)
            rot = np.linalg.inv(pose_all[idx, :3, :3].detach().cpu().numpy())
            normal_img = (
                np.matmul(rot[None, :, :], normal_img[:, :, None]).reshape(
                    [H, W, 3, -1]
                )
                * 128
                + 128
            ).clip(0, 255)

        os.makedirs(os.path.join(self.base_exp_dir, "validations_fine"), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, "normals"), exist_ok=True)

        for i in range(img_fine.shape[-1]):
            if len(out_rgb_fine) > 0:
                cv.imwrite(
                    os.path.join(
                        self.base_exp_dir,
                        "validations_fine",
                        "{:0>8d}_{:0>8d}_{}_{}.png".format(
                            self.current_image, self.iter_step, i, idx
                        ),
                    ),
                    np.concatenate(
                        [
                            img_fine[..., i],
                            self.dataset.image_at(
                                idx, resolution_level=resolution_level
                            ),
                        ]
                    ),
                )
            if len(out_normal_fine) > 0:
                cv.imwrite(
                    os.path.join(
                        self.base_exp_dir,
                        "normals",
                        "{:0>8d}_{:0>8d}_{}_{}.png".format(
                            self.current_image, self.iter_step, i, idx
                        ),
                    ),
                    normal_img[..., i],
                )

    def validate_all_images(self, resolution_level=-1):
        imgs = []
        # evenly sample 10 images from self.dataset.n_images
        if self.dataset.n_images < 10:
            idxs = np.arange(self.dataset.n_images)
        else:
            idxs = np.linspace(0, self.dataset.n_images - 1, 10, dtype=int)
        for i in tqdm(idxs):
            img_fine = self.validate_image(
                i, resolution_level=resolution_level, return_img=True
            )
            img_fine = cv.cvtColor(img_fine, cv.COLOR_BGR2RGB)
            imgs.append(img_fine)
        imageio.mimsave(os.path.join(self.base_exp_dir, "imgs.gif"), imgs, fps=2)

    def render_novel_image(self, idx_0, idx_1, ratio, resolution_level):
        """
        Interpolate view between two cameras.
        """
        pose_all = self.dataset.pose_all
        if self.conf["model.barf"]:
            pose_refine = camera.lie.se3_to_SE3(
                self.sdf_network.se3_refine.weight, only_rot=self.only_rotation
            )
            pose_all = camera.pose.compose(
                [pose_refine, self.sdf_network.noise_poses[:, :3, :]]
            )
        rays_o, rays_d = self.dataset.gen_rays_between(
            idx_0, idx_1, ratio, resolution_level=resolution_level, pose_all=pose_all
        )
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(
                rays_o_batch,
                rays_d_batch,
                near,
                far,
                cos_anneal_ratio=self.get_cos_anneal_ratio(),
                background_rgb=background_rgb,
            )

            out_rgb_fine.append(render_out["color_fine"].detach().cpu().numpy())

            del render_out

        img_fine = (
            (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 256)
            .clip(0, 255)
            .astype(np.uint8)
        )
        return img_fine

    def validate_mesh(
        self,
        world_space=False,
        resolution=64,
        threshold=0.0,
        use_norml_color=False,
        add_textured=False,
        mesh_scale=1.0,
    ):
        print("mesh_scale", mesh_scale)
        bound_min = (
            torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32) * mesh_scale
        )
        bound_max = (
            torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32) * mesh_scale
        )

        vertices, triangles = self.renderer.extract_geometry(
            bound_min, bound_max, resolution=resolution, threshold=threshold
        )
        os.makedirs(os.path.join(self.base_exp_dir, "meshes"), exist_ok=True)

        if world_space:
            vertices = (
                vertices * self.dataset.scale_mats_np[0][0, 0]
                + self.dataset.scale_mats_np[0][:3, 3][None]
            )

        if use_norml_color:
            vertices_tensor = torch.tensor(vertices, dtype=torch.float32).to(
                self.device
            )
            # split the vertices in batch
            vertices_tensor = vertices_tensor.split(self.batch_size)
            gradients = []
            for vertices_batch in tqdm(vertices_tensor):
                gradients.append(
                    self.renderer.sdf_network.gradient(vertices_batch)
                    .detach()
                    .cpu()
                    .numpy()
                )
            gradients = np.concatenate(gradients, axis=0)
            # normalize it
            gradients = gradients / np.linalg.norm(gradients, axis=-1, keepdims=True)
            gradients = (gradients + 1) / 2
            color = gradients.reshape(-1, 3)
        else:
            color = None

        mesh = trimesh.Trimesh(vertices, triangles, vertex_colors=color)
        step = self.iter_step - (self.iter_step % self.val_mesh_freq)
        mesh.export(
            os.path.join(
                self.base_exp_dir,
                "meshes",
                "{:0>8d}_{:0>8d}_{}_{}.ply".format(
                    self.current_image, step, resolution, self.mode
                ),
            )
        )
        print(
            "Mesh saved path",
            os.path.join(
                self.base_exp_dir,
                "meshes",
                "{:0>8d}_{:0>8d}_{}_{}.ply".format(
                    self.current_image, step, resolution, self.mode
                ),
            ),
        )
        if add_textured and not use_norml_color:
            textured_mesh(
                os.path.join(
                    self.base_exp_dir,
                    "meshes",
                    "{:0>8d}_{:0>8d}_{}_{}.ply".format(
                        self.current_image, step, resolution, self.mode
                    ),
                ),
                self,
            )
        logging.info("End")

    def interpolate_view(self, img_idx_0, img_idx_1):
        images = []
        n_frames = 60
        for i in range(n_frames):
            print(i)
            images.append(
                self.render_novel_image(
                    img_idx_0,
                    img_idx_1,
                    np.sin(((i / n_frames) - 0.5) * np.pi) * 0.5 + 0.5,
                    resolution_level=4,
                )
            )
        for i in range(n_frames):
            images.append(images[n_frames - i - 1])

        fourcc = cv.VideoWriter_fourcc(*"mp4v")
        video_dir = os.path.join(self.base_exp_dir, "render")
        os.makedirs(video_dir, exist_ok=True)
        h, w, _ = images[0].shape
        writer = cv.VideoWriter(
            os.path.join(
                video_dir,
                "{:0>8d}_{}_{}.mp4".format(self.iter_step, img_idx_0, img_idx_1),
            ),
            fourcc,
            30,
            (w, h),
        )

        for image in images:
            writer.write(image)

        writer.release()

    def save_alignment_materials(self, step=4, align_dir=None):
        case = self.case.split("_")[0]
        ori_cam_path = f"./data/HO3Dv3/ann/{case}.npz"
        camera_dict = np.load(ori_cam_path)
        img_ids = []
        for i in range(self.dataset.n_images):
            tag = f"scale_mat_{self.dataset.index_to_frame[i]}"
            if tag not in camera_dict:
                continue
            img_ids.append(i)
        world_pts_3D = []
        for i in img_ids[:: len(img_ids) // step]:
            print("frame", self.dataset.index_to_frame[i])
            pose = self.pose_network(torch.tensor(i).cuda())
            with torch.no_grad():
                rays_o, rays_d, ys, xs, p_norm = self.dataset.get_rays_based_on_mask(
                    i, pose
                )
                p_norm = p_norm.reshape(-1).cpu().numpy()
                depths = []
                rays_o = rays_o.reshape(-1, 3).split(self.batch_size, dim=0)
                rays_d = rays_d.reshape(-1, 3).split(self.batch_size, dim=0)
                for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
                    near, far = self.dataset.near_far_from_sphere(
                        rays_o_batch, rays_d_batch
                    )
                    render_out = self.renderer.render(
                        rays_o_batch, rays_d_batch, near, far
                    )
                    depth_fine = render_out["depth_fine"]
                    depths.append(depth_fine.cpu().numpy())
                depths = np.concatenate(depths, axis=0).squeeze(-1) / p_norm
                depth_map = np.zeros((self.dataset.H, self.dataset.W))
                depth_map[ys, xs] = depths
            # calculate the world coordinates
            K = self.dataset.intrinsics_all[i].cpu().numpy()[:3, :3]
            xy_hom = np.stack([xs, ys, np.ones_like(xs)], axis=0)
            cam_pts = (np.linalg.inv(K) @ xy_hom).T * depth_map[ys, xs][:, None]
            cam_pts = np.concatenate([cam_pts, np.ones((cam_pts.shape[0], 1))], axis=1)
            pose = pose.detach().cpu().numpy()  # camera to world pose
            world_pts = (pose @ cam_pts.T).T
            world_pts_3D.append(world_pts)
        world_pts_3D = np.concatenate(world_pts_3D, axis=0)
        if align_dir is not None:
            save_path = os.path.join(align_dir, f"{self.case}_world_pts_3D.npy")
        else:
            save_path = os.path.join(self.base_exp_dir, "world_pts_3D.npy")
        np.save(save_path, world_pts_3D)
        return save_path

    def render_poses(self, resolution_level=1, reduce_res=2, wo_normal=False):
        import pickle

        def load_pickle_data(f_name):
            """Loads the pickle data"""
            if not os.path.exists(f_name):
                raise Exception(
                    "Unable to find annotations picle file at %s. Aborting." % (f_name)
                )
            with open(f_name, "rb") as f:
                try:
                    pickle_data = pickle.load(f, encoding="latin1")
                except:
                    pickle_data = pickle.load(f)

            return pickle_data

        import tempfile

        if True:
            mesh_path = os.path.join(
                self.base_exp_dir,
                "meshes",
                "{:0>8d}_{}_{}.ply".format(self.iter_step, 512, "validate_mesh"),
            )
            if not os.path.exists(mesh_path):
                mesh_path = os.path.join(
                    self.base_exp_dir,
                    "meshes",
                    "{:0>8d}_{:0>8d}_{}_{}.ply".format(
                        self.current_image, self.iter_step, 64, "train"
                    ),
                )
            mesh = trimesh.load_mesh(mesh_path)
            vertices = mesh.bounding_box_oriented.vertices
        else:
            raise NotImplementedError

        def find_closest_points(points, target_point):
            # we first find the closest point
            min_dist = 1e10
            min_idx = -1
            self_idx = -1
            for i in range(len(points)):
                dist = np.linalg.norm(target_point - points[i])
                if dist == 0:
                    self_idx = i
                if dist < min_dist and dist > 0:
                    min_dist = dist
                    min_idx = i
            # we form a line
            vec = points[min_idx] - target_point
            candidates = {}
            for i in range(len(points)):
                if i != min_idx and i != self_idx:
                    vec2 = points[i] - target_point
                    # if angle is 90
                    if np.abs(np.dot(vec, vec2)) < 1e-5:
                        candidates[i] = np.linalg.norm(vec2)
            # find smallest 2 points
            closest_indices = list(sorted(candidates, key=candidates.get)[:2])
            return closest_indices + [min_idx]

        def reconstruct_edges(points):
            edges = set()
            for i, point in enumerate(points):
                closest_indices = find_closest_points(points, point)
                for idx in closest_indices:
                    edges.add(tuple(sorted([i, idx])))  # 排序以确保唯一性
            return edges

        edges = reconstruct_edges(vertices)
        imgs = []
        pred_poses = []

        step_map = {}
        step = 0
        normal_dir = os.path.join(self.base_exp_dir, "normal_vis")
        os.makedirs(normal_dir, exist_ok=True)
        normal_lists = []
        centers = []
        max_side = 0
        for i in tqdm(range(self.dataset.n_images)):  # self.dataset.avai_ann_frame:
            with torch.no_grad():
                img_id = torch.tensor(i).cuda()
                pose = self.pose_network(img_id)
                # we want to add normal picture
                if not wo_normal:
                    rays_o, rays_d, ys, xs, _ = self.dataset.get_rays_based_on_mask(
                        img_id, pose[:3]
                    )
                    max_side = max(max_side, ys.max() - ys.min(), xs.max() - xs.min())
                    centers.append(
                        [(xs.max() + xs.min()) // 2, (ys.max() + ys.min()) // 2]
                    )
                    out_normal_fine = []
                    out_mask_fine = []
                    rays_o = rays_o.reshape(-1, 3).split(self.batch_size, dim=0)
                    rays_d = rays_d.reshape(-1, 3).split(self.batch_size, dim=0)
                    for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
                        near, far = self.dataset.near_far_from_sphere(
                            rays_o_batch, rays_d_batch
                        )
                        render_out = self.renderer.render(
                            rays_o_batch, rays_d_batch, near, far
                        )
                        n_samples = self.renderer.n_samples + self.renderer.n_importance
                        normals = (
                            render_out["gradients"]
                            * render_out["weights"][:, :n_samples, None]
                        )
                        normals = normals.sum(dim=1).detach().cpu().numpy()
                        pred_mask = (
                            render_out["weight_sum"].detach().cpu().numpy()[..., 0]
                            > 0.5
                        )
                        out_mask_fine.append(pred_mask)
                        out_normal_fine.append(normals)
                    masks = np.concatenate(out_mask_fine, axis=0)
                    normals = np.concatenate(out_normal_fine, axis=0)
                    normals = normals[masks]
                    xs, ys = xs[masks], ys[masks]
                    rot = np.linalg.inv(pose[:3, :3].detach().cpu().numpy())
                    normals = (rot @ normals.T).T
                    normal_img = np.ones((self.dataset.H, self.dataset.W, 3))
                    normal_img[ys, xs] = normals
                    normal_img = (normal_img * 128 + 128).clip(0, 255)
                    normal_img = normal_img.astype(np.uint8)
                    cv.imwrite(
                        os.path.join(
                            normal_dir, f"{self.dataset.index_to_frame[i]}.jpg"
                        ),
                        normal_img,
                    )
                    normal_lists.append(normal_img)
            pose4x4 = np.eye(4)
            pose4x4[:3] = pose[:3].cpu().numpy()
            pred_poses.append(pose4x4)
            step_map[i] = step
            step += 1
        pred_poses = torch.from_numpy(np.array(pred_poses)).cpu()
        # pred_poses = align_ate_c2b_use_a2b(pred_poses, gt_poses)
        pred_poses = pred_poses.numpy()
        # gt_poses = gt_poses.numpy()
        # ate = compute_ATE(gt_poses, pred_poses)
        # rpe_trans, rpe_rot = compute_rpe(gt_poses, pred_poses)
        # print(ate, rpe_trans, rpe_rot)
        # print(step_map.keys())
        # center crop the normals
        max_side = max_side // 2
        crop_normals = []
        for i in range(len(normal_lists)):
            center = centers[i]
            normal_img = normal_lists[i]
            crop_normal = normal_img[
                center[1] - max_side : center[1] + max_side,
                center[0] - max_side : center[0] + max_side,
            ]
            crop_normals.append(crop_normal)
        pose_dir = os.path.join(self.base_exp_dir, "pose_vis")
        os.makedirs(pose_dir, exist_ok=True)
        for i in range(self.dataset.n_images):  # [idx1, idx2],
            img = self.dataset.image_at(i, resolution_level)
            step = step_map[i]
            # change to rgb
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            for pose, color in [(pred_poses[step], (0, 255, 0))]:  # ,
                obj_pose = np.linalg.inv(pose)
                R, t = obj_pose[:3, :3], obj_pose[:3, 3]
                R, _ = cv.Rodrigues(R)
                R = R.astype(np.float32)
                t = t.astype(np.float32)
                K = (
                    self.dataset.intrinsics_all[i]
                    .cpu()
                    .numpy()[:3, :3]
                    .astype(np.float32)
                )
                # project the points to image plane
                img_points, _ = cv.projectPoints(vertices, R, t, K, None)
                img_points = img_points / resolution_level

                for edge in edges:
                    p1, p2 = img_points[edge[0]], img_points[edge[1]]
                    p1 = p1.astype(int)
                    p2 = p2.astype(int)
                    cv.line(img, tuple(p1[0]), tuple(p2[0]), color, 3)
            # # save img
            cv.imwrite(
                os.path.join(pose_dir, f"{self.dataset.index_to_frame[i]}.jpg"),
                cv.cvtColor(img, cv.COLOR_BGR2RGB),
            )

            if wo_normal:
                step += 1
                continue
            # continue
            with tempfile.TemporaryDirectory() as temp_dir:
                img_path = os.path.join(temp_dir, f"{i}.jpg")
                vis_simple_traj(
                    pred_poses[: step + 1],
                    None,
                    img_path,
                    no_gt=True,
                    H=self.dataset.H,
                    W=self.dataset.W,
                )
                # vis_simple_traj(pred_poses[:step + 1], gt_poses[:step + 1], img_path)
                pose_traj = cv.imread(img_path, cv.IMREAD_UNCHANGED)
                # change it to rgb
                pose_traj = cv.cvtColor(pose_traj, cv.COLOR_BGR2RGB)
                normal_img = cv.resize(
                    crop_normals[step], (self.dataset.H, self.dataset.H)
                )
                img = np.concatenate([img, normal_img, pose_traj], axis=1)
                # reduce the size by 4
                img = cv.resize(
                    img, (img.shape[1] // reduce_res, img.shape[0] // reduce_res)
                )
            imgs.append(img)
            step += 1
        if not wo_normal:
            imageio.mimsave(
                os.path.join(self.base_exp_dir, f"poses_{self.iter_step}.gif"),
                imgs,
                fps=5,
            )
            imageio.mimsave(
                os.path.join(self.base_exp_dir, f"poses_{self.iter_step}.mp4"),
                imgs,
                fps=5,
            )
        pass


if __name__ == "__main__":
    print("Hello FMOV")

    torch.set_default_tensor_type("torch.cuda.FloatTensor")

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=str, default="./confs/base.conf")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--mcube_threshold", type=float, default=0.0)
    parser.add_argument("--is_continue", default=False, action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--case", type=str, default="")
    parser.add_argument("--dataset", type=str, default="DTU")
    parser.add_argument("--start_at", type=int, default=-1)
    parser.add_argument("--start_img_idx", type=int, default=0)
    parser.add_argument("--ori_cam_path", type=str, default="None")
    parser.add_argument("--gradient_analysis", default=False, action="store_true")
    parser.add_argument("--global_conf", type=str, default="None")
    parser.add_argument("--flow_interval", type=int, default=-1)
    parser.add_argument("--reset_rot_degree", type=int, default=-1)
    parser.add_argument("--image_interval", type=int, default=-1)
    parser.add_argument("--mesh_scale", type=float, default=1.0)
    parser.add_argument("--align_dir", type=str, default=None)

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    runner = Runner(
        args.conf,
        args.mode,
        args.case,
        args.dataset,
        args.is_continue,
        args.start_at,
        args.start_img_idx,
        args.gradient_analysis,
        has_global_conf=args.global_conf != "None" or "GT.conf" in args.conf,
        flow_interval=args.flow_interval,
        reset_rot_degree=args.reset_rot_degree,
        image_interval=args.image_interval,
    )
    print("mode: ", args.mode)
    if args.mode == "train":
        # after training we need reboot the system for global training
        if args.global_conf != "None":
            case = runner.case.split("_")[0]
            if "ho3d" in args.global_conf:
                global_mask_dir = f"./data/HO3Dv3/{case}/mask_obj"
            elif "ml" in args.global_conf:
                global_mask_dir = f"./data/ML/{case}/mask_obj"
            else:
                raise NotImplementedError
            original_exp_dir = runner.base_exp_dir
            conf_name = args.global_conf.split("/")[-1].split(".")[0]
            new_exp_dir = os.path.join(original_exp_dir, conf_name)
            if not os.path.exists(new_exp_dir):
                try:
                    runner.train()
                except Exception as e:
                    with open(
                        os.path.join(
                            original_exp_dir, "error_during_progressive_learning.txt"
                        ),
                        "w",
                    ) as f:
                        f.write("Exception occurred: " + str(e) + "\n")
                        f.write(traceback.format_exc())
                runner.save_aligned_poses(
                    save_dataset=True,
                    normalize_trans=True,
                    tgt_dir=os.path.join(original_exp_dir, conf_name),
                    save_meta=False,
                    global_mask_dir=global_mask_dir,
                )
            runner = Runner(
                args.global_conf,
                mode="train",
                case=case,
                dataset=args.dataset,
                is_continue=os.path.exists(os.path.join(new_exp_dir, "checkpoints")),
                start_at=args.start_at,
                start_img_idx=args.start_img_idx,
                gradient_analysis=args.gradient_analysis,
                exp_dir=os.path.join(original_exp_dir, conf_name),
                has_global_conf=os.path.exists(new_exp_dir),
            )
            print(
                "reboot the system for global training--------------------------------"
            )
            runner.train()
            # final evaluation...
            runner.render_poses()
            runner.validate_mesh(resolution=512, use_norml_color=True)
            runner.save_poses_simple()
        else:
            # we don't need to reboot the system
            runner.train()
            runner.render_poses()
            runner.validate_mesh(resolution=512, use_norml_color=True)
    elif args.mode == "validate_mesh":
        print("validate_mesh.....")
        if args.global_conf == "None":
            runner.validate_mesh(
                resolution=512, use_norml_color=True, mesh_scale=args.mesh_scale
            )
        else:
            case = runner.case.split("_")[0]
            if "ho3d" in args.global_conf:
                global_mask_dir = f"./data/HO3Dv3/{case}/mask_obj"
            elif "ml" in args.global_conf:
                global_mask_dir = f"./data/ML/{case}/mask_obj"
            else:
                raise NotImplementedError
            original_exp_dir = runner.base_exp_dir
            conf_name = args.global_conf.split("/")[-1].split(".")[0]
            new_exp_dir = os.path.join(original_exp_dir, conf_name)
            runner = Runner(
                args.global_conf,
                mode="train",
                case=case,
                dataset=args.dataset,
                is_continue=os.path.exists(os.path.join(new_exp_dir, "checkpoints")),
                start_at=args.start_at,
                start_img_idx=args.start_img_idx,
                gradient_analysis=args.gradient_analysis,
                exp_dir=os.path.join(original_exp_dir, conf_name),
                has_global_conf=os.path.exists(new_exp_dir),
            )
            runner.validate_mesh(
                resolution=256, use_norml_color=True, mesh_scale=args.mesh_scale
            )
    elif args.mode == "validate_poses":
        runner.validate_poses()  # save_pose=True
    elif args.mode.startswith(
        "interpolate"
    ):  # Interpolate views given two image indices
        _, img_idx_0, img_idx_1 = args.mode.split("_")
        img_idx_0 = int(img_idx_0)
        img_idx_1 = int(img_idx_1)
        runner.interpolate_view(img_idx_0, img_idx_1)
    elif args.mode == "validate_all_images":
        runner.validate_all_images(resolution_level=4)
    elif args.mode == "save_poses":
        runner.save_poses()
    elif args.mode == "save_aligned_poses":
        runner.save_aligned_poses()
    elif args.mode == "render_poses":
        case = runner.case.split("_")[0]
        if "ho3d" in args.global_conf:
            global_mask_dir = f"./data/HO3Dv3/{case}/mask_obj"
        elif "ml" in args.global_conf:
            global_mask_dir = f"./data/ML/{case}/mask_obj"
        else:
            raise NotImplementedError
        original_exp_dir = runner.base_exp_dir
        conf_name = args.global_conf.split("/")[-1].split(".")[0]
        new_exp_dir = os.path.join(original_exp_dir, conf_name)
        runner = Runner(
            args.global_conf,
            mode="train",
            case=case,
            dataset=args.dataset,
            is_continue=os.path.exists(os.path.join(new_exp_dir, "checkpoints")),
            start_at=args.start_at,
            start_img_idx=args.start_img_idx,
            gradient_analysis=args.gradient_analysis,
            exp_dir=os.path.join(original_exp_dir, conf_name),
            has_global_conf=os.path.exists(new_exp_dir),
        )
        runner.render_poses()
    elif args.mode == "pure_render_poses":
        runner.render_poses(wo_normal=True)
    elif args.mode == "generate_textured_mesh":
        case = runner.case.split("_")[0]
        original_exp_dir = runner.base_exp_dir
        conf_name = args.global_conf.split("/")[-1].split(".")[0]
        new_exp_dir = os.path.join(original_exp_dir, conf_name)
        runner = Runner(
            args.global_conf,
            mode="validate",
            case=case,
            dataset=args.dataset,
            is_continue=os.path.exists(os.path.join(new_exp_dir, "checkpoints")),
            start_at=args.start_at,
            start_img_idx=args.start_img_idx,
            gradient_analysis=args.gradient_analysis,
            exp_dir=os.path.join(original_exp_dir, conf_name),
            has_global_conf=os.path.exists(new_exp_dir),
        )
        runner.validate_mesh(resolution=64, add_textured=True)
    elif args.mode == "save_poses_simple":
        if args.global_conf == "None":
            runner.save_poses_simple(align_dir=args.align_dir)
        else:
            case = runner.case.split("_")[0]
            original_exp_dir = runner.base_exp_dir
            conf_name = args.global_conf.split("/")[-1].split(".")[0]
            new_exp_dir = os.path.join(original_exp_dir, conf_name)
            runner = Runner(
                args.global_conf,
                mode="validate",
                case=case,
                dataset=args.dataset,
                is_continue=os.path.exists(os.path.join(new_exp_dir, "checkpoints")),
                start_at=args.start_at,
                start_img_idx=args.start_img_idx,
                gradient_analysis=args.gradient_analysis,
                exp_dir=os.path.join(original_exp_dir, conf_name),
                has_global_conf=os.path.exists(new_exp_dir),
            )
            runner.save_poses_simple()
    elif args.mode == "save_alignment_materials":
        if args.global_conf == "None":
            runner.save_alignment_materials(align_dir=args.align_dir)
        else:
            case = runner.case.split("_")[0]
            original_exp_dir = runner.base_exp_dir
            conf_name = args.global_conf.split("/")[-1].split(".")[0]
            new_exp_dir = os.path.join(original_exp_dir, conf_name)
            runner = Runner(
                args.global_conf,
                mode="validate",
                case=case,
                dataset=args.dataset,
                is_continue=os.path.exists(os.path.join(new_exp_dir, "checkpoints")),
                start_at=args.start_at,
                start_img_idx=args.start_img_idx,
                gradient_analysis=args.gradient_analysis,
                exp_dir=os.path.join(original_exp_dir, conf_name),
                has_global_conf=os.path.exists(new_exp_dir),
            )
            runner.save_alignment_materials()
    elif args.mode == "validate_textured_mesh":
        print("validate_mesh.....")
        if args.global_conf == "None":
            runner.validate_mesh(resolution=64, add_textured=True)
        else:
            case = runner.case.split("_")[0]
            if "ho3d" in args.global_conf:
                global_mask_dir = f"./data/HO3Dv3/{case}/mask_obj"
            elif "ml" in args.global_conf:
                global_mask_dir = f"./data/ML/{case}/mask_obj"
            else:
                raise NotImplementedError
            original_exp_dir = runner.base_exp_dir
            conf_name = args.global_conf.split("/")[-1].split(".")[0]
            new_exp_dir = os.path.join(original_exp_dir, conf_name)
            runner = Runner(
                args.global_conf,
                mode="train",
                case=case,
                dataset=args.dataset,
                is_continue=os.path.exists(os.path.join(new_exp_dir, "checkpoints")),
                start_at=args.start_at,
                start_img_idx=args.start_img_idx,
                gradient_analysis=args.gradient_analysis,
                exp_dir=os.path.join(original_exp_dir, conf_name),
                has_global_conf=os.path.exists(new_exp_dir),
            )
            runner.validate_mesh(resolution=64, add_textured=True)
    else:
        raise NotImplementedError
