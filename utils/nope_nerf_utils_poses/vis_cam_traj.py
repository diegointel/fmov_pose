# This file is modified from NeRF++: https://github.com/Kai-46/nerfplusplus

import numpy as np

try:
    import open3d as o3d
except ImportError:
    pass
from utils.draw_plotly import draw_plotly

import matplotlib.pyplot as plt
import torch


def frustums2lineset(frustums):
    N = len(frustums)
    merged_points = np.zeros((N * 5, 3))  # 5 vertices per frustum
    merged_lines = np.zeros((N * 8, 2))  # 8 lines per frustum
    merged_colors = np.zeros((N * 8, 3))  # each line gets a color

    for i, (frustum_points, frustum_lines, frustum_colors) in enumerate(frustums):
        merged_points[i * 5 : (i + 1) * 5, :] = frustum_points
        merged_lines[i * 8 : (i + 1) * 8, :] = frustum_lines + i * 5
        merged_colors[i * 8 : (i + 1) * 8, :] = frustum_colors

    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(merged_points)
    lineset.lines = o3d.utility.Vector2iVector(merged_lines)
    lineset.colors = o3d.utility.Vector3dVector(merged_colors)

    return lineset


def get_camera_frustum_opengl_coord(
    H, W, fx, fy, W2C, frustum_length=0.5, color=np.array([0.0, 1.0, 0.0])
):
    """X right, Y up, Z backward to the observer.
    :param H, W:
    :param fx, fy:
    :param W2C:             (4, 4)  matrix
    :param frustum_length:  scalar: scale the frustum
    :param color:           (3,)    list, frustum line color
    :return:
        frustum_points:     (5, 3)  frustum points in world coordinate
        frustum_lines:      (8, 2)  8 lines connect 5 frustum points, specified in line start/end index.
        frustum_colors:     (8, 3)  colors for 8 lines.
    """
    hfov = np.rad2deg(np.arctan(W / 2.0 / fx) * 2.0)
    vfov = np.rad2deg(np.arctan(H / 2.0 / fy) * 2.0)
    half_w = frustum_length * np.tan(np.deg2rad(hfov / 2.0))
    half_h = frustum_length * np.tan(np.deg2rad(vfov / 2.0))

    # build view frustum in camera space in homogenous coordinate (5, 4)
    frustum_points = np.array(
        [
            [0.0, 0.0, 0.0, 1.0],  # frustum origin
            [-half_w, half_h, -frustum_length, 1.0],  # top-left image corner
            [half_w, half_h, -frustum_length, 1.0],  # top-right image corner
            [half_w, -half_h, -frustum_length, 1.0],  # bottom-right image corner
            [-half_w, -half_h, -frustum_length, 1.0],
        ]
    )  # bottom-left image corner
    frustum_lines = np.array(
        [[0, i] for i in range(1, 5)] + [[i, (i + 1)] for i in range(1, 4)] + [[4, 1]]
    )  # (8, 2)
    frustum_colors = np.tile(
        color.reshape((1, 3)), (frustum_lines.shape[0], 1)
    )  # (8, 3)

    # transform view frustum from camera space to world space
    C2W = np.linalg.inv(W2C)
    frustum_points = np.matmul(C2W, frustum_points.T).T  # (5, 4)
    frustum_points = (
        frustum_points[:, :3] / frustum_points[:, 3:4]
    )  # (5, 3)  remove homogenous coordinate
    return frustum_points, frustum_lines, frustum_colors


def get_camera_frustum_opencv_coord(
    H, W, fx, fy, W2C, frustum_length=0.5, color=np.array([0.0, 1.0, 0.0])
):
    """X right, Y up, Z backward to the observer.
    :param H, W:
    :param fx, fy:
    :param W2C:             (4, 4)  matrix
    :param frustum_length:  scalar: scale the frustum
    :param color:           (3,)    list, frustum line color
    :return:
        frustum_points:     (5, 3)  frustum points in world coordinate
        frustum_lines:      (8, 2)  8 lines connect 5 frustum points, specified in line start/end index.
        frustum_colors:     (8, 3)  colors for 8 lines.
    """
    hfov = np.rad2deg(np.arctan(W / 2.0 / fx) * 2.0)
    vfov = np.rad2deg(np.arctan(H / 2.0 / fy) * 2.0)
    half_w = frustum_length * np.tan(np.deg2rad(hfov / 2.0))
    half_h = frustum_length * np.tan(np.deg2rad(vfov / 2.0))

    # build view frustum in camera space in homogenous coordinate (5, 4)
    frustum_points = np.array(
        [
            [0.0, 0.0, 0.0, 1.0],  # frustum origin
            [-half_w, -half_h, frustum_length, 1.0],  # top-left image corner
            [half_w, -half_h, frustum_length, 1.0],  # top-right image corner
            [half_w, half_h, frustum_length, 1.0],  # bottom-right image corner
            [-half_w, +half_h, frustum_length, 1.0],
        ]
    )  # bottom-left image corner
    frustum_lines = np.array(
        [[0, i] for i in range(1, 5)] + [[i, (i + 1)] for i in range(1, 4)] + [[4, 1]]
    )  # (8, 2)
    frustum_colors = np.tile(
        color.reshape((1, 3)), (frustum_lines.shape[0], 1)
    )  # (8, 3)

    # transform view frustum from camera space to world space
    C2W = np.linalg.inv(W2C)
    frustum_points = np.matmul(C2W, frustum_points.T).T  # (5, 4)
    frustum_points = (
        frustum_points[:, :3] / frustum_points[:, 3:4]
    )  # (5, 3)  remove homogenous coordinate
    return frustum_points, frustum_lines, frustum_colors


def draw_camera_frustum_geometry(
    c2ws,
    H,
    W,
    fx=600.0,
    fy=600.0,
    frustum_length=0.5,
    color=np.array([29.0, 53.0, 87.0]) / 255.0,
    draw_now=False,
    coord="opengl",
):
    """
    :param c2ws:            (N, 4, 4)  np.array
    :param H:               scalar
    :param W:               scalar
    :param fx:              scalar
    :param fy:              scalar
    :param frustum_length:  scalar
    :param color:           None or (N, 3) or (3, ) or (1, 3) or (3, 1) np array
    :param draw_now:        True/False call o3d vis now
    :return:
    """
    N = c2ws.shape[0]

    num_ele = color.flatten().shape[0]
    if num_ele == 3:
        color = color.reshape(1, 3)
        color = np.tile(color, (N, 1))

    frustum_list = []
    if coord == "opengl":
        for i in range(N):
            frustum_list.append(
                get_camera_frustum_opengl_coord(
                    H,
                    W,
                    fx,
                    fy,
                    W2C=np.linalg.inv(c2ws[i]),
                    frustum_length=frustum_length,
                    color=color[i],
                )
            )
    elif coord == "opencv":
        for i in range(N):
            frustum_list.append(
                get_camera_frustum_opencv_coord(
                    H,
                    W,
                    fx,
                    fy,
                    W2C=np.linalg.inv(c2ws[i]),
                    frustum_length=frustum_length,
                    color=color[i],
                )
            )
    else:
        print("Undefined coordinate system. Exit")
        exit()

    frustums_geometry = frustums2lineset(frustum_list)

    if draw_now:
        o3d.visualization.draw_geometries([frustums_geometry])

    return frustums_geometry  # this is an o3d geometry object.


frustum_length = 0.1
est_traj_color = np.array([39, 125, 161], dtype=np.float32) / 255
cmp_traj_color = np.array([249, 65, 68], dtype=np.float32) / 255


def vis_poses(pred_poses, gt_poses, H, W, fx, fy, save_path):
    import open3d as o3d

    frustum_est_list = draw_camera_frustum_geometry(
        pred_poses.cpu().numpy(),
        H,
        W,
        fx,
        fy,
        frustum_length,
        est_traj_color,
        coord="opencv",
    )
    frustum_colmap_list = draw_camera_frustum_geometry(
        gt_poses.cpu().numpy(),
        H,
        W,
        fx,
        fy,
        frustum_length,
        cmp_traj_color,
        coord="opencv",
    )

    geometry_to_draw = []
    geometry_to_draw.append(frustum_est_list)
    geometry_to_draw.append(frustum_colmap_list)

    """o3d for line drawing"""
    t_est_list = pred_poses[:, :3, 3]
    t_cmp_list = gt_poses[:, :3, 3]

    """line set to note pose correspondence between two trajs"""
    t_est_list = t_est_list.cpu()
    line_points = torch.cat([t_est_list, t_cmp_list], dim=0).cpu().numpy()  # (2N, 3)
    N_imgs = t_est_list.shape[0]
    line_ends = [
        [i, i + N_imgs] for i in range(N_imgs)
    ]  # (N, 2) connect two end points.

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(line_points)
    line_set.lines = o3d.utility.Vector2iVector(line_ends)
    unit_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=2)
    unit_sphere = o3d.geometry.LineSet.create_from_triangle_mesh(unit_sphere)
    unit_sphere.paint_uniform_color((0, 1, 0))

    geometry_to_draw.append(line_set)
    draw_plotly(geometry_to_draw, save_path=save_path)


def vis_single_pose(pred_poses, H, W, fx, fy, save_path):
    frustum_est_list = draw_camera_frustum_geometry(
        pred_poses.cpu().numpy(),
        H,
        W,
        fx,
        fy,
        frustum_length,
        est_traj_color,
        coord="opencv",
    )
    geometry_to_draw = []
    geometry_to_draw.append(frustum_est_list)
    draw_plotly(geometry_to_draw, save_path=save_path)


# we set limits based on GT poses so it is comparable :)
def vis_simple_traj(
    pred_poses, gt_poses, save_path, no_gt=False, H=400, W=400, tag1="GT", tag2="Ours"
):
    # plot two camera trajectory
    # Assume we have two camera trajectories
    if not no_gt:
        camera1_trajectory = gt_poses[:, :3, 3]
    camera2_trajectory = pred_poses[:, :3, 3]
    if H is not None and W is not None:
        fig = plt.figure(figsize=(W / 100, H / 100))
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    else:
        fig = plt.figure()

    # Create 3D subplot
    ax = fig.add_subplot(111, projection="3d")
    ax.grid(False)
    # Unpack the coordinates
    if not no_gt:
        x1, y1, z1 = zip(*camera1_trajectory)
    x2, y2, z2 = zip(*camera2_trajectory)
    # Plot the trajectories
    # ax.autoscale()
    if not no_gt:
        ax.plot(
            x1,
            y1,
            z1,
            label=tag1,
            linestyle=(0, (3, 1)),
            color="b",
            linewidth=2,
            alpha=1,
        )
    ax.plot(
        x2, y2, z2, label=tag2, color="g", linestyle=(0, (1, 1)), linewidth=2, alpha=1
    )
    # ax.view_init(elev=20., azim=50.)
    if no_gt:
        x1 = x2
        y1 = y2
        z1 = z2

    # set limit based on GT poses
    min_x = np.min(x1)
    if min_x > 0:
        min_x = min_x * 0.8
    else:
        min_x = min_x * 1.2
    max_x = np.max(x1)
    min_y = np.min(y1)
    if min_y > 0:
        min_y = min_y * 0.8
    else:
        min_y = min_y * 1.2
    max_y = np.max(y1)
    min_z = np.min(z1)
    if min_z > 0:
        min_z = min_z * 0.8
    else:
        min_z = min_z * 1.2
    max_z = np.max(z1)
    # set limit
    ax.set_xlim([min_x, max_x])
    ax.set_ylim([min_y, max_y])
    ax.set_zlim([min_z, max_z])

    # labels
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.tick_params(axis='both', which='major', labelsize=15)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    plt.legend(prop={"size": 15}, loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path)
    # plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    # plt.savefig(save_path, pad_inches=0)
