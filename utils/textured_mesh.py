import xatlas
import torch
import numpy as np
from tqdm import tqdm
import trimesh
import cv2


# refering to https://github.com/autonomousvision/sdfstudio/blob/370902a10dbef08cb3fe4391bd3ed1e227b5c165/nerfstudio/exporter/texture_utils.py#L210
# we are at torch 1.9.1 https://pytorch.org/docs/1.9.1/generated/torch.meshgrid.html?highlight=torch%20mesh
def get_texture_image(num_pixels_w, num_pixels_h, device):
    """Get a texture image."""
    px_w = 1.0 / num_pixels_w
    px_h = 1.0 / num_pixels_h
    us, vs = torch.meshgrid(
        torch.arange(num_pixels_w, device=device),
        torch.arange(num_pixels_h, device=device),
    )
    uv_indices = torch.stack([us.T, vs.T], dim=-1)
    linspace_h = torch.linspace(px_h / 2, 1 - px_h / 2, num_pixels_h, device=device)
    linspace_w = torch.linspace(px_w / 2, 1 - px_w / 2, num_pixels_w, device=device)
    us, vs = torch.meshgrid(linspace_w, linspace_h)
    uv_coords = torch.stack([us.T, vs.T], dim=-1)  # (num_pixels_h, num_pixels_w, 2)
    return uv_coords, uv_indices


def get_parallelogram_area(p, v0, v1):
    """Given three 2D points, return the area defined by the parallelogram. I.e., 2x the triangle area.

    Args:
        p: The origin of the parallelogram.
        v0: The first vector of the parallelogram.
        v1: The second vector of the parallelogram.

    Returns:
        The area of the parallelogram.
    """
    return (p[..., 0] - v0[..., 0]) * (v1[..., 1] - v0[..., 1]) - (
        p[..., 1] - v0[..., 1]
    ) * (v1[..., 0] - v0[..., 0])


def unwrap_mesh_with_xatlas(
    vertices,
    faces,
    vertex_normals,
    num_pixels_per_side=1024,
    num_faces_per_barycentric_chunk=10,
):
    """Unwrap a mesh using xatlas. We use xatlas to unwrap the mesh with UV coordinates.
    Then we rasterize the mesh with a square pattern. We interpolate the XYZ and normal
    values for every pixel in the texture image. We return the texture coordinates, the
    origins, and the directions for every pixel.

    Args:
        vertices: Tensor of mesh vertices.
        faces: Tensor of mesh faces.
        vertex_normals: Tensor of mesh vertex normals.
        num_pixels_per_side: Number of pixels per side of the texture image. We use a square.
        num_faces_per_barycentric_chunk: Number of faces to use for barycentric chunk computation.

    Returns:
        texture_coordinates: Tensor of texture coordinates for every face.
        origins: Tensor of origins for every pixel.
        directions: Tensor of directions for every pixel.
    """

    device = vertices.device

    # unwrap the mesh
    vertices_np = vertices.cpu().numpy()
    faces_np = faces.cpu().numpy()
    vertex_normals_np = vertex_normals.cpu().cpu().numpy()
    vmapping, indices, uvs = xatlas.parametrize(  # pylint: disable=c-extension-no-member
        vertices_np, faces_np, vertex_normals_np
    )

    # vertices texture coordinates
    vertices_tc = torch.from_numpy(uvs.astype(np.float32)).to(device)

    # render uv maps
    vertices_tc = vertices_tc * 2.0 - 1.0  # uvs to range [-1, 1]
    vertices_tc = torch.cat(
        (
            vertices_tc,
            torch.zeros_like(vertices_tc[..., :1]),
            torch.ones_like(vertices_tc[..., :1]),
        ),
        dim=-1,
    )  # [num_verts, 4]

    texture_coordinates = torch.from_numpy(uvs[indices]).to(device)  # (num_faces, 3, 2)

    # Now find the triangle indices for every pixel and the barycentric coordinates
    # which can be used to interpolate the XYZ and normal values to then query with NeRF
    uv_coords, _ = get_texture_image(num_pixels_per_side, num_pixels_per_side, device)
    uv_coords_shape = uv_coords.shape
    p = uv_coords.reshape(1, -1, 2)  # (1, N, 2)
    num_vertices = p.shape[1]
    num_faces = texture_coordinates.shape[0]
    triangle_distances = (
        torch.ones_like(p[..., 0]) * torch.finfo(torch.float32).max
    )  # (1, N)
    triangle_indices = torch.zeros_like(p[..., 0]).long()  # (1, N)
    triangle_w0 = torch.zeros_like(p[..., 0])  # (1, N)
    triangle_w1 = torch.zeros_like(p[..., 0])  # (1, N)
    triangle_w2 = torch.zeros_like(p[..., 0])  # (1, N)
    arange_list = torch.arange(num_vertices, device=device)

    for i in tqdm(range(num_faces // num_faces_per_barycentric_chunk)):
        s = i * num_faces_per_barycentric_chunk
        e = min((i + 1) * num_faces_per_barycentric_chunk, num_faces)
        v0 = texture_coordinates[s:e, 0:1, :]  # (F, 1, 2)
        v1 = texture_coordinates[s:e, 1:2, :]  # (F, 1, 2)
        v2 = texture_coordinates[s:e, 2:3, :]  # (F, 1, 2)
        # NOTE: could try clockwise vs counter clockwise
        area = get_parallelogram_area(v2, v0, v1)  # 2x face area.
        w0 = (
            get_parallelogram_area(p, v1, v2) / area
        )  # (num_faces_per_barycentric_chunk, N)
        w1 = get_parallelogram_area(p, v2, v0) / area
        w2 = get_parallelogram_area(p, v0, v1) / area
        # get distance from center of triangle
        dist_to_center = torch.abs(w0) + torch.abs(w1) + torch.abs(w2)
        d_values, d_indices = torch.min(dist_to_center, dim=0, keepdim=True)
        d_indices_with_offset = d_indices + s  # add offset
        condition = d_values < triangle_distances
        triangle_distances = torch.where(condition, d_values, triangle_distances)
        triangle_indices = torch.where(
            condition, d_indices_with_offset, triangle_indices
        )
        w0_selected = w0[d_indices[0], arange_list].unsqueeze(0)  # (1, N)
        w1_selected = w1[d_indices[0], arange_list].unsqueeze(0)  # (1, N)
        w2_selected = w2[d_indices[0], arange_list].unsqueeze(0)  # (1, N)
        triangle_w0 = torch.where(condition, w0_selected, triangle_w0)
        triangle_w1 = torch.where(condition, w1_selected, triangle_w1)
        triangle_w2 = torch.where(condition, w2_selected, triangle_w2)

    nearby_vertices = vertices[faces[triangle_indices[0]]]  # (N, 3, 3)
    nearby_normals = vertex_normals[faces[triangle_indices[0]]]  # (N, 3, 3)

    origins = (
        nearby_vertices[..., 0, :] * triangle_w0[0, :, None]
        + nearby_vertices[..., 1, :] * triangle_w1[0, :, None]
        + nearby_vertices[..., 2, :] * triangle_w2[0, :, None]
    ).float()
    directions = -(
        nearby_normals[..., 0, :] * triangle_w0[0, :, None]
        + nearby_normals[..., 1, :] * triangle_w1[0, :, None]
        + nearby_normals[..., 2, :] * triangle_w2[0, :, None]
    ).float()

    origins = origins.reshape(uv_coords_shape[0], uv_coords_shape[1], 3)
    directions = directions.reshape(uv_coords_shape[0], uv_coords_shape[1], 3)

    # normalize the direction vector to make it a unit vector
    directions = torch.nn.functional.normalize(directions, dim=-1)

    return texture_coordinates, origins, directions


def textured_mesh(ply_path, runner):
    mesh = trimesh.load(ply_path)
    vertices = torch.from_numpy(mesh.vertices).float().cuda()
    faces = torch.from_numpy(mesh.faces).long().cuda()
    batch_size = 512
    # prepare the normals
    vertices_list = vertices.split(batch_size)
    normals = []
    for cur_vertices in vertices_list:
        cur_normals = runner.sdf_network.gradient(cur_vertices).detach()
        normals.append(cur_normals.reshape(-1, 3))
    normals = torch.cat(normals, dim=0)

    # unwrap the mesh
    texture_coordinates, origins, directions = unwrap_mesh_with_xatlas(
        vertices, faces, normals
    )

    # prepare the rendering
    face_vertices = vertices[faces]
    raylen = (
        2.0
        * torch.mean(
            torch.norm(face_vertices[:, 1, :] - face_vertices[:, 0, :], dim=-1)
        ).float()
    )
    origins = origins - 0.5 * raylen * directions
    directions = directions.reshape(-1, 3)
    origins = origins.reshape(-1, 3)
    directions_list = directions.split(batch_size)
    origins_list = origins.split(batch_size)

    # start rendering
    colors = []
    for cur_directions, cur_origins in tqdm(zip(directions_list, origins_list)):
        pose = torch.eye(4)[:3][None, ...].repeat(cur_directions.shape[0], 1, 1).cuda()
        pose[:, :3, 3] = cur_origins
        with torch.no_grad():
            nears = torch.zeros_like(cur_origins[..., 0:1])
            fars = torch.ones_like(cur_origins[..., 0:1]) * raylen
            render_out = runner.renderer.render(
                cur_origins, cur_directions, nears, fars, cos_anneal_ratio=1, eval=True
            )
        colors.append(render_out["color_fine"])
    colors = torch.cat(colors, dim=0)

    # prepare the texture image
    texture_image = (colors.cpu().numpy() * 255).reshape(1024, 1024, 3).astype(np.uint8)

    # writing objects
    import os

    output_dir = os.path.join(
        os.path.dirname(ply_path),
        f'textured_{os.path.basename(ply_path).split(".")[0]}',
    )
    os.makedirs(output_dir, exist_ok=True)
    # write png
    cv2.imwrite(os.path.join(output_dir, "material_0.png"), texture_image)
    lines_mtl = [
        "# Generated with nerfstudio",
        "newmtl material_0",
        "Ka 1.000 1.000 1.000",
        "Kd 1.000 1.000 1.000",
        "Ks 0.000 0.000 0.000",
        "d 1.0",
        "illum 2",
        "Ns 1.00000000",
        "map_Kd material_0.png",
    ]
    from pathlib import Path

    output_dir = Path(output_dir)
    lines_mtl = [line + "\n" for line in lines_mtl]
    file_mtl = open(output_dir / "material_0.mtl", "w", encoding="utf-8")  # pylint: disable=consider-using-with
    file_mtl.writelines(lines_mtl)
    file_mtl.close()

    # create the .obj file
    lines_obj = [
        "# Generated with nerfstudio",
        "mtllib material_0.mtl",
        "usemtl material_0",
    ]
    lines_obj = [line + "\n" for line in lines_obj]
    file_obj = open(output_dir / "mesh.obj", "w", encoding="utf-8")  # pylint: disable=consider-using-with
    file_obj.writelines(lines_obj)

    # write the geometric vertices
    for i in tqdm(range(len(vertices))):
        vertex = vertices[i]
        line = f"v {vertex[0]} {vertex[1]} {vertex[2]}\n"
        file_obj.write(line)

    # write the texture coordinates
    texture_coordinates_np = texture_coordinates.cpu().numpy()
    for i in tqdm(range(len(faces))):
        for uv in texture_coordinates_np[i]:
            line = f"vt {uv[0]} {1.0 - uv[1]}\n"
            file_obj.write(line)

    # write the vertex normals
    vertex_normals = normals.cpu().numpy()
    for i in tqdm(range(len(vertex_normals))):
        normal = vertex_normals[i]
        line = f"vn {normal[0]} {normal[1]} {normal[2]}\n"
        file_obj.write(line)

    # write the faces
    faces_np = faces.cpu().numpy()
    for i in tqdm(range(len(faces))):
        face = faces_np[i]
        v1 = face[0] + 1
        v2 = face[1] + 1
        v3 = face[2] + 1
        vt1 = i * 3 + 1
        vt2 = i * 3 + 2
        vt3 = i * 3 + 3
        vn1 = v1
        vn2 = v2
        vn3 = v3
        line = f"f {v1}/{vt1}/{vn1} {v2}/{vt2}/{vn2} {v3}/{vt3}/{vn3}\n"
        file_obj.write(line)

    file_obj.close()
    pass
