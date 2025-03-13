import igl
import trimesh
import numpy as np


# Function to clean and preprocess the mesh from an STL file
def clean_mesh(filepath):
    """加载并清理 STL 网格"""
    # 读取网格数据
    v, f = igl.read_triangle_mesh(filepath)

    # 移除未被面片引用的顶点
    v, f, _, _ = igl.remove_unreferenced(v, f)

    # 处理异常索引
    if f.min() == 1:
        f = f - 1

    # 构建 trimesh 进行进一步清理
    mesh = trimesh.Trimesh(vertices=v, faces=f)
    mesh.remove_infinite_values()

    # 删除退化面片
    mask = mesh.nondegenerate_faces().astype(bool)
    mesh.update_faces(mask)

    return mesh.vertices, mesh.faces


# Function to obtain reduced vertices and faces
def decimate_mesh(vertices, faces, target_faces):
    """网格简化至指定面片数"""
    result = igl.decimate(vertices, faces, target_faces)
    return result[1], result[2]  # 返回简化后的顶点和面


# Function to normalize mesh to unit sphere
def normalize_mesh(vertices):
    """标准化过程-质心对齐和缩放至单位球"""
    # 质心对齐
    centroid = np.mean(vertices, axis=0)
    centered_vertices = vertices - centroid

    # 单位球缩放
    max_radius = np.max(np.linalg.norm(centered_vertices, axis=1))
    normalized_vertices = centered_vertices / max_radius

    # radii = np.linalg.norm(normalized_vertices, axis=1)
    # print(f"[归一化验证] 半径范围: {radii.min():.4f} ~ {radii.max():.4f}")

    return normalized_vertices

