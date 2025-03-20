import pyvista as pv
import numpy as np
from scipy.spatial import KDTree
import igl
import trimesh


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


# Function to compute Hausdorff distance between two point clouds
def hausdorff_distance(points1, points2):
    """计算两个点云之间的 Hausdorff 距离"""
    tree1 = KDTree(points1)
    d1_to_2, _ = tree1.query(points2)
    tree2 = KDTree(points2)
    d2_to_1, _ = tree2.query(points1)
    return max(np.max(d1_to_2), np.max(d2_to_1))


# Function to visualize error between original and decimated mesh
def visualize_error(vertices, decimated_vertices):
    """可视化 Hausdorff 误差"""
    error_mesh = pv.PolyData(vertices)
    error_mesh["Distance"] = KDTree(decimated_vertices).query(vertices)[0]

    plotter = pv.Plotter()
    plotter.add_mesh(error_mesh,
                     scalars="Distance",
                     cmap="coolwarm",
                     opacity=1.0,
                     show_edges=True,
                     scalar_bar_args={"title": "error(mm)"})
    plotter.add_mesh(pv.PolyData(decimated_vertices), color="cyan", show_edges=True, opacity=0.8)
    plotter.show()


# Function to normalize mesh to unit sphere
def normalize_mesh(vertices):
    # 质心对齐
    centroid = np.mean(vertices, axis=0)
    centered_vertices = vertices - centroid

    # 单位球缩放
    max_radius = np.max(np.linalg.norm(centered_vertices, axis=1))
    normalized_vertices = centered_vertices / max_radius

    # radii = np.linalg.norm(normalized_vertices, axis=1)
    # print(f"[归一化验证] 半径范围: {radii.min():.4f} ~ {radii.max():.4f}")

    return normalized_vertices


# Function to visualize normalized mesh
def visualize_normalization(normalized_vertices, decimated_faces):
    """验证归一化"""
    pyvista_faces = np.insert(decimated_faces.astype(np.int64), 0, 3, axis=1).ravel()

    # 验证面片数组格式
    print("面片数组验证：")
    print(f"原始面片形状: {decimated_faces.shape} (应为[n_faces, 3])")
    print(f"转换后形状: {pyvista_faces.shape} (应为[n_faces*4,])")
    print(f"面片索引范围: {decimated_faces.min()}~{decimated_faces.max()} (应 < {len(normalized_vertices)})")

    # 归一化网格后可视化
    normalized_mesh = pv.PolyData(normalized_vertices, pyvista_faces)
    plotter = pv.Plotter()
    plotter.add_mesh(normalized_mesh,
                     color="lightblue",
                     show_edges=True,
                     edge_color="gray",
                     opacity=1,
                     label=f"简化网格 ({len(pyvista_faces) // 4} 面片)")
    plotter.add_axes(box_args={'color': 'red'})
    plotter.add_title(f"归一化网格验证\nvertices_number {len(normalized_vertices)}\nfaces_number: {len(pyvista_faces) // 4}")
    plotter.show()

