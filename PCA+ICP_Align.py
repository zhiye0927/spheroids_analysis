import igl
import trimesh
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import open3d as o3d
import copy

def clean_mesh(filepath):
    """
    基础清理函数
    Args:
        filepath (str): STL文件路径
    Returns:
        tuple: (清理后的顶点，清理后的面片)
    """
    # 读取原始网格
    v, f = igl.read_triangle_mesh(filepath)

    # 移除未被引用的顶点
    v, f, _, _ = igl.remove_unreferenced(v, f)

    # 处理面片索引格式问题（确保从0开始）
    if f.min() == 1:
        f = f - 1

    # 用trimesh进一步清理
    mesh = trimesh.Trimesh(vertices=v, faces=f)
    mesh.remove_infinite_values()

    # 删除退化面片
    non_deg_mask = mesh.nondegenerate_faces()
    mesh.update_faces(non_deg_mask)

    return mesh.vertices, mesh.faces


def decimate_mesh(vertices, faces, target_faces=20000):
    """
    网格简化函数
    Args:
        vertices (np.ndarray): 顶点数组
        faces (np.ndarray): 面片数组
        target_faces (int): 目标面片数（默认20000）
    Returns:
        tuple: (简化后面片，简化后顶点)
    """

    result = igl.decimate(vertices, faces, target_faces)
    v_decim = result[1]
    f_decim = result[2]
    # 有效性检查
    if f_decim.shape[0] == 0:
        raise ValueError(f"网格简化失败，当前面片数: {faces.shape[0]}，尝试降低目标面片数")

    return v_decim, f_decim


def normalize_mesh(vertices):
    # 质心对齐
    centroid = np.mean(vertices, axis=0)
    centered_vertices = vertices - centroid

    # 单位球缩放
    max_radius = np.max(np.linalg.norm(centered_vertices, axis=1))
    normalized_vertices = centered_vertices / max_radius

    radii = np.linalg.norm(normalized_vertices, axis=1)
    print(f"[归一化验证] 半径范围: {radii.min():.4f} ~ {radii.max():.4f}")

    return normalized_vertices


def visualize_normalization(normalized_vertices, decimated_faces):
    """验证归一化"""

    # 点云图带有坐标
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_normalized[:, 0], points_normalized[:, 1], points_normalized[:, 2],
               s=1, c='b', alpha=0.6)
    ax.set_title("Normalized Point Cloud")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()

    # 还原三维形状
    pyvista_faces = np.insert(decimated_faces.astype(np.int64), 0, 3, axis=1).ravel()

    # 验证面片数组格式
    print("面片数组验证：")
    print(f"原始面片形状: {decimated_faces.shape} (应为[n_faces, 3])")
    print(f"转换后形状: {pyvista_faces.shape} (应为[n_faces*4,])")
    print(f"面片索引范围: {decimated_faces.min()}~{decimated_faces.max()} (应 < {len(v_decim)})")

    # 归一化网格后可视化
    normalized_mesh = pv.PolyData(normalized_vertices, pyvista_faces)
    plotter = pv.Plotter()
    plotter.add_mesh(normalized_mesh,
                     color="lightblue",
                     show_edges=True,
                     edge_color="gray",
                     opacity=0.8,
                     label=f"简化网格 ({len(pyvista_faces) // 4} 面片)")
    plotter.add_axes(box_args={'color': 'red'})
    plotter.add_title(f"归一化网格验证\nvertices_number {len(normalized_vertices)}\nfaces_number: {len(pyvista_faces) // 4}")
    plotter.show()


def robust_pca_alignment(points, enforce_direction=True, verbose=True):
    """
    增强版PCA对齐算法 (优化修正版)

    参数说明：
    - points: (N,3) 输入点云坐标数组
    - enforce_direction: 是否强制主方向一致性
    - verbose: 显示调试信息

    返回值：
    - aligned_points: 对齐后的点云
    - rotation_matrix: 3x3旋转矩阵
    """

    # ========== 输入验证 ==========
    if not isinstance(points, np.ndarray) or points.shape[1] != 3:
        raise ValueError("输入点云必须是Nx3的NumPy数组")
    if len(points) < 3:
        raise ValueError("至少需要3个点才能计算主方向")

    # ========== 数据预处理 ==========
    # 去中心化
    centroid = np.mean(points, axis=0)
    centered = points - centroid

    # ========== 主成分计算 ==========
    # 使用SVD分解提高数值稳定性
    cov_matrix = np.cov(centered.T)
    U, s, Vt = np.linalg.svd(cov_matrix)

    # ========== 坐标系方向修正 ==========
    # 确保右手坐标系 (修正关键错误)
    rotation_matrix = Vt.T  # 注意这里取转置得到正确的旋转矩阵

    # 检查行列式符号
    if np.linalg.det(rotation_matrix) < 0:
        if verbose:
            print("检测到左手坐标系，正在修正...")
        # 通过翻转第三列保持右手系
        rotation_matrix[:, 2] *= -1

    # ========== 主方向一致性强制 ==========
    if enforce_direction:
        # 投影到主成分空间
        projected = centered @ rotation_matrix

        # 使用中位数判断方向 (比均值更鲁棒)
        if np.median(projected[:, 0]) < 0:
            if verbose:
                print("主方向翻转检测，正在校正...")
            rotation_matrix[:, 0] *= -1  # 翻转第一主成分方向

    # ========== 应用变换 ==========
    aligned_points = centered @ rotation_matrix

    # ========== 验证步骤 ==========
    if verbose:
        print("\n===== 验证报告 =====")
        print("旋转矩阵行列式:", np.linalg.det(rotation_matrix))
        print("主成分方向:")
        print(f"PC1 (X轴): {rotation_matrix[:, 0]}")
        print(f"PC2 (Y轴): {rotation_matrix[:, 1]}")
        print(f"PC3 (Z轴): {rotation_matrix[:, 2]}")
        print("对齐后坐标统计:")
        print(f"X范围: [{aligned_points[:, 0].min():.3f}, {aligned_points[:, 0].max():.3f}]")
        print(f"Y范围: [{aligned_points[:, 1].min():.3f}, {aligned_points[:, 1].max():.3f}]")
        print(f"Z范围: [{aligned_points[:, 2].min():.3f}, {aligned_points[:, 2].max():.3f}]")

    return aligned_points, rotation_matrix


def plot_pca_aligned_points(points):
    """
    绘制对齐后的点云，并在质心处绘制XYZ坐标轴
    Args:
        points (np.ndarray): 对齐后的点云数据 (N, 3)
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制点云
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               s=1, c='gray', alpha=0.6, label='Point Cloud')

    # 计算质心，并绘制质心点
    centroid = np.mean(points, axis=0)
    ax.scatter(centroid[0], centroid[1], centroid[2],
               s=50, c='k', marker='o', label='Centroid')

    # 根据点云范围确定坐标轴线的长度
    max_range = np.max(np.ptp(points, axis=0))  # 点云在各维度上的范围差
    axis_length = max_range * 1.1  # 这里设置轴长为点云范围的60%

    # 绘制X轴（红色）
    ax.plot([centroid[0] - axis_length / 2, centroid[0] + axis_length / 2],
            [centroid[1], centroid[1]],
            [centroid[2], centroid[2]],
            color='r', lw=2, label='X axis')

    # 绘制Y轴（绿色）
    ax.plot([centroid[0], centroid[0]],
            [centroid[1] - axis_length / 2, centroid[1] + axis_length / 2],
            [centroid[2], centroid[2]],
            color='g', lw=2, label='Y axis')

    # 绘制Z轴（蓝色）
    ax.plot([centroid[0], centroid[0]],
            [centroid[1], centroid[1]],
            [centroid[2] - axis_length / 2, centroid[2] + axis_length / 2],
            color='b', lw=2, label='Z axis')

    ax.set_box_aspect([1, 1, 1])

    ax.set_title("PCA Aligned Point Cloud with Coordinate Axes")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()


def load_template(template_path):
    """加载并预处理模板石球"""
    # 加载清理模板
    v_temp, f_temp = clean_mesh(template_path)

    # 简化模板
    v_temp_decim, f_temp_decim = decimate_mesh(v_temp, f_temp, 20000)

    # 归一化模板
    template_normalized = normalize_mesh(v_temp_decim)

    # PCA对齐模板
    template_aligned, _ = robust_pca_alignment(template_normalized)

    return template_aligned


def prepare_pointcloud(points):
    """将numpy数组转换为Open3D点云对象"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def multi_resolution_icp(source, target, init_transform=np.eye(4), verbose=True):
    """
    多分辨率ICP配准
    :param source: 待配准点云 (o3d.geometry.PointCloud)
    :param target: 模板点云 (o3d.geometry.PointCloud)
    :param init_transform: 初始变换矩阵
    :return: 优化后的变换矩阵
    """
    # 多尺度参数
    voxel_sizes = [0.1, 0.05, 0.02]  # 从粗到细的分辨率
    max_iterations = [200, 100, 50]

    current_transform = init_transform
    final_result = None  # 用于存储最终评估结果

    for i, (voxel_size, max_iter) in enumerate(zip(voxel_sizes, max_iterations)):
        # 降采样
        source_down = source.voxel_down_sample(voxel_size)
        target_down = target.voxel_down_sample(voxel_size)

        # 计算法线（Point-to-Plane ICP需要）
        source_down.estimate_normals()
        target_down.estimate_normals()

        # 执行ICP
        result = o3d.pipelines.registration.registration_icp(
            source_down, target_down, max_correspondence_distance=voxel_size * 2,
            init=current_transform,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=max_iter))

        current_transform = result.transformation

        final_result = o3d.pipelines.registration.registration_icp(
            source, target,  # 使用原始点云
            max_correspondence_distance=voxel_sizes[-1],  # 使用最细分辨率
            init=current_transform,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )

        if verbose:
            evaluate_icp_result(final_result)

    return final_result.transformation


def evaluate_icp_result(result):
    """增强版评估函数（添加单位校验和异常处理）"""
    try:
        print("\n=== ICP质量评估报告 ===")

        # 基础指标
        print(f"配准分数(Fitness): {result.fitness:.4f} (0-1, 1为最佳)")
        print(f"内点RMSE: {result.inlier_rmse * 1000:.2f} mm")  # 转换为毫米

        # 变换矩阵分析
        T = result.transformation
        R = T[:3, :3]
        t = T[:3, 3]

        # 旋转矩阵验证
        det = np.linalg.det(R)
        if abs(det - 1) > 1e-3:
            print(f"⚠警告: 旋转矩阵行列式异常 {det:.6f} (应接近±1)")

        # 平移量分析
        trans_norm = np.linalg.norm(t)
        print(f"平移量模长: {trans_norm * 1000:.2f} mm")
        print(f"平移向量: [{t[0] * 1000:.2f}, {t[1] * 1000:.2f}, {t[2] * 1000:.2f}] mm")

        # 条件数分析
        cond_num = np.linalg.cond(T)
        print(f"变换矩阵条件数: {cond_num:.2e}")
        if cond_num > 1e6:
            print("⚠警告: 矩阵接近奇异，配准结果不可靠")

        # 对应点数量（添加版本兼容处理）
        if hasattr(result, "correspondence_set"):
            print(f"有效对应点数量: {len(result.correspondence_set)}")
        else:
            print("对应点信息不可用（Open3D版本兼容性问题）")

    except Exception as e:
        print(f"评估过程出错: {str(e)}")


def visualize_icp_result(source, target, transform):
    """增强版可视化函数"""
    # 创建深拷贝避免污染原始数据
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)

    # 应用变换
    source_temp.transform(transform)

    # 颜色编码
    source_temp.paint_uniform_color([1, 0.706, 0])  # 金色：配准后点云
    target_temp.paint_uniform_color([0, 0.651, 0.929])  # 蓝色：目标点云

    # 创建坐标轴
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    # 带法线显示
    o3d.visualization.draw_geometries([source_temp, target_temp, coord_frame],
                                      window_name="ICP Result Verification",
                                      width=1200,
                                      height=800,
                                      point_show_normal=True)  # 显示法线方向


def align_all_to_template(template_path, stone_paths):
    """主对齐流程"""
    # 加载模板
    template_points = load_template(template_path)
    template_pcd = prepare_pointcloud(template_points)

    # 预处理模板
    template_pcd.estimate_normals()

    # 存储结果
    aligned_results = []

    for path in stone_paths:

        # 1. 预处理当前石球
        v_clean, f_clean = clean_mesh(path)
        v_decim, f_decim = decimate_mesh(v_clean, f_clean, 20000)
        points_norm = normalize_mesh(v_decim)

        # 2. PCA粗对齐
        points_pca, rot_matrix = robust_pca_alignment(points_norm)

        # 3. 转换为Open3D格式
        stone_pcd = prepare_pointcloud(points_pca)
        stone_pcd.estimate_normals()

        # 4. ICP精对齐
        transform = multi_resolution_icp(stone_pcd, template_pcd)

        # 5. 传递变换矩阵到可视化函数
        visualize_alignment(stone_pcd, template_pcd, transform)
        # visualize_icp_result(stone_pcd, template_pcd, transform)

        stone_pcd.transform(transform)

        # 6. 保存结果
        aligned_results.append(np.asarray(stone_pcd.points))

    return aligned_results


def visualize_alignment(source_pcd, target_pcd, transform):
    """改进的可视化函数（支持变换前后对比）"""
    # 创建原始点云副本
    source_original = copy.deepcopy(source_pcd)

    # 应用变换后的点云
    source_aligned = copy.deepcopy(source_pcd)
    source_aligned.transform(transform)

    # 设置颜色
    source_original.paint_uniform_color([1, 0, 0])  # 红色：对齐前
    source_aligned.paint_uniform_color([0, 0, 1])  # 蓝色：对齐后
    target_pcd.paint_uniform_color([0, 1, 0])  # 绿色：模板

    # 可视化对比
    o3d.visualization.draw_geometries(
        [source_original, source_aligned, target_pcd],
        window_name="Alignment Comparison",
        width=1200,
        height=800,
        left=200,  # 窗口位置
        top=200
    )


def visualize_icp_result(source, target, transform):
    """增强版可视化函数"""
    # 创建深拷贝避免污染原始数据
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)

    # 应用变换
    source_temp.transform(transform)

    # 颜色编码
    source_temp.paint_uniform_color([1, 0.706, 0])  # 金色：配准后点云
    target_temp.paint_uniform_color([0, 0.651, 0.929])  # 蓝色：目标点云

    # 创建坐标轴
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    # 带法线显示
    o3d.visualization.draw_geometries([source_temp, target_temp, coord_frame],
                                      window_name="ICP Result Verification",
                                      width=1200,
                                      height=800,
                                      point_show_normal=True)  # 显示法线方向


# main
# 1. 加载与清理
v_clean, f_clean = clean_mesh("E:/python_spharm/flowshape/code/demo_data/QSY_A_1098.stl")

# 2. 简化
v_decim, f_decim = decimate_mesh(v_clean, f_clean, 20000)

# 3. 最终归一化
points_normalized = normalize_mesh(v_decim)
visualize_normalization(points_normalized, f_decim )

# 4. PCA
aligned_points, rotation_matrix = robust_pca_alignment(points_normalized, enforce_direction=True, verbose=True)
plot_pca_aligned_points(aligned_points)


# main
template_path = "E:/python_spharm/flowshape/code/demo_data/QSY_A_1098.stl"
stone_paths = [
    "E:/python_spharm/flowshape/code/demo_data/QSY_A_0252.stl",
    "E:/python_spharm/flowshape/code/demo_data/QSY_A_2797.stl"
]
aligned_data = align_all_to_template(template_path, stone_paths)
