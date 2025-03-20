# main.py
import numpy as np
import pyshtools as pysh
import mesh_processing
import pca_align
import trimesh
import spherical_harmonics


'''
mesh_processing
'''
# 1. 读取文件
stl_path = "E:/python_spharm/flowshape/code/demo_data/refined_diamond.stl"
vertices, faces = mesh_processing.clean_mesh(stl_path)

# 2. 统一目标面数
target_faces = 20000
result = mesh_processing.decimate(vertices, faces, target_faces)
decimated_vertices = result[1]
decimated_faces = result[2]
mesh = trimesh.Trimesh(vertices=decimated_vertices, faces=decimated_faces)
trimesh.filter_laplacian(mesh, iterations=3)
decimated_vertices = mesh.vertices

# 计算 Hausdorff 距离
# hd = hausdorff_distance(vertices, decimated_vertices)
# print(f"Hausdorff 距离: {hd:.4f} mm")

# 可视化删减后面片及误差
mesh_processing.visualize_error(vertices, decimated_vertices)

# 3. 执行归一化处理
normalized_vertices = mesh_processing.normalize_mesh(decimated_vertices)

# 可视化归一化后网格
mesh_processing.visualize_normalization(normalized_vertices, decimated_faces)
align_vertices= pca_align.robust_pca_alignment(normalized_vertices, enforce_direction=True, verbose=True)

'''
mesh_processing
'''

# 4. 转换到球坐标
spherical_coords = cartesian_to_spherical(align_vertices)
R = spherical_coords[:, 0]  # 半径
theta = spherical_coords[:, 1]  # 极角
phi = spherical_coords[:, 2]  # 方位角

# 5. 执行插值（建议grid_size为偶数，满足DH网格要求）
grid_size = 256  # 可调整为所需分辨率
grid_r = spherical_interpolate(R, theta, phi, grid_size)
print(f"半径范围: {np.nanmin(grid_r):.3f} ~ {np.nanmax(grid_r):.3f}")
# visualize_interpolated(grid_r)


# 6. 计算球谐系数
if grid_r is not None:
    # 转换为pyshtools兼容的网格格式
    grid_sh = pysh.SHGrid.from_array(grid_r, grid='DH')

    # 执行球谐展开
    clm = compute_spherical_harmonics(
        grid_r,
        normalize=True,
        normalization_method='zero-component'
    )
    #print(clm)

# 7. 可视化+频谱
clm_sh = pysh.SHCoeffs.from_array(clm)
full, stat = process_spherical_harmonics(clm_sh, 'output.csv')
visualize_power_spectrum(stat, max_degree=30, log_scale=True, filename=None)

clm_array=clm_to_1d_standard(clm)
csv_filename = "spherical_coeffs1.csv"
np.savetxt(csv_filename, clm_array, delimiter=",", fmt="%.6e")
print("已保存一维球谐系数数组到", csv_filename)

# 8. 根据球谐函数展开重建表面
reconstructed_grid = clm_sh.expand(grid='DH')
clm_sh_truncated = clm_sh.pad(lmax=30)  # 只保留 30 阶以内的球谐系数
reconstructed_grid_truncated = clm_sh_truncated.expand(grid='DH')
visualize_spherical_harmonics_reconstruction(reconstructed_grid_truncated)

if __name__ == "__main__":
    main()