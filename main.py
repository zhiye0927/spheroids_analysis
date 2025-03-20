import numpy as np
import pyshtools as pysh
import mesh_processing
import pca_align
import visualization
import trimesh
import spherical_harmonics
import igl
from trimesh.smoothing import filter_laplacian


def main():
    '''
    mesh_processing
    '''
    # 1. 读取文件
    stl_path = "E:/python_spharm/flowshape/code/demo_data/diamond_cartoon.stl"
    vertices, faces = mesh_processing.clean_mesh(stl_path)

    # 2. 统一目标面数
    target_faces = 20000
    result = igl.decimate(vertices, faces, target_faces)
    decimated_vertices = result[1]
    decimated_faces = result[2]
    mesh = trimesh.Trimesh(vertices=decimated_vertices, faces=decimated_faces)
    filter_laplacian(mesh, iterations=3)
    decimated_vertices = mesh.vertices

    # 可视化删减后面片及误差
    mesh_processing.visualize_error(vertices, decimated_vertices)

    # 3. 执行归一化处理
    normalized_vertices = mesh_processing.normalize_mesh(decimated_vertices)

    # 可视化归一化后网格
    mesh_processing.visualize_normalization(normalized_vertices, decimated_faces)
    align_vertices = pca_align.robust_pca_alignment(normalized_vertices, enforce_direction=True, verbose=True)

    '''
    mesh_processing
    '''
    # 4. 转换到球坐标
    spherical_coords = spherical_harmonics.cartesian_to_spherical(align_vertices)
    R = spherical_coords[:, 0]  # 半径
    theta = spherical_coords[:, 1]  # 极角
    phi = spherical_coords[:, 2]  # 方位角

    # 5. 执行插值（建议grid_size为偶数，满足DH网格要求）
    grid_size = 256  # 可调整为所需分辨率
    grid_r = spherical_harmonics.spherical_interpolate(R, theta, phi, grid_size)
    print(f"半径范围: {np.nanmin(grid_r):.3f} ~ {np.nanmax(grid_r):.3f}")

    # 6. 计算球谐系数
    if grid_r is not None:
        grid_sh = pysh.SHGrid.from_array(grid_r, grid='DH')
        clm = spherical_harmonics.compute_spherical_harmonics(
            grid_r,
            normalize=True,
            normalization_method='zero-component'
        )

    # 7. 可视化+频谱
    clm_sh = pysh.SHCoeffs.from_array(clm)
    full, stat = spherical_harmonics.process_spherical_harmonics(clm_sh, 'output.csv')
    visualization.visualize_power_spectrum(stat, max_degree=30, log_scale=True, filename=None)

    # 保存球谐系数
    clm_array = spherical_harmonics.clm_to_1d_standard(clm)
    csv_filename = "spherical_coeffs1.csv"
    np.savetxt(csv_filename, clm_array, delimiter=",", fmt="%.6e")
    print("已保存一维球谐系数数组到", csv_filename)

    # 8. 根据球谐展开重建表面
    clm_sh_truncated = clm_sh.pad(lmax=30)
    reconstructed_grid_truncated = clm_sh_truncated.expand(grid='DH')
    visualization.visualize_spherical_harmonics_reconstruction(reconstructed_grid_truncated)


# 确保此脚本作为主程序运行时才执行 main()
if __name__ == "__main__":
    main()

