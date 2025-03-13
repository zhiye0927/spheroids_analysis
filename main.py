# main.py
import numpy as np
import pyshtools as pysh
from mesh_processing import clean_mesh, normalize_mesh, decimate_mesh
from spherical_harmonics import (
    spherical_interpolate,
    compute_spherical_harmonics,
    convert_clm_to_csv
)
from visualization import (
    visualize_error,
    visualize_normalization,
    plot_reconstruction_comparison
)


def main():
    # ========== 配置参数 ==========
    config = {
        'stl_path': "demo_data/QSY_A_2797.stl",
        'target_faces': 20000,
        'grid_size': 256,
        'show_intermediate': True
    }

    # ========== 数据预处理 ==========
    # 1. 清理网格
    vertices, faces = clean_mesh(config['stl_path'])

    # 2. 网格简化
    decimated_vertices, decimated_faces = decimate_mesh(
        vertices, faces, config['target_faces']
    )

    # 3. 归一化处理
    normalized_vertices = normalize_mesh(decimated_vertices)

    if config['show_intermediate']:
        visualize_normalization(normalized_vertices, decimated_faces)

    # ========== 球谐分析 ==========
    # 4. 转换到球坐标系
    spherical_coords = cartesian_to_spherical(normalized_vertices)
    R, theta, phi = spherical_coords[:, 0], spherical_coords[:, 1], spherical_coords[:, 2]

    # 5. 执行插值
    grid_r = spherical_interpolate(R, theta, phi, config['grid_size'])

    # 6. 计算球谐系数
    clm = compute_spherical_harmonics(
        grid_r,
        normalize=True,
        normalization_method='zero-component'
    )

    # 7. 保存结果
    harmonics_df = convert_clm_to_csv(clm)
    harmonics_df.to_csv("spherical_harmonics.csv", index=False)

    # ========== 重建验证 ==========
    # 8. 执行重建
    clm_sh = pysh.SHCoeffs.from_array(clm)
    full_recon = clm_sh.expand(grid='DH')
    truncated_recon = clm_sh.pad(lmax=10).expand(grid='DH')

    # 9. 可视化对比
    plotter = plot_reconstruction_comparison(
        full_recon.data,
        truncated_recon.data
    )
    plotter.show()


if __name__ == "__main__":
    main()